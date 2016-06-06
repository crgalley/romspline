import numpy as np, matplotlib.pyplot as plt
from scipy.stats.mstats import mquantiles

# Modules for parallelization
try:
  from concurrent.futures import ProcessPoolExecutor, wait, as_completed
  from multiprocessing import cpu_count
  _parallel = True
except:
  print "Cannot import from concurrent.futures and/or multiprocessing modules."
  _parallel = False

# Module for building reduced-order splines
import greedySpline


########################
### Helper functions ###
########################

def Linfty(x, y, arg=False):
  """L-infinity norm of x - y"""
  diff = np.abs(x-y)
  max_diff = np.max(diff)
  if arg:
    return max_diff, np.argmax(diff)
  else:  
    return max_diff


def partitions(n, K):
  """Split array with n samples into K (nearly) equal partitions"""
  assert n >= K, "Number of partitions must not exceed number of samples."
  return np.asarray( np.array_split(np.arange(n), K) )


def random_partitions(n, K):
  """Split array with n samples into K (nearly) equal partitions of non-overlapping random subsets"""
  assert n >= K, "Number of folds must not exceed number of samples."
  
  # Make array with unique random integers
  rand = np.random.choice(range(n), n, replace=False)
  
  # Split into K (nearly) equal partitions
  return [np.sort(rand[pp]) for pp in partitions(n, K)]


###################################
###  Classes for assessing the  ###
### reduced-order spline errors ###
###################################


#########################
# Class for convergence #
#  of greedy algorithm  #
#########################

class Convergence(object):
  
  
  def __init__(self, spline=None, tol=1e-6, rel=False, deg=5):
    self._tol = tol
    self._rel = rel
    self._deg = deg
    
    self.splines = dict()
    self.errors = dict()
    self.compressions = dict()
    
    if spline is None:
      self._spline = False
    else:
      self._spline = spline
    self._made = False
  
  
  def __call__(self, x, y, levs=None):
    return self.make(x, y, levs=levs)
  
  
  def make(self, x, y, levs=None):
    
    if levs is None:
      self.levs = np.array([8, 4, 2])
    else:
      self.levs = np.array(levs)
    if 1 not in self.levs:
      self.levs = np.hstack([self.levs, 1])
    
    if self._spline:
      self.splines[1] = self._spline
    else:
      self.splines[1] = greedySpline.ReducedOrderSpline(x, y, deg=self._deg, tol=self._tol, rel=self._rel)
    
    for ll in self.levs:
      
      if ll != 1:
        x_lev = x[::ll]
        y_lev = y[::ll]
        if x[-1] not in x_lev:
          x_lev = np.hstack([x_lev, x[-1]])
          y_lev = np.hstack([y_lev, y[-1]])
            
        self.splines[ll] = greedySpline.ReducedOrderSpline(x_lev, y_lev, deg=self._deg, tol=self._tol, rel=self._rel)
      self.errors[ll] = Linfty(self.splines[ll].eval(x), y)
      self.compressions[ll] = self.splines[ll].compression
    
    self._made = True
  
  
  def _plot_greedy_errors(self, ax, axes=None):
    
    #plot = ax
    if axes is None:
      axes = 'semilogy'
    if axes == 'semilogy':
      [ax.semilogy(self.splines[ll].errors) for ll in self.levs]
    elif axes == 'semilogx':
      [ax.semilogx(self.splines[ll].errors) for ll in self.levs]
    elif axes == 'loglog':
      [ax.loglog(self.splines[ll].errors) for ll in self.levs]
    elif axes == 'plot':
      [ax.plot(self.splines[ll].errors) for ll in self.levs]
    
    return ax
  
  
  def plot_greedy_errors(self, ax=None, show=True, axes=None, xlabel='Size of reduced data', ylabel='Greedy errors'):
    
    if self._made:
      if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
      
      ax = self._plot_greedy_errors(ax, axes=axes)
      ax.set_xlabel(xlabel)
      ax.set_ylabel(ylabel)
      
      if show:
        plt.show()
      else:
        if ax is None:
          return fig, ax
        else:
          return ax
      
    else:
      print "No data to plot. Run make method."
  
  
  def _plot_Linfty_errors(self, ax):
    
    [ax.loglog(ll, self.errors[ll], 'o') for ll in self.levs]
    
    return ax
  
  
  def plot_Linfty_errors(self, ax=None, show=True, xlabel='Decimation factor', ylabel='Max spline errors'):
    if self._made:
      
      if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
      
      #[ax.loglog(ll, self.errors[ll], 'o') for ll in self.levs]
      ax = self._plot_Linfty_errors(ax)
      ax.set_xlim(self.levs.min(), 1.1*self.levs.max())
      ax.set_xlabel(xlabel)
      ax.set_ylabel(ylabel)
      
      if show:
        plt.show()
      else:
        if ax is None:
          return fig, ax
        else:
          return ax
        
    else:
      print "No data to plot. Run make method."
  
  
  def _plot_sizes(self, ax):
    
    [ax.loglog(ll, self.splines[ll].size, 'o') for ll in self.levs]
    
    return ax
  
  
  def plot_sizes(self, ax=None, show=True, xlabel='Decimation factor', ylabel='Reduced data sizes'):
    if self._made:
      
      if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
      
      ax = self._plot_sizes(ax)
      ax.set_xlim(self.levs.min(), 1.1*self.levs.max())
      ax.set_xlabel(xlabel)
      ax.set_ylabel(ylabel)
      
      if show:
        plt.show()
      else:
        if ax is None:
          return fig, ax
        else:
          return ax
        
    else:
      print "No data to plot. Run make method."


######################################
# Random seeds for greedy algorithm  #
######################################

def _randomSeeds(x, y, tol=1e-6, rel=False, deg=5):
  """Build reduced-order splines from a random set of seeds for greedy algorithm"""
  # Seeds
  seeds = np.sort( np.random.choice(range(len(x)), deg+1, replace=False) )
  
  # Build spline
  spline = greedySpline.ReducedOrderSpline(x, y, tol=tol, rel=rel, deg=deg, seeds=seeds)
  
  # Return last greedy error, size, seeds, and reduced data samples
  return spline.errors[-1], spline.size, seeds, spline.X


class RandomSeeds(object):
  
  def __init__(self, tol=1e-6, rel=False, deg=5):
    self._tol = tol
    self._rel = rel
    self._deg = deg
    self._made = False
  
  
  def __call__(self, x, y, Nseeds):
    return self.make(x, y, Nseeds)
  
  
  def make(self, x, y, Nseeds, parallel=True):
    
    # Allocate some memory
    self.errors = np.zeros(Nseeds, dtype='double')
    self.sizes = np.zeros(Nseeds, dtype='double')
    self.seeds = np.zeros((Nseeds, self._deg+1), dtype='double')
    self.Xs = []
    
    if parallel is False:
      for nn in range(Nseeds):
        self.errors[nn], self.sizes[nn], self.seeds[nn] = _randomSeeds(x, y, tol=self._tol, rel=self._rel, deg=self._deg)
    else:
      # Determine the number of processes to run on
      if parallel is True:
        try:
          numprocs = cpu_count()
        except NotImplementedError:
          numprocs = 1
      else:
        numprocs = parallel
      
      # Create a parallel process executor
      executor = ProcessPoolExecutor(max_workers=numprocs)
      
      output = []
      for nn in range(Nseeds):
        output.append(executor.submit(_randomSeeds, x, y, tol=self._tol, rel=self._rel, deg=self._deg))
      
      # Gather the results as they complete
      for ii, oo in enumerate(as_completed(output)):
        self.errors[ii], self.sizes[ii], self.seeds[ii], Xs = oo.result()
        self.Xs.append(Xs)
        
    self._made = True
  
  
  def plot_sizes(self, n=20, ax=None, show=True, xlabel='Size of reduced data'):
    if self._made:
      
      if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
      
      ax.hist(self.sizes, n, color='k', alpha=0.33);
      
      mean = np.mean(self.sizes)
      std = np.std(self.sizes)
      
      ax.axvline(x=mean, color='k', linewidth=2)
      ax.axvline(x=mean+std, color='k', linestyle='--', linewidth=1)
      ax.axvline(x=mean-std, color='k', linestyle='--', linewidth=1)
      
      ax.set_xlabel(xlabel)
      
      if show:
        plt.show()
      else:
        if ax is None:
          return fig, ax
        else:
          return ax
    
    else:
      print "No data to plot. Run make method or set _made attribute to True."
    
  
  def plot_samples(self, x, y, ax=None, show=True, xlabel='$x$', ylabel='Size of reduced data, $n$'):
    if self._made:
      if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
      
      # Assemble all reduced samples into a 2d array
      longest = max(map(len, self.Xs))
      Xs = max(x)*np.ones((len(self.Xs),longest), dtype='double')
      for ii, kk in enumerate(self.Xs):
        Xs[ii][:len(kk)] = kk
        
      # Plot ranges of selected samples for each iteration
      mins, maxes = [], []
      for cc in Xs.T:
        mins.append(np.min(cc))
        maxes.append(np.max(cc))
      ax.plot(mins, np.arange(len(mins)), 'k-', maxes, np.arange(len(maxes)), 'k-')
      
      # Shade between the min and max number of reduced samples
      spline_mins = greedySpline.UnivariateSpline(np.sort(mins), np.arange(len(mins)), s=0)
      ax.fill_between(np.sort(maxes), spline_mins(np.sort(maxes)), np.arange(len(maxes)), facecolor='k', alpha=0.33)
      
      ax.set_xlabel(xlabel)
      ax.set_ylabel(ylabel)
      
      if show:
        plt.show()
      else:
        if ax is None:
          return fig, ax
        else:
          return ax
    
    else:
      print "No data to plot. Run make method or set _made attribute to True."
  

###########################
# K-fold cross validation #
###########################

def _Kfold(x, y, K, i_fold, folds, tol=1e-6, rel=False, deg=5, seeds=None):
  """K-fold cross validation on a given partition (fold)"""
  
  # Assemble training data excluding the i_fold'th partition for validation
  folds = np.asarray(folds)
  complement = np.ones(len(folds), dtype='bool')
  complement[i_fold] = False
  f_trial = np.sort( np.concatenate([ff for ff in folds[complement]]) )
  fold = folds[i_fold]

  # Form trial data
  x_trial = x[f_trial]
  y_trial = y[f_trial]
  
  # Build trial spline
  spline = greedySpline._greedy(x_trial, y_trial, tol=tol, rel=rel, deg=deg, seeds=seeds)[0]
  
  # Compute L-infinity errors between trial and data in validation partition
  error, arg_error = Linfty(spline(x[fold]), y[fold], arg=True)
  rel_error = error / np.abs(y[fold][arg_error])
  
  return fold[arg_error], error, rel_error


class CrossValidation(object):
  
  def __init__(self, tol=1e-6, rel=False, deg=5):
    self._tol = tol
    self._rel = rel
    self._deg = deg
    self._made = False
  
  
  def __call__(self, x, y, K=10, parallel=True):
    return self.Kfold(x, y, K=K, parallel=parallel)
  
  
  def Kfold(self, x, y, K=10, parallel=True, seeds=None, random=True):
    """K-fold cross-validation"""
    assert len(x) == len(y), "Expecting input data to have same length."
    
    self._K = int(K)
    self._size = len(x)
    
    # Allocate some memory
    self.arg_errors = np.zeros(K, dtype='int')
    self.errors = np.zeros(K, dtype='double')
    self.rel_errors = np.zeros(K, dtype='double')
    
    # Divide into K random or nearly equal partitions
    if random:
      self._partitions = random_partitions(self._size, self._K)
    else:
      self._partitions = partitions(self._size, self._K)
    
    if parallel is False:
      for ii in range(self._K):
        self.arg_errors[ii], self.errors[ii], self.rel_errors[ii] = _Kfold(x, y, self._K, ii, self._partitions, tol=self._tol, rel=self._rel, deg=self._deg, seeds=seeds)
    else:
      # Determine the number of processes to run on
      if parallel is True:
        try:
          numprocs = cpu_count()
        except NotImplementedError:
          numprocs = 1
      else:
        numprocs = parallel
      
      # Create a parallel process executor
      executor = ProcessPoolExecutor(max_workers=numprocs)
      
      # Compute the spline errors on each validation partition
      output = []
      for ii in range(self._K):
        output.append( executor.submit(_Kfold, x, y, self._K, ii, self._partitions, tol=self._tol, rel=self._rel, deg=self._deg, seeds=seeds) )
      
      # Gather the results as they complete
      for ii, ee in enumerate(as_completed(output)):
        self.arg_errors[ii], self.errors[ii], self.rel_errors[ii] = ee.result()
    
    self._made = True
  
  
  def Kfold_ensemble(self, x, y, n, K=10, parallel=True, seeds=None, random=True, verbose=False):
    """Perform a number n of K-fold cross-validations"""
    self.ens_arg_errors, self.ens_errors, self.ens_rel_errors = [], [], []
    for nn in range(n):
      if verbose and not (nn+1)%10:
        print "Trials completed:", nn+1
      self.Kfold(x, y, K=K, parallel=parallel, seeds=seeds, random=random)
      self.ens_arg_errors.append( self.arg_errors )
      self.ens_errors.append( self.errors )
      self.ens_rel_errors.append( self.rel_errors )
  
  
  def ensemble_stats(self, lq=0.05, uq=0.95, rel=False):
    """Compute mean, median, and two quantiles of ensemble of mean errors"""
    # Compute means of ensemble errors
    if rel:
      error_means = map(np.mean, self.ens_rel_errors)
    else:
      error_means = map(np.mean, self.ens_errors)
    
    # Mean of error means and sample standard deviation
    mean = np.mean(error_means)
    std = np.std(error_means, ddof=1)

    # Specific quantiles of error means
    lower, median, upper = mquantiles(error_means, prob=[lq, 0.5, uq])
    
    return mean, std, [lower, median, upper]
  
  
  def plot_ensemble_errors(self, lq=0.05, uq=0.95, n=20, ax=None, rel=False, show=True):
    
    if self._made:
      if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
      
      if rel:
        y_plot = map(np.mean, self.ens_rel_errors)
        ax.set_xlabel('Mean relative spline errors')
      else:
        y_plot = map(np.mean, self.ens_errors)
        ax.set_xlabel('Mean absolute spline error')
      
      lower, median, upper = mquantiles(y_plot, prob=[lq, 0.5, uq])
      
      # Plot histogram of mean errors
      ax.hist(np.log10(y_plot), n, color='k', alpha=0.33)
      
      # Plot mean (blue) and median (red) of mean errors
      ax.axvline(np.log10(np.mean(y_plot)), 0, 1, color='b', label='Mean')
      ax.axvline(np.log10(median), 0, 1, color='r', label='Median')
      
      # Plot lq and uq quantiles (dashed red)
      ax.axvline(np.log10(upper), 0, 1, color='r', linestyle='--', label=str(int(uq*100))+'th quantile')
      ax.axvline(np.log10(lower), 0, 1, color='r', linestyle='--', label=str(int(lq*100))+'th quantile')
      
      ax.set_xlim(np.log10(0.95*np.min(y_plot)), np.log10(1.05*np.max(y_plot)))
      ax.legend(loc='upper right', prop={'size':10})
      
      if show:
        plt.show()
      else:
        if ax is None:
          return fig, ax
        else:
          return ax
      
    else:
      print "No data to plot. Run Kfold_ensemble."
      
  
  def plot_partition_errors(self, x=None, ax=None, rel=False, show=True, excl=False):
    
    if self._made:
      if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
      
      if x is not None:
        x_plot = np.array([np.mean(x[pp]) for pp in self._partitions])
        ax.set_xlabel('$x$ partition')
      else:
        x_plot = range(len(self._partitions))
        ax.set_xlabel('Partition')
      
      if rel:
        y_plot = self.rel_errors
        ax.set_ylabel('Relative L-infinity spline error')
      else:
        y_plot = self.errors
        ax.set_ylabel('$L_\\infty$ spline error')
      
      # Exclude boundary pieces, if req'd, since spline is extrapolating there
      if excl:
        ax.plot(x_plot[1:-1], y_plot[1:-1], 'ko-')
      else:
        ax.plot(x_plot, y_plot, 'ko-')
      
      ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
      
      if show:
        plt.show()
      else:
        if ax is None:
          return fig, ax
        else:
          return ax
      
    else:
      print "No data to plot. Run Kfold method."
    
    
  def plot_Linfty_errors(self, n=20, ax=None, rel=False, show=True, excl=False):
    
    if self._made:
      if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
      
      if rel:
        data = np.log10(self.rel_errors)
        ax.set_xlabel('$\\log_{10}$(Relative L-infinity spline error)')
      else:
        data = np.log10(self.errors)
        ax.set_xlabel('$\\log_{10}$(L-infinity spline error)')
      
      # Exclude boundary pieces, if req'd, since spline is extrapolating there
      if excl:
        ax.hist(data[1:-1], n, color='k', alpha=0.33)
      else:
        ax.hist(data, n, color='k', alpha=0.33)
      
      if show:
        plt.show()
      else:
        if ax is None:
          return fig, ax
        else:
          return ax
      
    else:
      print "No data to plot. Run Kfold."


####################################
#    Estimate uncertainty of a     #
# reduced-order spline interpolant #
####################################

class QuantifyUncertainty(object):
  # FIXME: Add documentation.
  # FIXME: Test
  
  def __init__(self, x, y, X=None, Y=None, tol=1e-6, deg=5, num=10, K=10, levs=None, prob=[0.05, 0.95]):
    
    assert len(prob) == 2, "Expecting two quantiles to fill between for plotting."
    
    # Build fiducial reduced-order spline (ROS) interpolant
    print "Building reduced-order spline interpolant..."
    self.spline = greedySpline.ReducedOrderSpline(x, y, tol=tol, deg=deg)
    print "  Size of reduced data =", self.spline.size
    print "  Compression factor = ", self.spline.compression
    
    # Check convergence of fiducial ROS
    print "Checking convergence..."
    self.convergence = Convergence(tol=tol, deg=deg)
    self.convergence.make(x, y, levs=levs)
    
    levs = np.sort(self.convergence.levs)
    frac12 = abs(1.-float(self.convergence.splines[levs[1]].size)/self.convergence.splines[levs[0]].size )
    frac23 = abs(1.-float(self.convergence.splines[levs[2]].size)/self.convergence.splines[levs[1]].size )
    if abs(frac12) <= 0.1:
      _convergence = 0
      if abs(frac23) <= 0.1:
        _convergence = 1
    else:
      _convergence = -1
    
    if _convergence == 1:
      print "  Reduced data is convergent."
    elif _convergence == -1:
      print "  Reduced data does not appear to be convergent."
    elif _convergence == 0:
      print "  Reduced data may or may not be convergent."
    
    # K-fold cross-validation ensemble studies
    print "K-fold cross-validation..."
    self.crossvalidation = CrossValidation(tol=tol, deg=deg)
    self.crossvalidation.Kfold_ensemble(x, y, n=num, K=K, verbose=True)
    quantiles = mquantiles(self.crossvalidation.ens_errors, prob=prob)
    quantiles_max = mquantiles(map(np.max, self.crossvalidation.ens_errors), prob=prob)
    # FIXME: Determine if CV errors are consistent with spline tolerance
    # FIXME: Print to screen the result of the line above
    
    ### Plot results
    self.fig, self.ax = plt.subplots(nrows=2, ncols=2, figsize=[10,10])
    
    # Plot greedy errors for each convergence trial spline
    self.ax[0,0] = self.convergence._plot_greedy_errors(self.ax[0,0], axes='loglog')
    self.ax[0,0].set_xlabel('Size of reduced data')
    self.ax[0,0].set_ylabel('Greedy error')
    
    #
    self.ax[0,1] = self.convergence._plot_Linfty_errors(self.ax[0,1])
    self.ax[0,1].set_xlabel('Decimation factor')
    self.ax[0,1].set_ylabel('Last greedy error')
    
    # Plot cross-validation errors as histogram
    flat_errors = np.hstack(self.crossvalidation.ens_errors)
    self.ax[1,0].hist(np.log10(flat_errors), 20, color='k', alpha=0.33)
    self.ax[1,0].axvline(np.log10(np.mean(flat_errors)), 0, 1, color='b', linestyle='-')
    self.ax[1,0].axvline(np.log10(quantiles[0]), 0, 1, color='b', linestyle='--')
    self.ax[1,0].axvline(np.log10(quantiles[1]), 0, 1, color='b', linestyle='--')
    flat_errors_max = np.hstack(map(np.max, self.crossvalidation.ens_errors))
    self.ax[1,0].axvline(np.log10(np.mean(flat_errors_max)), 0, 1, color='r', linestyle='-')
    self.ax[1,0].axvline(np.log10(quantiles_max[0]), 0, 1, color='r', linestyle='--')
    self.ax[1,0].axvline(np.log10(quantiles_max[1]), 0, 1, color='r', linestyle='--')
    self.ax[1,0].axvline(np.log10(tol), 0, 1, color='k', linestyle='--')
    #self.ax[1,0].axvline(np.log10(np.median(flat_errors)), 0, 1, color='k', linestyle='-')
    self.ax[1,0].set_xlabel('$\\log_{10}$(Validation errors)')
    self.ax[1,0].set_ylabel('Occurrence')
    
    # Plot (L-infinity) spline errors along with K-fold
    # cross-validation error estimates (and uncertainties)
    ones = np.ones(len(x))
    if X is None and Y is None:
      X = x
      Y = y
    assert len(X) == len(Y), "Resampled x and y elements must have same length."
    
    delta = np.abs(Y-self.spline(X))
    self.ax[1,1].semilogy(X, delta, 'k-')
    self.ax[1,1].fill_between(x, quantiles[0]*ones, quantiles[1]*ones, facecolor='b', alpha=0.33)
    self.ax[1,1].semilogy(x, np.mean(self.crossvalidation.ens_errors)*ones, 'b-')
    self.ax[1,1].fill_between(x, quantiles_max[0]*ones, quantiles_max[1]*ones, facecolor='r', alpha=0.33)
    self.ax[1,1].semilogy(x, np.mean(map(np.max, self.crossvalidation.ens_errors))*ones, 'r-')
    self.ax[1,1].semilogy(x, np.max(self.crossvalidation.ens_errors)*ones, 'g-')
    self.ax[1,1].semilogy(x, tol*ones, 'k--')
    self.ax[1,1].set_ylim(0.1*np.max(delta), 1.1*np.max(self.crossvalidation.ens_errors))
    self.ax[1,1].set_xlim(min(X), max(X))
    self.ax[1,1].set_xlabel('$x$')
    self.ax[1,1].set_ylabel('Absolute error');
