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

import greedy  # For building reduced-order splines


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
  """Split array with n samples into K (nearly) equal 
  partitions of non-overlapping random subsets
  """
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
  """A class for determining if the reduced-order spline greedy algorithm is
  convergent. 
  
  The original data set is decimated and reduced-order splines
  are built for each decimation. Convergence is met if the size of the
  reduced dataset is essentially unchanged for consecutive decimations.
  """
  
  def __init__(self, spline=None, tol=1e-6, rel=False, deg=5):
    """Create a Convergence object.
    
    Input
    -----
      spline -- a greedy.ReducedOrderSpline class object
                (default is None)
      tol    -- L-infinity error tolerance for reduced-order spline 
                (default 1e-6)
      rel    -- L-infinity error tolerance is relative to max abs of data?
                (default False)
      deg    -- degree of interpolating polynomials 
                (default 5)
    """
    
    # Initialize some variables
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
      # Add self._made = True?
    self._made = False
  
  
  def __call__(self, x, y, levs=None):
    """Decimate the input data at the required levs and compute the 
    resulting reduced-order spline interpolants.
    
    Input
    -----
      x    -- samples
      y    -- data to be downsampled
      levs -- decimation levels (levs) such that if levs=4 then 
              every fourth entry is skipped for building the spline
              (default None)
    
    Attributes
    ----------
      splines      -- reduced-order spline objects for each lev
      errors       -- L-infinity spline errors for each lev as
                      compared to the full input data
      compressions -- compression factors of the reduced data
                      for each lev
    """
    return self.make(x, y, levs=levs)
  
  
  def make(self, x, y, levs=None):
    """Decimate the input data at the required levs and compute the 
    resulting reduced-order spline interpolants.
    
    Input
    -----
      x    -- samples
      y    -- data to be downsampled
      levs -- decimation levels (levs) such that if levs=4 then 
              every fourth entry is skipped for building the spline
              (default None)
    
    Attributes
    ----------
      splines      -- reduced-order spline objects for each lev
      errors       -- L-infinity spline errors for each lev as
                      compared to the full input data
      compressions -- compression factors of the reduced data
                      for each lev
    """
    if levs is None:
      self.levs = np.array([8, 4, 2])
    else:
      self.levs = np.array(levs)
    if 1 not in self.levs:
      self.levs = np.hstack([self.levs, 1])
    
    # Build reduced-order spline on full data (e.g., lev=1)
    if self._spline:
      self.splines[1] = self._spline
    else:
      self.splines[1] = greedy.ReducedOrderSpline(x, y, deg=self._deg, tol=self._tol, rel=self._rel)
    
    # Build reduced-order splines for each requested decimation level
    for ll in self.levs:
      
      # Build a reduced_order spline
      if ll != 1:
        x_lev = x[::ll]
        y_lev = y[::ll]
        if x[-1] not in x_lev:
          x_lev = np.hstack([x_lev, x[-1]])
          y_lev = np.hstack([y_lev, y[-1]])
            
        self.splines[ll] = greedy.ReducedOrderSpline(x_lev, y_lev, deg=self._deg, tol=self._tol, rel=self._rel)
      
      # Compute the compression factor and L-infinity absolute error for this lev
      self.errors[ll] = Linfty(self.splines[ll].eval(x), y)
      self.compressions[ll] = self.splines[ll].compression
    
    self._made = True
  
  
  def _plot_greedy_errors(self, ax, axes=None):
    """Simple function that selects the plotting axes type
    
    Input
    -----
      ax   -- plot object
      axes -- type of axes to plot
              (default None)
    
    Output
    ------
      plot object
    """
    
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
  
  
  def plot_greedy_errors(self, ax=None, show=True, axes='semilogy', xlabel='Size of reduced data', ylabel='Greedy errors'):
    """Plot the greedy errors versus size of the reduced data.
    
    Input
    -----
      ax     -- matplotlib plot/axis object
      show   -- display the plot?
                (default True)
      axes   -- axis scales for plotting
                (default 'semilogy')
      xlabel -- label of x-axis
                (default 'Size of reduced data')
      ylabel -- label of y-axis
                (default 'Greedy errors')
    
    Output
    ------
      If show=True then the plot is displayed.
      Otherwise, the matplotlib plot/axis object is output.
    """
    
    if self._made:  # Check if spline data made
      if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
      
      ax = self._plot_greedy_errors(ax, axes=axes)
      ax.set_xlabel(xlabel)
      ax.set_ylabel(ylabel)
      
      if show:  # Display the plot
        plt.show()  
      else:     # Otherwise, return plot objects for editing the plot in the future
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
  """Build reduced-order splines from a random set of seeds for greedy algorithm.
  See RandomSeeds for further details.
  
  Input
  -----
    x   -- samples
    y   -- data to be downsampled
    tol -- L-infinity error tolerance for reduced-order spline 
           (default 1e-6)
    rel -- L-infinity error tolerance is relative to max abs of data?
           (default False)
    deg -- degree of interpolating polynomials 
           (default 5)
  """
  # Seeds
  seeds = np.sort( np.random.choice(range(len(x)), deg+1, replace=False) )
  
  # Build spline
  spline = greedy.ReducedOrderSpline(x, y, tol=tol, rel=rel, deg=deg, seeds=seeds)
  
  # Return last greedy error, size, seeds, and reduced data samples
  return spline.errors[-1], spline.size, seeds, spline.X


class RandomSeeds(object):
  """A class for studying how the initial distribution of 
  seed points effects the size of the reduced data set.
  
  The default distribution of seed points depends on the
  degree (deg) of the spline polynomial used for interpolation.
  In general, one needs deg+1 points to fit a polynomial
  of degree deg. The default seed points include the first
  and last points of the data and then deg-1 points that
  are as equally spaced as possible.
  
  In this class, a set of deg+1 seed points are randomly
  selected and a reduced-order spline is generated for
  each such seed selection. The resulting sizes of the
  reduced data are recorded.
  """
  
  def __init__(self, tol=1e-6, rel=False, deg=5):
    """Create a RandomSeeds object.
    
    Input
    -----
      tol -- L-infinity error tolerance for reduced-order spline 
             (default 1e-6)
      rel -- L-infinity error tolerance is relative to max abs of data?
             (default False)
      deg -- degree of interpolating polynomials 
             (default 5)
    """
    self._tol = tol
    self._rel = rel
    self._deg = deg
    self._made = False  # Assume that random seeds study is not made yet
  
  
  def __call__(self, x, y, Nseeds, parallel=True):
    """Make reduced-order splines for Nseeds number of random sets of seed points.
    
    Input
    -----
      x        -- samples
      y        -- data to be downsampled
      Nseeds   -- number of random sets of seed points
      parallel -- parallelize the computation for each random seed set?
                  (default True)
    
    Attributes
    ----------
      errors -- greedy errors associated with each random set of seed points
      sizes  -- reduced data sizes from each random set of seed points
      seeds  -- the full set of random seed points
      Xs     -- reduced data from each random set of seed points
    
    Comments
    --------
    The `parallel` option also accepts positive integers indicating the
    number of processors that the user wants to use. 
    
    Parallelization uses some features from the concurrent.futures module.
    """
    return self.make(x, y, Nseeds, parallel=parallel)
  
  
  def make(self, x, y, Nseeds, parallel=True):
    """Make reduced-order splines for Nseeds number of random sets of seed points.
    
    Input
    -----
      x        -- samples
      y        -- data to be downsampled
      Nseeds   -- number of random sets of seed points
      parallel -- parallelize the computation for each random seed set?
                  (default True)
    
    Attributes
    ----------
      errors -- greedy errors associated with each random set of seed points
      sizes  -- reduced data sizes from each random set of seed points
      seeds  -- the full set of random seed points
      Xs     -- reduced data from each random set of seed points
    
    Comments
    --------
    The `parallel` option also accepts positive integers indicating the
    number of processors that the user wants to use. 
    
    Parallelization uses some features from the concurrent.futures module.
    """
    
    # Allocate some memory
    self.errors = np.zeros(Nseeds, dtype='double')
    self.sizes = np.zeros(Nseeds, dtype='double')
    self.seeds = np.zeros((Nseeds, self._deg+1), dtype='double')
    self.Xs = []
    
    if (parallel is False) or (_parallel is False):
      for nn in range(Nseeds):
        self.errors[nn], self.sizes[nn], self.seeds[nn] = _randomSeeds(x, y, tol=self._tol, rel=self._rel, deg=self._deg)
    elif _parallel is True:
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
      
      # Build reduced-order splines for many random sets of seed points
      output = []
      for nn in range(Nseeds):
        output.append(executor.submit(_randomSeeds, x, y, tol=self._tol, rel=self._rel, deg=self._deg))
      
      # Gather the results as they complete
      for ii, oo in enumerate(as_completed(output)):
        self.errors[ii], self.sizes[ii], self.seeds[ii], Xs = oo.result()
        self.Xs.append(Xs)
        
    self._made = True
  
  
  def plot_sizes(self, n=20, ax=None, show=True, xlabel='Size of reduced data', ylabel='Occurrence'):
    """Plot a histogram of the reduced data sizes produced by building
    reduced-order splines from random sets of seed points.
    
    Input
    -----
      n      -- number of histogram bins
                (default 20)
      ax     -- matplotlib plot/axis object
                (default None)
      show   -- display the plot?
                (default True)
      xlabel -- x-axis label 
                (default 'Size of reduced data')
      ylabel -- y-axis label
                (default 'Occurrence')
    
    Output
    ------
      If show=True then the plot is displayed.
      Otherwise, the matplotlib plot/axis object is output.
    """
    
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
      ax.set_ylabel(ylabel)
      
      if show:  # Display the plot
        plt.show()
      else:     # Otherwise, return plot objects for editing the plot in the future
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
      spline_mins = greedy.UnivariateSpline(np.sort(mins), np.arange(len(mins)), s=0)
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
  """K-fold cross validation on a given partition"""
  
  # Assemble training data excluding the i_fold'th partition for validation
  # TODO: These next few lines could be implemented more efficiently...
  folds = np.asarray(folds)
  complement = np.ones(len(folds), dtype='bool')
  complement[i_fold] = False
  f_trial = np.sort( np.concatenate([ff for ff in folds[complement]]) )
  fold = folds[i_fold]

  # Form trial data
  x_trial = x[f_trial]
  y_trial = y[f_trial]
  
  # Build trial spline
  spline = greedy._greedy(x_trial, y_trial, tol=tol, rel=rel, deg=deg, seeds=seeds)[0]
  
  # Compute L-infinity errors between trial and data in validation partition
  error, arg_error = Linfty(spline(x[fold]), y[fold], arg=True)
  if rel:
    error /= np.max(np.abs(y))
  
  return fold[arg_error], error


class CrossValidation(object):
  """A class for performing K-fold cross-validation studies to assess
  the errors in reduced-order spline interpolation.
  
  A K-fold cross-validation study consists of randomly distributing
  the original dataset into K partitions such that their union, when
  sorted, is the original data. Select the first partition for validation
  and build a reduced-order spline interpolant on the remaining K-1 
  partitions. Compute the L-infinty error between the resulting trial
  spline and the actual data in the validation partition. Repeat using
  each of the K partitions as a validation set. This results in K
  validation errors, one for each validation partition. The mean (median)
  of these is the mean (median) validation error. K-fold cross-validation
  is implemented in the `Kfold` method.
  
  This study can be repeated for different realizations of the random
  distribution of the original data into the K partitions. Each draw is from
  an independent and identical distribution so standard statistics applies
  to the mean (median) validation errors computed for the ensemble of all
  draws. Use the `MonteCarloKfold` method for this kind of study.
  """
  
  def __init__(self, tol=1e-6, rel=False, deg=5):
    """Create a CrossValidation object.
    
    Input
    -----
      tol -- L-infinity error tolerance for reduced-order spline 
             (default 1e-6)
      rel -- L-infinity error tolerance is relative to max abs of data?
             (default False)
      deg -- degree of interpolating polynomials 
             (default 5)
    """
    self._tol = tol
    self._rel = rel
    self._deg = deg
    self._made = False
  
  
  def Kfold(self, x, y, K=10, parallel=True, seeds=None, random=True):
    """K-fold cross-validation
    
    Input
    -----
      x        -- samples
      y        -- data to be downsampled
      K        -- number of partitions
                  (default 10)
      parallel -- parallelize the computation over each partition?
                  (default True)
      seeds    -- seed points for greedy algorithm
                  (default None)
      random   -- fill the partitions randomly from the data?
                  (default True)
    
    Attributes
    ----------
      args   -- data array indexes of largest validation errors 
                on each validation set
      errors -- validation errors on each validation set
    
    Comments
    --------
    The default option for `K` is 10 so that a random 10% subset of the 
    data is used to validate the trial reduced-order splines.
    
    If `seeds` option is None then the default seed points are used
    for the greedy algorithm. See greedy.ReducedOrderSpline documentation
    for further information.
    
    If `random` option is False then the partitions are filled with the
    data in sequential order. This is not a good idea for trying to 
    assess interpolation errors because interpolating over large regions
    of data will likely incur large errors. This option may be removed in 
    future romSpline releases.
    """
    
    assert len(x) == len(y), "Expecting input data to have same length."
    
    self._K = int(K)
    self._size = len(x)
    
    # Allocate some memory
    self.args = np.zeros(K, dtype='int')
    self.errors = np.zeros(K, dtype='double')
    
    # Divide into K nearly equal partitions
    if random:
      self._partitions = random_partitions(self._size, self._K)
    else:
      self._partitions = partitions(self._size, self._K)
    
    if (parallel is False) or (_parallel is False):
      for ii in range(self._K):
        self.args[ii], self.errors[ii] = _Kfold(x, y, self._K, ii, self._partitions, tol=self._tol, rel=self._rel, deg=self._deg, seeds=seeds)
    elif _parallel is True:
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
        self.args[ii], self.errors[ii] = ee.result()
    
    self._made = True
  
  
  def MonteCarloKfold(self, x, y, n, K=10, parallel=True, seeds=None, random=True, verbose=False):
    """Perform a number n of K-fold cross-validations
    
    Input
    -----
      x        -- samples
      y        -- data to be downsampled
      n        -- number of times to repeat K-fold cross-validation
                  with a random distribution of the data into the 
                  K partitions
      K        -- number of partitions
                  (default 10)
      parallel -- parallelize the computation over each partition?
                  (default True)
      seeds    -- seed points for greedy algorithm
                  (default None)
      random   -- fill the partitions randomly from the data?
                  (default True)
      verbose  -- print progress to screen?
                  (default False)
    
    Attributes
    ----------
      mc_args   -- array of data array indexes of validation errors on each
                   validation set
      mc_errors -- array of validation errors on each realized validation set
    
    Comments
    --------
    The default option for `K` is 10 so that a random 10% subset of the 
    data is used to validate the trial reduced-order splines.
    
    If `seeds` option is None then the default seed points are used
    for the greedy algorithm. See greedy.ReducedOrderSpline documentation
    for further information.
    
    If `random` option is False then the partitions are filled with the
    data in sequential order. This is not a good idea for trying to 
    assess interpolation errors because interpolating over large regions
    of data will likely incur large errors. This option may be removed in 
    future romSpline releases.
    """
    
    self.mc_args, self.mc_errors = [], []
    for nn in range(n):
      
      if verbose and not (nn+1)%10:
        print "Trials completed:", nn+1
      
      self.Kfold(x, y, K=K, parallel=parallel, seeds=seeds, random=random)
      self.mc_args.append( self.args )
      self.mc_errors.append( self.errors )
      
      self.mc_mean_errors = map(np.mean, self.mc_errors)
  
  
  def stats(self, lq=0.05, uq=0.95):
    """Compute mean, median, and two quantiles of ensemble of mean errors
    
    Input
    -----
      lq  -- low quantile (< 0.5)
             (default 0.05)
      uq  -- upper quantile (> 0.5)
             (default 0.95)
    
    Attributes
    ----------
      mc_mean           -- mean of ensemble of mean errors
      mc_std            -- sample standard deviation of ensemble of mean errors
      mc_lower_quantile -- lower quantile of ensemble of mean errors
      mc_upper_quantile -- upper quantile of ensemble of mean errors
      mc_median         -- median of ensemble of mean errors
    """
    
    # Mean of error means and sample standard deviation
    self.mc_mean = np.mean(self.mc_mean_errors)
    self.mc_std = np.std(self.mc_mean_errors, ddof=1)
    
    # Specific quantiles of error means
    self.mc_lower_quantile, self.mc_median, self.mc_upper_quantile = mquantiles(self.mc_mean_errors, prob=[lq, 0.5, uq])
  
  
  def plot_mc_errors(self, lq=0.05, uq=0.95, n=20, ax=None, show=True):
    """Plot a histogram of the mean validation errors from 
    a Monte Carlo K-fold cross-validation study
    
    Input
    -----
      lq   -- low quantile (< 0.5)
              (default 0.05)
      uq   -- upper quantile (> 0.5)
              (default 0.95)
      n    -- number of histogram bins
              (default 20)
      ax   -- matplotlib plot/axis object
              (default None)
      show -- display the plot?
              (default True)
    
    Output
    ------
      If show=True then the plot is displayed.
      Otherwise, the matplotlib plot/axis object is output.
    """
    
    if self._made:
      if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
      
      # Compute statistics of MonteCarloKfold output
      self.stats(lq=lq, uq=uq)
      
      # Plot histogram of mean errors
      ax.hist(np.log10(self.mc_mean_errors), n, color='k', alpha=0.33)
      
      # Plot mean (blue) and median (red) of mean errors
      ax.axvline(np.log10(self.mc_mean), 0, 1, color='b', label='Mean')
      ax.axvline(np.log10(self.mc_median), 0, 1, color='r', label='Median')
      
      # Plot lq and uq quantiles (dashed red)
      ax.axvline(np.log10(self.mc_upper_quantile), 0, 1, color='r', linestyle='--', label=str(int(uq*100))+'th quantile')
      ax.axvline(np.log10(self.mc_lower_quantile), 0, 1, color='r', linestyle='--', label=str(int(lq*100))+'th quantile')
      
      ax.set_xlim(np.log10(0.95*np.min(self.mc_mean_errors)), np.log10(1.05*np.max(self.mc_mean_errors)))
      ax.set_xlabel('Mean spline errors')
      ax.legend(loc='upper right', prop={'size':10})
      
      if show:  # Display the plot
        plt.show()
      else:     # Otherwise, return plot objects for editing the plot in the future
        if ax is None:
          return fig, ax
        else:
          return ax
      
    else:
      print "No data to plot. Run `MonteCarloKfold` method."
      
  
  def plot_partition_errors(self, ax=None, show=True):
    """Plot the largest interpolation error for each validation
    partition from a K-fold cross-validation study
    
    Input
    -----
      ax   -- matplotlib plot/axis object
              (default None)
      show -- display the plot?
              (default True)
    
    Output
    ------
      If show=True then the plot is displayed.
      Otherwise, the matplotlib plot/axis object is output.
    """
    if self._made:
      if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
      
      x_plot = range(len(self._partitions))
      ax.plot(x_plot, self.errors, 'ko-')
      
      ax.set_xlabel('Partition')
      ax.set_ylabel('Validation error')
      ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
      
      if show:  # Display the plot
        plt.show()
      else:     # Otherwise, return plot objects for editing the plot in the future
        if ax is None:
          return fig, ax
        else:
          return ax
      
    else:
      print "No data to plot. Run `Kfold` method."
  
  
  def plot_args_errors(self, x=None, axes='plot', ax=None, show=True):
    """Plot all the Monte Carlo K-fold cross-validation errors
    versus the samples at which the error is recorded 
    in each of the validation partitions.
    
    Input
    -----
      x    -- samples
              (default None)
      axes -- axis scales for plotting
              (default 'plot')
      ax   -- matplotlib plot/axis object
              (default None)
      show -- display the plot?
              (default True)
    """
    
    if self._made:
      if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
      
      # Flatten mc_args and mc_errors arrays
      # generated from MonteCarloKFold method
      args = np.asarray(self.mc_args).flatten()
      errors = np.asarray(self.mc_errors).flatten()
      
      if x is None:
        xplot = args
        ax.set_xlabel('Data array indexes')
      else:
        xplot = x[args]
        ax.set_xlabel('$x$')
      
      if axes == 'semilogy':
        ax.semilogy(xplot, errors, 'k.')
      elif axes == 'plot':
        ax.plot(xplot, errors, 'k.')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        
      ax.set_ylabel('Validation errors')
      
      if show:  # Display the plot
        plt.show()
      else:     # Otherwise, return plot objects for editing the plot in the future
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
    self.spline = greedy.ReducedOrderSpline(x, y, tol=tol, deg=deg)
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
