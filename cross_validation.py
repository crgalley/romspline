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

import greedy
from helpers import *



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


