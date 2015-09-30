"""romSpline
   =========

romSpline is an easy-to-use code for generating a reduced-order spline 
interpolant of 1d data.

romSpline uses a greedy algorithm on 1d data to find a minimal subset of 
samples for recovering the original data, to the requested tolerance, with 
a univariate spline interpolant. This code should be useful for downsampling 
or compressing large data arrays to their essential components needed for 
reconstructing the original information. The degree of downsampling is 
often significant (e.g., orders of magnitude) for relatively smooth data.

See the accompanying IPython notebook (romSpline_example.ipynb) for a 
tutorial on using the code.

"""

__copyright__ = "Copyright (C) 2015 Chad Galley"
__author__ = "Chad Galley"
__email__ = "crgalley@tapir.caltech.edu, crgalley@gmail.com"
__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


import numpy as np, h5py
from scipy.interpolate import UnivariateSpline


########################
# Class for building a #
# reduced-order spline #
########################

class ReducedOrderSpline(object):
  """Class for building a reduced-order spline interpolant"""
  
  # TODO: Accommodate arrays of arrays
  
  def __init__(self, x=None, y=None, deg=5, tol=1e-6, rel=False, verbose=False, seeds=None):
    self._deg = deg
    self._rel = rel
    self._made = False
  
    # Run greedy algorithm if data is given for class instantiation
    if y is not None:
      if x is None:
        x = np.arange(len(y))
      assert len(x) == len(y), "Array sizes must be equal."
      self.greedy(x, y, deg=deg, tol=tol, rel=rel, verbose=verbose, seeds=seeds)
    
  def seed(self, x, seeds=None):
    """Seed the greedy algorithm with (deg+1) evenly spaced indices"""
    if seeds is None:
      f = lambda m, n: [ii*n//m + n//(2*m) for ii in range(m)]
      self.indices = np.sort(np.hstack([[0, len(x)-1], f(self._deg-1, len(x))]))
    else:
      self.indices = seeds
    self._indices = self.indices[:]  # Keep track of unsorted indices
    self.errors = []
  
  def greedy(self, x, y, tol=1e-6, rel=False, deg=5, verbose=False, seeds=None):
    """
    Greedy algorithm for finding optimal knots to build a reduced-order 
    spline interpolant
    
    Input
    -----
    x       -- samples
    y       -- data to be downsampled
    deg     -- degree of interpolating polynomials 
               (default 5)
    tol     -- L-infinity error tolerance for reduced-order spline 
               (default 1e-6)
    rel     -- L-infinity error tolerance is relative to max abs of data?
               (default is False)
    verbose -- write greedy algorithm output to screen 
               (default False)
    
    Output attributes
    -----------------
    errors  -- L-infinity errors of successive spline interpolants 
              produced by greedy algorithm until tol is reached
    knots   -- "Optimal" knots for spline interpolant of given degree
    indices -- Array indices of knots
    
    Comments
    --------
    If x is not given then interpolation is built using y array index labels
    as the samples.
    """
    
    self._deg = deg
    self._rel = rel
    
    if rel:
      ymax = np.max(np.abs(y))
      assert ymax > 0., "All data samples are zero."
    else:
      ymax = 1.
    self._tol = tol
    #self.tol = self._tol*ymax
    
    # Seed greedy algorithm
    _seed(x, deg=self._deg, seeds=seeds)
      
    # Greedy algorithm
    self._spline, self.knots, self.indices, self.errors, self.compression, self.tol = _greedy(x, y, tol=self._tol, rel=self._rel, deg=self._deg, verbose=verbose, seeds=seeds)
    
    # Define some attributes for spline evaluation
    self._data = y[self.indices]
    self.size = len(self.indices)
    self._made = True
    
  def __call__(self, x, dx=0):
   return self.eval(x, dx=dx)
  
  def eval(self, x, dx=0):
    """Evaluate reduced-order spline or its dx derviatives at x"""
    return self._spline(x, dx)
  
  def verify(self, x, y):
    """Verify the reduced-order spline on the training data"""
    if self._made:
      errors = y - self._spline(x)
    else:
      raise Exception, "No spline interpolant to compare against. Run the greedy method."
    print "Reduced-order spline meets tolerance:", np.all(np.abs(errors) <= self.tol)
    # TODO: Add plot of training data errors?
    return errors
    
  def read(self, file):
    """Load spline interpolant data from HDF5 file format"""
    # TODO: Include text and/or binary formats
    try:
      fp = h5py.File(file, 'r')
      isopen = True
    except:
      raise Exception, "Could not open file for reading."
    if isopen:
      self._deg = fp['deg'][()]
      self.tol = fp['tol'][()]
      self.knots = fp['knots'][:]
      self._data = fp['data'][:]
      self.errors = fp['errors'][:]
      fp.close()
      self._spline = UnivariateSpline(self.knots, self._data, k=self._deg, s=0)
      self._made = True
  
  def write(self, file):
    """Write spline interpolant data to HDF5 file format"""
    # TODO: Include text and/or binary formats
    try:
      fp = h5py.File(file, 'w')
      isopen = True
    except:
      raise Exception, "Could not open file for writing."
    if isopen:
      fp.create_dataset('deg', data=self._deg, dtype='int')
      fp.create_dataset('tol', data=self.tol, dtype='double')
      fp.create_dataset('knots', data=self.knots, dtype='double', compression='gzip', shuffle=True)
      fp.create_dataset('data', data=self._data, dtype='double', compression='gzip', shuffle=True)
      fp.create_dataset('errors', data=self.errors, dtype='double', compression='gzip', shuffle=True)
      fp.close()
    

def readSpline(self, file):
  """Load spline interpolant data from HDF5 file format 
  without having to instantiate ReducedOrderSpline class"""
  # TODO: Include text and/or binary formats
  try:
    fp = h5py.File(file, 'r')
    isopen = True
  except:
    raise Exception, "Could not open file for reading."
  if isopen:
    deg = fp['deg'][()]
    knots = fp['knots'][:]
    data = fp['data'][:]
    fp.close()
    return UnivariateSpline(knots, data, k=deg, s=0)
    

#################################
# Functions for parallelization #
#################################

def _seed(x, deg=5, seeds=None):
  """Seed the greedy algorithm with (deg+1) evenly spaced indices"""
  if seeds is None:
    f = lambda m, n: [ii*n//m + n//(2*m) for ii in range(m)]
    indices = np.sort(np.hstack([[0, len(x)-1], f(deg-1, len(x))]))
  else:
    indices = seeds
  errors = []
  
  return indices, errors

def _greedy(x, y, tol=1e-6, rel=False, deg=5, verbose=False, seeds=None):
  """Greedy algorithm for building a reduced-order spline"""
  if verbose:
    print "\nSize", "\t", "Error"
    print "="*13
  
  if rel:
    ymax = np.max(np.abs(y))
    assert ymax > 0., "All data samples are zero."
  else:
    ymax = 1.
  _tol = tol*ymax
  
  # Seed greedy algorithm
  indices, errors = _seed(x, deg=deg, seeds=seeds)
  _indices = indices[:]  # Keep track of unsorted indices
  
  # Greedy algorithm
  flag, ctr = 0, len(indices)+1
  while flag == 0 and ctr < len(x):	
    # Spline interpolant on current set of knots
    s = UnivariateSpline(x[indices], y[indices], k=deg, s=0)
    
    # L-infinity errors of current spline interpolant and data
    errs = np.abs( s(x)-y )
    
    # Get the index of the largest interpolation error on y
    imax = np.argmax( errs )
    
    # Update arrays with "worst knot"
    errors.append( errs[imax] )
    _indices = np.hstack([_indices, imax])
    indices = np.sort(np.hstack([indices, imax]))  # Knots must be sorted
    
    # Print to screen, if requested
    if verbose:
      print ctr, "\t", errors[-1]
    
    # Check if greedy error is below tolerance and exit if so
    if errors[-1] < _tol:
      flag = 1
      _indices = _indices[:-1]
      indices = np.sort(_indices)
      errors = np.array(errors[:-1])
    
    ctr += 1
  
  # Define some attributes for spline evaluation
  spline = UnivariateSpline(x[indices], y[indices], k=deg, s=0)
  knots = x[indices]
  size = len(indices)
  compression = float(len(y))/size
  
  return spline, knots, indices, errors, compression, _tol
