"""romSpline
   =========

romSpline is an easy-to-use code for generating a reduced-order spline 
interpolant of 1d data.

romSpline uses a greedy algorithm on 1d data to find a nearly minimal subset of 
samples for recovering the original data, to the requested tolerance, with 
a univariate spline interpolant. This code should be useful for downsampling 
or compressing large data arrays to their essential components needed for 
reconstructing the original information. The degree of downsampling is 
often significant (e.g., orders of magnitude) for relatively smooth data.

Future versions of romSpline may include support for multi-dimensional data 
and additional ways for estimating the reduced-order spline interpolation error
on data in the original set being compressed.

See the accompanying IPython notebook (romSpline_example.ipynb) for a 
tutorial on using the code. See also errors_example.ipynb for a tutorial
on using the interpolation error assessment and estimation features of the 
romSpline.

If you find this code useful for your publication work then please
cite the code repository (www.bitbucket.org/chadgalley/romSpline)
and the corresponding paper that discusses and characterizes the
reduced-order spline method (available soon).

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

from greedy import *            # For building reduced-order splines
from convergence import *       # For studying convergence
from random_seeds import *      # For studying the effect of seed points on reduced data sizes
from cross_validation import *  # For estimating (global) interpolation errors
from build_spline import *      # Convenience module for bulding reduced-order spline
                                # with a global interpolation error estimate from cross-validation



################################
#   Class for generating some  #
#   test data for showing how  #
#    to use the code in the    #
#   Jupyter/IPython notebooks  #
################################

class TestData(object):
  """Generate the test data used as in example IPython notebooks 
    for demonstrating the construction, properties, and errors 
    of a reduced-order spline interpolant.
  """
  
  def __init__(self, num=4001, noise=0., uv=0.):
    """Create a TestData object.
    
    Input
    -----
      num   -- number of samples to evaluate the function
               in domain [-1,1]
      noise -- amplitude of stochastic fluctuations added
               to smooth function values
               (default is 0.)
      uv    -- amplitude of high-frequency (i.e., ultra-violet)
               features added to smooth function values
               (default is 0.)
    
    Attributes
    ----------
      x -- samples
      y -- values of sampled function
    
    """
    
    # Generate test data
    self.x = np.linspace(-1, 1, num)
    self.y = self.f(self.x, noise=noise, uv=uv)
  
  
  def f(self, x, noise=0., uv=0.):
    """Function to sample for reduced-order spline examples
    
    Inputs
    ------
      x     -- values to sample the (smooth) function
      noise -- amplitude of stochastic fluctuations added
               to smooth function values
               (default is 0.)
      uv    -- amplitude of high-frequency (i.e., ultra-violet)
               features added to smooth function values
               (default is 0.)
    
    Output
    ------
      sampled function values
    
    Comments
    --------
    The function being evaluated is
    
      f(x) = 100.*( (1.+x) * sin(5.*(x-0.2)**2) 
                    + exp(-(x-0.5)**2/2./0.01) * sin(100*x) 
                  )
    """
    
    # Validate inputs
    x = np.asarray(x)
    
    # Return smooth function values
    ans = 100.*( (x+1.)*np.sin(5.*(x-0.2)**2) + np.exp(-(x-0.5)**2/2./0.01)*np.sin(100*x) )
    
    # Return smooth function values with high-frequency (UV) features
    if uv != 0.:
      assert type(uv) in [float, int], "Expecting integer or float type."
      ans += float(uv)*self.uv(x)
    
    # Return smooth function values with stochastic noise
    if noise != 0.:
      assert type(noise) in [float, int], "Expecting integer or float type."
      ans += float(noise)*np.random.randn(len(x))
    
    return ans
  
  
  def dfdx(self, x):
    """Analytic derivative of f
    
    Inputs
    ------
      x -- values to sample the derivative of 
           the function f(x) (see self.f method)
    
    Outputs
    -------
      ans -- values of analytically calculated 
             derivative of the function
    
    """
    x = np.asarray(x)
    
    a = 10.*(-0.2+x)*(1.+x)*np.cos(5.*(-0.2 + x)**2)
    b = 100.*np.exp(-50.*(-0.5+x)**2)*np.cos(100.*x)
    c = np.sin(5.*(-0.2+x)**2)
    d = -100.*np.exp(-50.*(-0.5+x)**2)*(-0.5+x)*np.sin(100.*x)
    
    ans = 100.*(a+b+c+d)
    
    return ans
  
  
  def uv(self, x, width=20):
    """Generate high-frequency oscillations
    
    Inputs
    ------
      x     -- values to sample the high-frequency
               oscillations
      width -- number of samples corresponding to
               the period of the high-frequency
               oscillations
               (default is 20)
    
    Outputs
    -------
      array of high-frequency oscillating values
    
    """
    X = x[width] - x[0]
    return np.sin(len(x)/X * x)


