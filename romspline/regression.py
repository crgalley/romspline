from __future__ import print_function

"""
Test the reduced-order spline greedy algorithm outputs with
previously generated data to make sure the code still works correctly.
"""

import h5py

try:
    from pathlib import Path # py 3
except ImportError:
    from pathlib2 import Path # py 2

import numpy as np
import romspline.greedy as greedy
import romspline.example as example


def regression():
  # Load the reference reduced-order spline data
  # which will be regarded as the "truth" for 
  # code-checking purposes.
  
  # But first, get the path to the data file
  path = Path(__file__).parent
  
  # Now load the data
  fp = h5py.File(path / 'regressionData.h5', 'r')
  X = fp['X'][:]
  Y = fp['Y'][:]
  deg = fp['deg'][()]
  tol = fp['tol'][()]
  errors = fp['errors'][:]
  args = fp['args'][:]
  fp.close()
  
  # Generate test data
  test = example.TestData()
  
  # Build reduced-order spline interpolant for the
  # test data using options found in regressionData.h5.
  print("Building reduced-order spline...",)
  test_spline = greedy.ReducedOrderSpline(test.x, test.y, deg=deg, tol=tol)
  print("Done")
  
  # Perform a few checks on the greedy algorithm
  print("Testing greedy algorithm:")
  
  print("  Comparing reduced data size...",)
  if test_spline.size == len(X):
    print("Passed")
  else:
    print("Failed [Size of reduced data ({}) does not equal {}]".format(test_spline.size, len(X)))
  
  print("  Comparing selected array indices...",)
  if np.all([test_spline.args[ii] == args[ii] for ii in range(len(args))]):
    print("Passed")
  else:
    print("Failed")
  
  print("  Comparing greedy errors...",)
  if np.all([np.isclose(test_spline.errors[ii], errors[ii]) for ii in range(len(errors))]):
    print("Passed")
  else:
    print("Failed")


