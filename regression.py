"""
Test the reduced-order spline greedy algorithm outputs with
previously generated data to make sure the code works correctly.
"""

import h5py, numpy as np
from __init__ import TestData
import greedy




# Load the reference reduced-order spline data
# which will be regarded as the "truth" for 
# code-checking purposes.
fp = h5py.File('regressionData.h5', 'r')
X = fp['X'][:]
Y = fp['Y'][:]
deg = fp['deg'][()]
tol = fp['tol'][()]
errors = fp['errors'][:]
args = fp['args'][:]
fp.close()

# Generate test data
test = TestData()

# Build reduced-order spline interpolant 
# for the test data using default options.
test_spline = greedy.ReducedOrderSpline(test.x, test.y, deg=deg, tol=tol)

# Perform a few checks on the greedy algorithm
print "Testing greedy algorithm:"

print "  Comparing reduced data size...",
if test_spline.size == len(X):
  print "Passed"
else:
  print "Failed [Size of reduced data ({}) does not equal {}]".format(test_spline.size, len(X))

print "  Comparing selected array indices...",
if np.all([test_spline.args[ii] == args[ii] for ii in range(len(args))]):
  print "Passed"
else:
  print "Failed"

print "  Comparing greedy errors...",
if np.all([np.isclose(test_spline.errors[ii], errors[ii]) for ii in range(len(errors))]):
  print "Passed"
else:
  print "Failed"


