import numpy as np



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


def integrate(y, x, weight=None, partial=False):
  """Trapezoidal rule for integration.
  
  Inputs
  ======
    y       -- data values or responses
    x       -- data samples
    weight  -- Weighting factor in the integrand
               (default is None)
    partial -- Output partial sums instead of the integral
               (default is False)
  
  Outputs
  =======
    Scalar equal to the numerical integration of y
  over x using the trapezoidal quadrature rule. If
  partial is True then the output is an array of 
  partial sums for each element of y over the previous
  x values.
  
  Comments
  ========
    The trapezoidal rule is second-order convergent in 
  the uniform increment of x.
  
  """
  
  # Check that the data arrays have the same length
  assert len(y) == len(x), "Expecting arrays `y` and `x` to have equal lengths."
  
  # Check that the samples are uniformly spaced
  dx = x[1]-x[0]
  #assert np.all([np.isclose(xx, dx) for xx in np.diff(x)]), "Expecting input sample array `x` to be uniformly spaced."
  
  # Build quadrature weights
  weights = np.ones(len(x))*dx
  weights[0] *= 0.5
  weights[-1] *= 0.5
  if weight is not None:
    assert len(x) == len(weight), "Expecting `weight` option to be an array with the same length as `x`."
    weights /= weight
  
  # Compute partial sum, if requested
  if partial:
    ans = np.zeros_like(y)
    for ii in range(len(ans[:-1])):
      ans[ii+1] = ans[ii] + weights[ii]*y[ii]
  # Otherwise, compute full integral
  else:
    ans = np.sum(weights*y)
  
  return ans


def overlap(x, y1, y2, weight=None, partial=None):
  
  assert len(y1) == len(y2)
  assert len(x) == len(y1)
  
  norm1 = integrate(np.abs(y1)**2, x, weight=weight)**0.5
  norm2 = integrate(np.abs(y2)**2, x, weight=weight)**0.5
  
  return integrate(np.conjugate(y1)*y2/norm1/norm2, x, weight=weight, partial=partial).real()


# def unique(x, y, rule='mean'):
#
#   assert rule in ['mean', 'median'], "Expecting either 'mean' or 'median' for option `rule`."
#   assert len(x) == len(y), "Expecting arrays x and y to have the same length."
#
#   x_unique = np.unique(x)
#   y_unique = np.zeros_like(x_unique)
#
#   for ii, xx in enumerate(x_unique):
#
#     # Gather all the y values that have the same x
#     # nonunique = []
#     # for jj in range(len(x)):
#     #   if x[jj] == x_unique[ii]:
#     #     nonunique.append(y[jj])
#
#     # Compute the mean or median of the y's at the
#     # nonunique x samples
#     # if rule == 'mean':
#     #   y_unique[ii] = np.mean(nonunique)
#     # elif rule == 'median':
#     #   y_unique[ii] = np.median(nonunique)
#     mask = (xx == x)
#     if rule == 'mean':
#       y_unique[ii] = np.mean(y[mask])
#     elif rule == 'median':
#       y_unique[ii] = np.median(y[mask])
#     #print y_unique[ii]
#
#   return x_unique, y_unique
#
#
# def k_nearest_neighbors(x, y, k, rule='mean'):
#
#   x_unique, y_unique = unique(x, y, rule=rule)
#
#   # Allocate some memory
#   mean = np.zeros_like(x_unique)
#   median = np.zeros_like(x_unique)
#
#   # Find nearest neighbors for each sample
#   for ii in range(len(x_unique)):
#     # Get k neighbors nearest to x[ii]
#     distance = np.array([np.abs(x_unique[ii]-x_unique[jj]) for jj in range(len(x_unique))])
#     sorted_indexes = np.argsort(distance)[1:k+1]  # Don't inlude the x[ii] point as a nearest neighbor
#     nearest_neighbor = y_unique[sorted_indexes]
#
#     # Compute average of these k nearest neighbors
#     mean[ii] = np.mean(nearest_neighbor)
#     median[ii] = np.median(nearest_neighbor)
#
#   return mean, median


