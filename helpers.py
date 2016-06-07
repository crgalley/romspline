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


