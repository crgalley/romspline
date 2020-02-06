[![github](https://img.shields.io/badge/GitHub-romspline-blue.svg)](https://github.com/crgalley/romspline)
[![PyPI version](https://badge.fury.io/py/romspline.svg)](https://badge.fury.io/py/romspline)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/romspline.svg)](https://anaconda.org/conda-forge/romspline)
[![arXiv:1611.07529](https://img.shields.io/badge/arXiv-1611.07529-B31B1B.svg)](https://arxiv.org/abs/1611.07529)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/crgalley/romspline/blob/master/LICENSE)

# Welcome to romSpline #

romSpline is an easy-to-use Python code for compressing and interpolating 1d data using reduced-order modeling and statistical learning methods.

romSpline uses a greedy algorithm to find a nearly optimal subset of data samples that recovers the original data, to the requested tolerance, with a univariate spline interpolant. The output is the optimal subset of data and the corresponding reduced-order spline interpolant.

romSpline also contains code to estimate the prediction errors at new samples due to interpolation. These methods are largely based on Monte Carlo K-fold cross-validation studies. The mean of the resulting validation errors can be used as a global and useful upper bound on the interpolation errors.

This code should be useful for downsampling or compressing large data arrays to their essential components from which the original information can be constructed and new data predicted through interpolation. The degree of downsampling can be significant (e.g., orders of magnitude) for relatively smooth data. Furthermore, the distribution of the resulting reduced dataset provides information into features and structures of the data that might otherwise not be readily observed.

Future releases of romSpline will provide locally adapted interpolation error estimations based on more refined cross-validation studies. In addition, enhancements to the greedy algorithm will incorporate possible additional information about data quality (such as uncertainties in the data values being compressed and interpolated, if available).

If you use this code for academic work then please cite the following paper describing the method and algorithm:

* C. R. Galley and P. Schmidt, 
"Fast and efficient evaluation of gravitational waveforms via reduced-order spline interpolation" 
[arXiv:1611.07529](https://arxiv.org/abs/1611.07529)


## Installation

### PyPI
_**romspline**_ is available through [PyPI](https://pypi.org/project/romspline/):

```shell
pip install romspline
```

### Conda
_**romspline**_ is available on [conda-forge](https://anaconda.org/conda-forge/romspline):

```shell
conda install -c conda-forge romspline
```

### From source

```shell
git clone https://github.com/crgalley/romspline.git
cd romspline
python setup.py install
```

If you do not have root permissions, replace the last step with
`python setup.py install --user`.  Instead of using `setup.py`
manually, you can also replace the last step with `pip install .` or
`pip install --user .`.

Alternatively, you can download or clone this repository and add the
download path to your PYTHONPATH variable.

As another alternative, include the following lines in your Python code:

    import sys
    sys.path.append(<path to romspline>)
    import romspline

## Dependencies
All of these can be installed through pip or conda.
* [numpy](https://docs.scipy.org/doc/numpy/user/install.html)
* [scipy](https://www.scipy.org/install.html)
* [h5py](https://pypi.org/project/h5py/)
* [pathlib2](https://pypi.org/project/pathlib2/) (backport of
  `pathlib` to pre-3.4 python)

romSpline requires NumPy, SciPy, and H5py, which come with most Python distributions. For parallelization, which is useful but not necessary for some of the cross-validation routines, romSpline currently uses the concurrent.futures module. If you are using Python 2 and do not have concurrent.futures installed you may install it using pip:

    pip install futures

Future versions of romSpline will not use concurrent.futures.


## Getting started

See the accompanying IPython notebooks (romSpline_example.ipynb and errors_example.ipynb) for simple tutorials on using the code and estimating
errors of the reduced-order spline interpolant for predicting new values. 

#### Author information ####
Copyright (C) 2015 Chad Galley (*crgalley "at" tapir "dot" caltech "dot" edu*). 
Released under the MIT/X Consortium license.
Comments and requests welcome.
