# Welcome to RomSpline #

RomSpline is an easy-to-use Python code for compressing and interpolating 1d data using reduced-order modeling and statistical learning methods.

RomSpline uses a greedy algorithm to find a nearly optimal subset of data samples that recovers the original data, to the requested tolerance, with a univariate spline interpolant. The output is the optimal subset of data and the corresponding reduced-order spline interpolant.

RomSpline also contains code to estimate the prediction errors at new samples due to interpolation. These methods are largely based on Monte Carlo K-fold cross-validation studies. The mean of the resulting validation errors can be used as a global and useful upper bound on the interpolation errors.

This code should be useful for downsampling or compressing large data arrays to their essential components from which the original information can be constructed and new data predicted through interpolation. The degree of downsampling can be significant (e.g., orders of magnitude) for relatively smooth data. Furthermore, the distribution of the resulting reduced dataset provides information into features and structures of the data that might otherwise not be readily observed.

Future releases of RomSpline will provide locally adapted interpolation error estimations based on more refined cross-validation studies. In addition, enhancements to the greedy algorithm will incorporate possible additional information about data quality (such as uncertainties in the data values being compressed and interpolated, if available).


### Installation ###

To set up RomSpline, download or clone this repository and add the download path to your PYTHONPATH variable. 

Alternatively, include the following lines in your Python code:

    import sys
    sys.path.append(<path to romspline>)
    import romspline


RomSpline requires NumPy, SciPy, and H5py, which come with most Python distributions. For parallelization, which is useful but not necessary for some of the cross-validation routines, RomSpline currently uses the concurrent.futures module. If you are using Python 2 and do not have concurrent.futures installed you may install it using pip:

    pip install futures

Future versions of RomSpline will not use concurrent.futures.


### Getting started ###

See the accompanying IPython notebooks (romSpline_example.ipynb and errors_example.ipynb) for simple tutorials on using the code and estimating
errors of the reduced-order spline interpolant for predicting new values. 

#### Author information ####
Copyright (C) 2015 Chad Galley (*crgalley "at" tapir "dot" caltech "dot" edu*). 
Released under the MIT/X Consortium license.
Comments and requests welcome.