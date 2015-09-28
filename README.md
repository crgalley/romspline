# Welcome to romSpline #

romSpline is an easy-to-use code for generating a reduced-order model for 
spline interpolation of 1d data.

romSpline uses a greedy algorithm to find a nearly optimal subset of samples that recovers the original data, to the requested tolerance, with a univariate spline interpolant. The output is the optimal subset of data and the corresponding reduced-order spline interpolant.

This code should be useful for downsampling or compressing large data arrays to their essential components from which the original information can be constructed. The degree of downsampling can be significant (e.g., orders of magnitude) for relatively smooth data.


### Installation ###

To set up romSpline, download or clone this repository and add the download path to your PYTHONPATH variable. 

Alternatively, include the following lines in your Python code:

    import sys
    sys.path.append(<path to romSpline>)
    import romSpline


romSpline requires NumPy, SciPy, and H5py, which come with most Python distributions.


### Getting started ###

See the accompanying IPython notebook (romSpline_example.ipynb) for a simple tutorial on using the code.

#### Author information ####
Copyright (C) 2015 Chad Galley (*crgalley@tapir.caltech.edu*, *crgalley@gmail.com*). 
Released under the MIT license.
Comments and requests welcome.