# Welcome to romSpline #

`romSpline` is an easy-to-use code for generating a reduced-order spline 
interpolant of 1d data.

`romSpline` uses a greedy algorithm on 1d data to find a minimal subset of 
samples that recovers the original data, to the requested tolerance, with 
a univariate spline interpolant. This code should be useful for downsampling 
or compressing large data arrays to their essential components from which 
the original information can be constructed. The degree of downsampling is 
often significant (e.g., orders of magnitude) for relatively smooth data.


### Installation ###

To set up `romSpline`, download or clone this repository and add the download 
path to your PYTHONPATH variable.

`romSpline` requires `numpy`, `scipy`, and `h5py`, which come with most Python
distributions.


### Getting started ###

See the accompanying IPython notebook (romSpline_example.ipynb) for a simple 
tutorial on using the code.

### Author information ###
Copyright (C) 2015 Chad Galley (*crgalley@tapir.caltech.edu*, *crgalley@gmail.com*). 
Released under the MIT license.
Comments and requests welcome.