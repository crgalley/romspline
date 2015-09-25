# Welcome to romSpline #

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

### How do I get set up? ###

To set up romSpline, download or clone this repository and add the path to
romSpline to your PYTHONPATH variable.

*romSpline* requires numpy, scipy, and h5py, which come with most Python
distributions.


Written by Chad Galley (crgalley@tapir.caltech.edu, crgalley@gmail.com)
### Who do I talk to? ###

* Repo owner or admin
* Other community or team contact