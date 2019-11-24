'''
Standard setup.py to upload the code on pypi.

    python setup.py sdist bdist_wheel
    twine upload dist/*
'''
import setuptools

with open("README.md", "rb") as fh:
    long_description = fh.read().decode("UTF-8")

import sys
sys.path.insert(0,"romspline")

from __version__ import __version__

setuptools.setup(
    name="romspline",
    version=__version__,
    author="Chad Galley",
    author_email="crgalley@tapir.caltech.edu",
    description="A Python package for building reduced-order models to interpolate univariate data.",
    keywords='reduced order model spline interpolation data compression reduction',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/crgalley/romspline/",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'h5py',
    ],
    package_data={'romspline':['regressionData.h5']},
    include_package_data=True,
    classifiers=[
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Operating System :: OS Independent",
    ],
)
