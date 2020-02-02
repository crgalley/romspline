import pytest

try:
    from pathlib import Path # py 3
except ImportError:
    from pathlib2 import Path # py 2

import romspline

def test_regression():
    """Run the builtin regression test"""

    romspline.regression()

def test_readwrite():
    """Test reading/writing both hdf5 and text formats"""

    regression_path = Path(romspline.__path__[0]) / 'regressionData.h5'

    # Read h5
    h5_read = romspline.readSpline(regression_path)

    # Write h5
    h5_write_path = Path(__file__).parent / 'test.h5'
    h5_read.write(h5_write_path)

    # Write txt
    txt_path = Path(__file__).parent / 'test.txt'
    h5_read.write(txt_path)

    # Read txt
    txt_read = romspline.readSpline(txt_path)
