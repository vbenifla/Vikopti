import os
import sys
import warnings
import numpy as np
import matplotlib
from contextlib import contextmanager, redirect_stdout, redirect_stderr


# Plots colors and format
CMAP = matplotlib.cm.coolwarm
CMAP.set_bad(color="lightgray")
BLUE = "cornflowerblue"


@contextmanager
def silencer(out=None, err=None):
    if out is None:
        out = open(os.devnull, "w")
    if err is None:
        err = open(os.devnull, "w")
    try:
        with redirect_stdout(out), redirect_stderr(err):
            yield
    finally:
        out.close()
        err.close()


@contextmanager
def silencer_full():
    file = open(os.devnull, "w")
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    stdout_fd_dup = os.dup(stdout_fd)
    stderr_fd_dup = os.dup(stderr_fd)
    os.dup2(file.fileno(), stdout_fd)
    os.dup2(file.fileno(), stderr_fd)
    file.close()
    try:
        yield
    finally:
        os.dup2(stdout_fd_dup, stdout_fd)
        os.dup2(stderr_fd_dup, stderr_fd)
        os.close(stdout_fd_dup)
        os.close(stderr_fd_dup)


def norm1D(arr: np.ndarray):
    """
    Normalize a 1D numpy array between 0 and 1.

    Parameters
    ----------
    arr : np.ndarray
        1D array to normalize.

    Returns
    -------
    np.ndarray
        Normalized array between 0 and 1.
        If the array is constant, the original array is returned.
    """
    # Check array dimension
    if arr.ndim != 1:
        raise ValueError("Input must be a 1D numpy array")

    # Get min, max values
    min_val = arr.min()
    max_val = arr.max()

    # Handle constant array
    if np.isclose(max_val, min_val):
        warnings.warn("Input array is constant; returning original array unchanged.")
        return arr.astype(float)

    return (arr - min_val) / (max_val - min_val)
