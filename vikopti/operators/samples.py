import math
import warnings
import numpy as np
from scipy.stats import qmc

# Default seed value
_DEFAULT_SEED = 92


def grid(n: int, bounds: list[list[float]]):
    """
    Generate a structured grid sample.

    Parameters
    ----------
    n : int
        Size.
    bounds : list of lists
        A two-element list: [lower_bounds, upper_bounds].
        Each must be a list of same length.

    Returns
    -------
    np.ndarray
        Sample.
    """
    lower, upper = np.array(bounds)
    n_var = len(lower)
    n_x = int(n ** (1 / n_var))
    x_grid = [np.linspace(lower[i], upper[i], n_x) for i in range(n_var)]
    mesh = np.meshgrid(*x_grid)
    x = np.vstack([m.ravel() for m in mesh]).T
    return x


def random(n: int, bounds: list[list[float]]):
    """
    Generate random sample.

    Parameters
    ----------
    n : int
        Size.
    bounds : list of lists
        A two-element list: [lower_bounds, upper_bounds].
        Each must be a list of same length.

    Returns
    -------
    np.ndarray
        Sample.
    """
    lower, upper = np.array(bounds)
    n_var = len(lower)
    x = np.random.uniform(lower, upper, (n, n_var))
    return x


def lhs(n: int, bounds: list[list[float]], seed: int = _DEFAULT_SEED):
    """
    Generate sample using Latin Hypercube Sampling.

    Parameters
    ----------
    n : int
        Size.
    bounds : list of lists
        A two-element list: [lower_bounds, upper_bounds].
        Each must be a list of same length.
    seed : int, optional
        Random seed, by default 92.

    Returns
    -------
    np.ndarray
        Sample.
    """
    lower, upper = np.array(bounds)
    n_var = len(lower)
    x_sample = qmc.LatinHypercube(n_var, seed=seed).random(n)
    x = qmc.scale(x_sample, lower, upper)
    return x


def sobol(n: int, bounds: list[list[float]], seed: int = _DEFAULT_SEED):
    """
    Generate sample using Sobol sequences.

    Parameters
    ----------
    n : int
        Size.
    bounds : list of lists
        A two-element list: [lower_bounds, upper_bounds].
        Each must be a list of same length.
    seed : int, optional
        Random seed, by default 92.

    Returns
    -------
    np.ndarray
        Sample.
    """
    lower, upper = np.array(bounds)
    n_var = len(lower)
    x_sample = qmc.Sobol(n_var, seed=seed).random(2 ** math.floor(np.log2(n)))
    x = qmc.scale(x_sample, lower, upper)
    return x


# Dictionary mapping method names to functions
_SAMPLES = {"grid": grid, "random": random, "lhs": lhs, "sobol": sobol}


def sample(
    n: int, bounds: list[list[float]], method: str = "lhs", seed: int = _DEFAULT_SEED
):
    """
    Generate sample using the specified method.

    Parameters
    ----------
    n : int
        Size.
    bounds : list of lists
        A two-element list: [lower_bounds, upper_bounds].
        Each must be a list of same length.
    method : str, optional
        Sampling method, by default "lhs".
    seed : int, optional
        Random seed, by default 92.

    Returns
    -------
    np.ndarray
        Sample.
    """
    # Some checks
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise ValueError("bounds must be a list [lower_bounds, upper_bounds]")
    if len(bounds[0]) != len(bounds[1]):
        raise ValueError("lower_bounds and upper_bounds must have the same length")
    if method not in _SAMPLES:
        warnings.warn(f"Unknown method '{method}', using 'lhs' instead.", UserWarning)
        method = "lhs"

    # Get chosen sampling method
    func = _SAMPLES[method]

    # Only pass seed if method supports it
    if method in ("lhs", "sobol"):
        return func(n, bounds, seed=seed)
    else:
        return func(n, bounds)
