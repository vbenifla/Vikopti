import math
import numpy as np
from scipy.stats import qmc

SEED = 92


def uniform(n: int, bounds: np.ndarray):
    """
    Generate uniform sample.

    Parameters
    ----------
    n : int
        Size.
    bounds : list
        A two element list with the lower and upper bounds of each variables.

    Returns
    -------
    np.ndarray
        Sample.
    """
    n_var = len(bounds[0])
    n_x = int(n ** (1 / n_var))
    x_grid = [np.linspace(bounds[0, i], bounds[1, i], n_x) for i in range(n_var)]
    mesh = np.meshgrid(*x_grid)
    x = np.vstack([m.ravel() for m in mesh]).T
    return x


def random(n: int, bounds: np.ndarray):
    """
    Generate random sample.

    Parameters
    ----------
    n : int
        Size.
    bounds : list
        A two element list with the lower and upper bounds of each variables.

    Returns
    -------
    np.ndarray
        Sample.
    """
    n_var = len(bounds[0])
    x = np.random.uniform(bounds[0], bounds[1], (n, n_var))
    return x


def lhs(n: int, bounds: np.ndarray):
    """
    Generate sample using Latin Hypercube Sampling.

    Parameters
    ----------
    n : int
        Size.
    bounds : list
        A two element list with the lower and upper bounds of each variables.

    Returns
    -------
    np.ndarray
        Sample.
    """
    n_var = len(bounds[0])
    x_sample = qmc.LatinHypercube(n_var, seed=SEED).random(n)
    x = qmc.scale(x_sample, bounds[0], bounds[1])
    return x


def sobol(n: int, bounds: np.ndarray):
    """
    Generate sample using Sobol sequences.

    Parameters
    ----------
    n : int
        Size.
    bounds : list
        A two element list with the lower and upper bounds of each variables.

    Returns
    -------
    np.ndarray
        Sample.
    """
    n_var = len(bounds[0])
    x_sample = qmc.Sobol(n_var, seed=SEED).random(2 ** math.floor(np.log2(n)))
    x = qmc.scale(x_sample, bounds[0], bounds[1])
    return x


# Dictionary mapping method names to functions
SAMPLES = {"uniform": uniform, "random": random, "lhs": lhs, "sobol": sobol}


def sample(n: int, bounds: np.ndarray, method: str = "lhs"):
    """
    Generate sample using the specified method.

    Parameters
    ----------
    n : int
        Size.
    bounds : list
        A two element list with the lower and upper bounds of each variables.
    method : str, optional
        sample method, by default "lhs".

    Returns
    -------
    np.ndarray
        Sample.
    """
    return SAMPLES[method](n, bounds)
