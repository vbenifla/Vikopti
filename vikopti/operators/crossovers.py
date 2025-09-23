import warnings
import numpy as np


def ux(x1: float, x2: float, xl: float, xu: float, eta: float = 5.0):
    """
    Uniform crossover between two parent variables.

    Parameters
    ----------
    x1 : float
        First parent variable.
    x2 : float
        Second parent variable.
    xl : float
        Lower bound for the variable.
    xu : float
        Upper bound for the variable.

    Returns
    -------
    float
        Offspring variable.
    """
    # Sample uniformly between the two parents
    xo = np.random.uniform(x1, x2)

    return xo


def sbx(x1: float, x2: float, xl: float, xu: float, eta: float = 5.0):
    """
    Perform Simulated Binary Crossover (SBX) between two parent variables.

    Parameters
    ----------
    x1 : float
        First parent variable.
    x2 : float
        Second parent variable.
    xl : float
        Lower bound for the variable.
    xu : float
        Upper bound for the variable.
    eta : float, optional
        Distribution index for SBX, by default 5.0.

    Returns
    -------
    tuple
        Two offspring variables.
    """

    # Order parents design variables
    x1, x2 = min(x1, x2), max(x1, x2)

    # Compute probability density
    u = np.random.rand()
    delta = min((x1 - xl), (xu - x2))
    gamma = 1 + 2 * delta / (x2 - x1)
    alpha = 2 - gamma ** (-eta - 1)
    if u <= 1 / alpha:
        beta = alpha * u ** (1.0 / (eta + 1))
    else:
        beta = (1.0 / (2.0 - alpha * u)) ** (1.0 / (eta + 1))

    # randomly picks between children
    if np.random.rand() < 0.5:
        xo = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
    else:
        xo = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)

    return xo


def pnx(x1: float, x2: float, xl: float, xu: float, eta: float = 5.0):
    """
    Perform Parent-Centric Normal Crossover (PNX) between two parent variables.

    Parameters
    ----------
    x1 : float
        First parent variable.
    x2 : float
        Second parent variable.
    xl : float
        Lower bound for the variable.
    xu : float
        Upper bound for the variable.
    eta : float, optional
        Scaling factor for PNX, by default 5.0.

    Returns
    -------
    tuple
        Two offspring variables.
    """
    # Check eta
    if eta == 0:
        eta = 1e-6

    # randomly picks between children
    if np.random.rand() < 0.5:
        xo = np.random.normal(x1, np.abs(x2 - x1) / eta)
    else:
        xo = np.random.normal(x2, np.abs(x2 - x1) / eta)
    return xo


# Dictionary mapping method names to functions
_CROSSOVERS = {"ux": ux, "sbx": sbx, "pnx": pnx}


def cross(
    xp1: np.ndarray, xp2: np.ndarray, bounds: np.ndarray, method="sbx", eta: float = 5.0
):
    """
    Perform crossover between two parents using the specified method.

    Parameters
    ----------
    xp1 : np.ndarray
        First parent variables.
    xp2 : np.ndarray
        Second parent variables.
    bounds : list
        A two element list with the lower and upper bounds of each variables.
    method : str, optional
        crossover method, by default "sbx".
    eta : float, optional
        Factor for the crossover operator, by default 5.0.

    Returns
    -------
    tuple
        Two offspring variables.
    """
    # Some checks
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise ValueError("bounds must be a list [lower_bounds, upper_bounds]")
    if len(bounds[0]) != len(bounds[1]):
        raise ValueError("lower_bounds and upper_bounds must have the same length")
    if method not in _CROSSOVERS:
        warnings.warn(f"Unknown method '{method}', using 'ux' instead.", UserWarning)
        method = "ux"

    # Init offspring
    n = len(xp1)
    xo = np.zeros(n)

    # Make crossover
    for i in range(n):
        if xp2[i] != xp1[i]:
            xo[i] = _CROSSOVERS[method](xp1[i], xp2[i], bounds[0][i], bounds[1][i], eta)
        else:
            xo[i] = xp1[i]

    # Make sure offspring is within bounds
    xo = np.clip(xo, bounds[0], bounds[1])

    return xo
