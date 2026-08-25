import warnings
import numpy as np


def ux(x1: float, x2: float, xl: float, xu: float, eta: float = None):
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
    eta : float, optional
        Scaling factor, by default None.

    Returns
    -------
    float
        Offspring variable.
    """

    return np.random.uniform(x1, x2)


def bx(x1: float, x2: float, xl: float, xu: float, eta: float = 0.1):
    """
    Blend crossover between two parent variables.

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
        Scaling factor, by default 0.1.

    Returns
    -------
    float
        Offspring variable.
    """

    # Blend the individuals
    gamma = (1.0 + 2.0 * eta) * np.random.uniform() - eta

    # Random pick
    if np.random.uniform() < 0.5:
        xo = (1.0 - gamma) * x1 + gamma * x2
    else:
        xo = gamma * x1 + (1.0 - gamma) * x2

    return xo


def sbx(x1: float, x2: float, xl: float, xu: float, eta: float = 5.0):
    """
    Simulated Binary Crossover (SBX) between two parent variables.

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
        Scaling factor, by default 5.0.

    Returns
    -------
    float
        Offspring variable.
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

    # Random pick
    if np.random.rand() < 0.5:
        xo = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
    else:
        xo = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)

    return xo


def pnx(x1: float, x2: float, xl: float, xu: float, eta: float = 5.0):
    """
    Parent-Centric Normal Crossover (PNX) between two parent variables.

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
        Scaling factor, by default 5.0.

    Returns
    -------
    float
        Offspring variable.
    """

    # Check eta
    if eta == 0:
        eta = 1e-6

    # Compute delta
    delta = np.abs(x2 - x1)

    # Random pick
    if np.random.rand() < 0.5:
        xo = np.random.normal(x1, delta / eta)
    else:
        xo = np.random.normal(x2, delta / eta)

    return xo


# Dictionary mapping method names to functions
_CROSSOVERS = {"ux": ux, "bx": bx, "sbx": sbx, "pnx": pnx}


def cross(
    xp1: np.ndarray, xp2: np.ndarray, bounds: np.ndarray, method="sbx", eta: float = 1.0
):
    """
    Perform crossover operation between two parents variables using the specified method.

    Parameters
    ----------
    xp1 : np.ndarray
        First parent variables.
    xp2 : np.ndarray
        Second parent variables.
    bounds : list
        A two element list with the lower and upper bounds of each variables.
    method : str, optional
        Crossover method, by default "sbx".
    eta : float, optional
        Scaling factor, by default 1.0.

    Returns
    -------
    np.ndarray
        Offspring variables.
    """

    # Some checks
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise ValueError("bounds must be a list [lower_bounds, upper_bounds]")
    if len(bounds[0]) != len(bounds[1]):
        raise ValueError("lower_bounds and upper_bounds must have the same length")
    if method not in _CROSSOVERS:
        warnings.warn(f"Unknown method '{method}', using 'sbx' instead.", UserWarning)
        method = "sbx"

    # Initialize offspring array
    n = len(xp1)
    xo = np.zeros(n)

    # Make crossover
    for i in range(n):

        # If parents are not the same
        if xp2[i] != xp1[i]:
            xo[i] = _CROSSOVERS[method](xp1[i], xp2[i], bounds[0][i], bounds[1][i], eta)
        else:
            xo[i] = xp1[i]

    # Make sure offspring is within bounds
    xo = np.clip(xo, bounds[0], bounds[1])

    return xo
