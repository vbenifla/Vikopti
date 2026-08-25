import warnings
import numpy as np


def um(x: float, xl: float, xu: float, eta: float = None):
    """
    Random mutation.

    Parameters
    ----------
    x : float
        Variable to mutate.
    xl : float
        Lower bound for the variable.
    xu : float
        Upper bound for the variable.
    eta : float, optional
        Scaling factor, by default None.

    Returns
    -------
    float
        Mutant variable.
    """

    return np.random.uniform(xl, xu)


def nm(x: float, xl: float, xu: float, eta: float = 5.0):
    """
    Normal distribution mutation.

    Parameters
    ----------
    x : float
        Variable to mutate.
    xl : float
        Lower bound for the variable.
    xu : float
        Upper bound for the variable.
    eta : float, optional
        Scaling factor, by default 5.0.

    Returns
    -------
    float
        Mutant variable.
    """

    return np.random.normal(x, np.abs(xu - xl) / eta)


def pm(x: float, xl: float, xu: float, eta: float = 5.0):
    """
    Polynomial Mutation.

    Parameters
    ----------
    x : float
        Variable to mutate.
    xl : float
        Lower bound for the variable.
    xu : float
        Upper bound for the variable.
    eta : float, optional
        Scaling factor, by default 5.0.

    Returns
    -------
    float
        Mutant variable.
    """

    # Compute distance to boundaries
    d1 = (x - xl) / (xu - xl)
    d2 = (xu - x) / (xu - xl)

    # random pick
    r = np.random.rand()
    if r <= 0.5:
        d = (2 * r + (1 - 2 * r) * (1 - d1) ** (eta + 1)) ** (1 / (eta + 1)) - 1
    else:
        d = 1 - (2 * (1 - r + (r - 1 / 2) * (1 - d2) ** (eta + 1))) ** (1 / (eta + 1))
    return x + d * (xu - xl)


# Dictionary mapping method names to functions
_MUTATIONS = {"um": um, "nm": nm, "pm": pm}


def mutate(
    x: np.ndarray, bounds: np.ndarray, method="pm", pm: float = 0.5, eta: float = 5.0
):
    """
    Perform mutation operation on an individual variables using the specified method.

    Parameters
    ----------
    x : np.ndarray
        Individual variables.
    bounds : list
        A two element list with the lower and upper bounds of each variables.
    method : str, optional
        Mutation method, by default "pm".
    pm: float, optional
        Mutation probability, by default 0.5.
    eta : float, optional
        Scaling factor, by default 5.0.

    Returns
    -------
    np.ndarray
        Mutant variables.
    """
    
    # Some checks
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise ValueError("bounds must be a list [lower_bounds, upper_bounds]")
    if len(bounds[0]) != len(bounds[1]):
        raise ValueError("lower_bounds and upper_bounds must have the same length")
    if method not in _MUTATIONS:
        warnings.warn(f"Unknown method '{method}', using 'pm' instead.", UserWarning)
        method = "pm"

    # Init mutant
    n = len(x)
    xm = np.zeros(n)

    # Make mutation
    for i in range(n):
        if np.random.rand() <= pm:
            xm[i] = _MUTATIONS[method](x[i], bounds[0][i], bounds[1][i], eta)
        else:
            xm[i] = x[i]

    # Make sure mutant is within bounds
    xm = np.clip(xm, bounds[0], bounds[1])

    return xm
