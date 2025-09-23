import warnings
import numpy as np


def um(x: float, xl: float, xu: float, eta: float = 5.0):
    """
    Perform random mutation.

    Parameters
    ----------
    x : float
        The variable to mutate.
    xl : float
        The lower bound for the variable.
    xu : float
        The upper bound for the variable.

    Returns
    -------
    float
        The mutated variable.
    """
    return np.random.uniform(xl, xu)


def nm(x: float, xl: float, xu: float, eta: float = 5.0):
    """
    Perform normal distribution mutation.

    Parameters
    ----------
    x : float
        The variable to mutate.
    xl : float
        The lower bound for the variable.
    xu : float
        The upper bound for the variable.
    eta : float, optional
        Scaling factor, by default 5.0.

    Returns
    -------
    float
        The mutated variable.
    """
    return np.random.normal(x, np.abs(xu - xl) / eta)


def pm(x: float, xl: float, xu: float, eta: float = 5.0):
    """
    Perform Polynomial Mutation.

    Parameters
    ----------
    x : float
        The variable to mutate.
    xl : float
        The lower bound for the variable.
    xu : float
        The upper bound for the variable.
    eta : float, optional
        Distribution index, by default 5.0.

    Returns
    -------
    float
        The mutated variable.
    """
    d1 = (x - xl) / (xu - xl)
    d2 = (xu - x) / (xu - xl)
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
    Mutate one individual using the specified method.

    Parameters
    ----------
    x : np.ndarray
        The individual to mutate.
    bounds : list
        A two element list with the lower and upper bounds of each variables.
    method : str, optional
        mutation method, by default "sbx".
    pm: float, optional
        Mutation probability, by default 0.5.
    eta : float, optional
        Factor for the mutation operator, by default 5.0.

    Returns
    -------
    np.ndarray
        One mutant.
    """
    # Some checks
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise ValueError("bounds must be a list [lower_bounds, upper_bounds]")
    if len(bounds[0]) != len(bounds[1]):
        raise ValueError("lower_bounds and upper_bounds must have the same length")
    if method not in _MUTATIONS:
        warnings.warn(f"Unknown method '{method}', using 'um' instead.", UserWarning)
        method = "um"

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
