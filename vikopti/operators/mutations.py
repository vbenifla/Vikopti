import numpy as np


def nm(x: float, b0: float, b1: float, eta: float = 5.0):
    """
    Perform Normal Mutation.

    Parameters
    ----------
    x : float
        The variable to mutate.
    b0 : float
        The lower bound for the variable.
    b1 : float
        The upper bound for the variable.
    eta : float, optional
        Scaling factor, by default 5.0.

    Returns
    -------
    float
        The mutated variable.
    """
    return np.random.normal(x, np.abs(b1 - b0) / eta)


def cm(x: float, b0: float, b1: float, eta: float = 5.0):
    """
    Perform Cauchy Mutation.

    Parameters
    ----------
    x : float
        The variable to mutate.
    b0 : float
        The lower bound for the variable.
    b1 : float
        The upper bound for the variable.
    eta : float, optional
        Scaling factor, by default 5.0.

    Returns
    -------
    float
        The mutated variable.
    """
    return x + np.random.standard_cauchy() * eta * (b1 - b0)


def pm(x: float, b0: float, b1: float, eta: float = 5.0):
    """
    Perform Polynomial Mutation.

    Parameters
    ----------
    x : float
        The variable to mutate.
    b0 : float
        The lower bound for the variable.
    b1 : float
        The upper bound for the variable.
    eta : float, optional
        Distribution index, by default 5.0.

    Returns
    -------
    float
        The mutated variable.
    """
    d1 = (x - b0) / (b1 - b0)
    d2 = (b1 - x) / (b1 - b0)
    r = np.random.rand()
    if r <= 0.5:
        d = (2 * r + (1 - 2 * r) * (1 - d1) ** (eta + 1)) ** (1 / (eta + 1)) - 1
    else:
        d = 1 - (2 * (1 - r + (r - 1 / 2) * (1 - d2) ** (eta + 1))) ** (1 / (eta + 1))
    return x + d * (b1 - b0)


# Dictionary mapping method names to functions
MUTATIONS = {"nm": nm, "cm": cm, "pm": pm}


def mutate(x: np.ndarray, bounds: np.ndarray, method="pm", p: float = 0.5, eta: float = 5.0):
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
    p: float, optional
        Mutation probability, by default 0.5.
    eta : float, optional
        Factor for the mutation operator, by default 5.0.

    Returns
    -------
    np.ndarray
        One mutant.
    """

    n = len(x)
    xm = np.copy(x)
    for i in range(n):
        if np.random.rand() <= p:
            xm[i] = MUTATIONS[method](x[i], bounds[0][i], bounds[1][i], eta)
            if xm[i] < bounds[0][i] or xm[i] > bounds[1][i]:
                xm[i] = x[i]

    return xm
