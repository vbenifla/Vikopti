import numpy as np


def sbx(x1: float, x2: float, b0: float, b1: float, eta: float = 5.0):
    """
    Perform the Simulated Binary Crossover (SBX).

    Parameters
    ----------
    x1 : float
        First parent variable.
    x2 : float
        Second parent variable.
    b0 : float
        Lower bound for the variable.
    b1 : float
        Upper bound for the variable.
    eta : float, optional
        Distribution index for SBX, by default 5.0.

    Returns
    -------
    tuple
        Two offspring variables.
    """
    x1, x2 = min(x1, x2), max(x1, x2)
    u = np.random.rand()
    delta = min((x1 - b0), (b1 - x2))
    gamma = 1 + 2 * delta / (x2 - x1)
    alpha = 2 - gamma ** (-eta - 1)
    if u <= 1 / alpha:
        beta = alpha * u ** (1.0 / (eta + 1))
    else:
        beta = (1.0 / (2.0 - alpha * u)) ** (1.0 / (eta + 1))

    if np.random.rand() < 0.5:
        xo1 = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
        xo2 = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
    else:
        xo2 = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
        xo1 = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)

    return xo1, xo2


def pnx(x1: float, x2: float, b0: float, b1: float, eta: float = 5.0):
    """
    Perform Parent-Centric Normal Crossover (PNX).

    Parameters
    ----------
    x1 : float
        First parent variable.
    x2 : float
        Second parent variable.
    b0 : float
        Lower bound for the variable.
    b1 : float
        Upper bound for the variable.
    eta : float, optional
        Scaling factor for PNX, by default 5.0.

    Returns
    -------
    tuple
        Two offspring variables.
    """
    if eta == 0:
        eta = 1e-6
    xo1 = np.random.normal(x1, np.abs(x2 - x1) / eta)
    xo2 = np.random.normal(x2, np.abs(x2 - x1) / eta)
    return xo1, xo2


# Dictionary mapping method names to functions
CROSSOVERS = {"sbx": sbx, "pnx": pnx}


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
    n = len(xp1)
    xo1 = np.copy(xp1)
    xo2 = np.copy(xp2)
    for i in range(n):
        if xp2[i] != xp1[i]:
            xo1[i], xo2[i] = CROSSOVERS[method](
                xp1[i], xp2[i], bounds[0][i], bounds[1][i], eta
            )
            if xo1[i] < bounds[0][i] or xo1[i] > bounds[1][i]:
                xo1[i] = xp1[i]
            if xo2[i] < bounds[0][i] or xo2[i] > bounds[1][i]:
                xo2[i] = xp2[i]

    return xo1, xo2
