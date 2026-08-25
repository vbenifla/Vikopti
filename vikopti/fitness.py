import warnings
import numpy as np
from .utils import norm1D


def compute_penalty(const: np.ndarray):
    """
    Compute penalty based on constraints violations.

    Parameters
    ----------
    const : np.ndarray
        Constraints values of the population.

    Returns
    -------
    tuple
        Penalty values.
    """
    # Compute constraint violations
    viol = np.clip(const, 0, None)

    # Norm to each maximum violation
    max_viol = viol.max(axis=0, keepdims=True)
    norm_viol = np.divide(viol, max_viol, out=np.zeros_like(viol), where=max_viol > 0)

    # Compute penalty as sum of all violations
    pen = np.sum(norm_viol**2, axis=1)

    return pen


def compute_fitness(obj: np.ndarray, const: np.ndarray):
    """
    Compute fitness based on objectives and constraints.

    Parameters
    ----------
    obj : np.ndarray
        Objective values of the population.
    const : np.ndarray
        Constraints values of the population.

    Returns
    -------
    tuple
        Fitness and penalty values.
    """
    # Initialize fitness and penalty
    fit = np.zeros_like(obj[:, 0])
    pen = np.full_like(obj[:, 0], np.nan)

    # Identify NaNs
    is_nan = np.isnan(obj[:, 0])
    _obj = obj[:, 0][~is_nan]

    # No constraints
    if const.size == 0:
        _fit = _obj
        _pen = np.zeros_like(_obj)

    # Constraint handling
    else:
        _const = const[~is_nan]
        _fit, _pen = cht(_obj, _const)

    # Minimize objective
    _fit = norm1D(-_fit)

    # Kill any NaNs
    fit[~is_nan] = _fit
    pen[~is_nan] = _pen

    return fit, pen


def cht(obj: np.ndarray, const: np.ndarray, method="max"):
    """
    Compute fitness based on objectives and constraints.

    Parameters
    ----------
    obj : np.ndarray
        Objective values of the population.
    const : np.ndarray
        Constraints values of the population.

    Returns
    -------
    tuple
        Fitness and penalty values.
    """

    # Compute constraint violations
    fit = np.zeros_like(obj)

    # Compare unfeasible designs to worst feasible
    if method == "max":

        # Norm objective and compute norm penalty
        obj_norm = norm1D(obj)
        pen = compute_penalty(const)
        feasible = pen == 0

        # At least one feasible
        if np.any(feasible):
            fit[feasible] = obj_norm[feasible]
            fit[~feasible] = obj_norm[feasible].max() + pen[~feasible]

        # No feasible
        else:
            fit = pen.copy()

    # Blend objective and penalty
    else:
        w1 = 100
        w2 = 100
        viol = np.sum(np.clip(const, 0, None), axis=1)
        n_viol = (const > 0).sum(axis=1)
        pen = w1 * viol + w2 * n_viol
        fit = obj + pen

    return fit, pen


def scale_fitness(
    f: np.ndarray, pen: np.ndarray, dist: np.ndarray, d: float = 0.1, n: int = 10
):
    """
    Scales fitness values to each local optima identified, as in M. Hall. (2012).

    Parameters
    ----------
    f : np.ndarray
        Fitness values of the population.
    dist : np.ndarray
        Distance matrix between individuals of the population.
    d : float, optional
        Distance threshold for defining neighborhood, by default 0.1.
    n : int, optional
        Minimum number of neighbors to consider, by default 4.

    Returns
    -------
    tuple
        Scaled fitness values and local optima indices.
    """
    # Some epsilon
    eps = 1e-12

    # Get local optima
    optima = get_optima(f, dist, d, n)

    # Scale fitness values to each local optima
    fss = np.zeros((f.shape[0], len(optima)))
    for i, idx in enumerate(optima):
        fso = np.clip(f / f[idx], 0, 1)
        mask = (fso != 0) & (fso != 1)
        m = np.median(np.concatenate(([0], fso[mask], [1])))
        m = np.clip(m, eps, 1 - eps)
        p = np.log(0.5) / np.log(m)
        fss[:, i] = np.power(fso, p)

    # Compute local optima proximity term
    prox = 1.0 / (dist[:, optima] + eps)
    prox /= np.sum(prox, axis=1, keepdims=True)

    # Compute weighted sum
    fs = np.sum(fss * prox, axis=1)

    return fs, optima


def get_optima(f: np.ndarray, dist: np.ndarray, d: float = 0.1, n: int = 4):
    """
    Identify local optima based on neighborhood fitness.

    Parameters
    ----------
    f : np.ndarray
        Fitness values of the population.
    dist : np.ndarray
        Distance matrix between individuals of the population.
    d : float, optional
        Distance threshold for defining neighborhood, by default 0.1.
    n : int, optional
        Minimum number of neighbors to consider, by default 4.

    Returns
    -------
    list of int
        Indices of individuals identified as local optima, sorted in descending order of fitness.
    """
    # Get neighbors within d_mins for each individual
    mask = dist <= d
    np.fill_diagonal(mask, False)
    idx_neighbors = [np.where(row)[0] for row in mask]

    # Loop on individuals
    optima = []
    for i in range(len(f)):
        neighbors = idx_neighbors[i]

        # Make sure enough neighbors to compare
        if len(neighbors) < n:
            neighbors = np.argsort(dist[i, :])[1 : n + 1]

        # Check if optimum
        if np.all(f[i] > f[neighbors]):
            optima.append(i)
    
    # Make sure at least optima is there!
    if np.argmax(f) not in optima:
        optima.append(np.argmax(f))

    # Sort optima
    optima = sorted(set(optima), key=lambda i: f[i], reverse=True)

    return optima
