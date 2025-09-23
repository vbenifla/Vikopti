import warnings
import numpy as np
from .utils import norm1D, Znorm


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
    pen = norm_viol.sum(axis=1)

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
    _obj_norm = Znorm(_obj)

    # No constraints
    if const.size == 0:
        _fit = _obj_norm
        _pen = np.zeros_like(_obj)

    # Constraint handling
    else:

        # Compute constraint violations
        _fit = np.zeros_like(_obj)
        _const = const[~is_nan]
        _pen = compute_penalty(_const)
        feasible = _pen == 0

        # At least one feasible
        if np.any(feasible):
            _fit[feasible] = _obj_norm[feasible]
            _fit[~feasible] = _obj_norm[feasible].max() + _pen[~feasible]

        # No feasible
        else:
            _fit = _pen.copy()

    # Minimize objective
    _fit = norm1D(-_fit)

    # Kill any NaNs
    fit[~is_nan] = _fit
    pen[~is_nan] = _pen

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
    # Get local optima
    optima = get_optima(f, dist, d, n)

    # Check feasible optima
    feasible = pen == 0
    if np.any(feasible):
        optima = [idx for idx in optima if feasible[idx]]

    # Scale fitness values to each local optima
    f_optima = np.zeros((f.shape[0], len(optima)))
    for i, idx in enumerate(optima):
        f_optima[:, i] = np.clip(f / f[idx], 0, 1)

    # Compute local optima proximity term
    eps = 1e-6
    prox = 1.0 / (dist[:, optima] + eps)
    prox /= np.sum(prox, axis=1, keepdims=True)

    # Compute weighted sum
    fs = np.sum(f_optima * prox, axis=1)

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
    neighbors = [np.where(row)[0] for row in mask]

    # Loop on individuals
    optima = []
    for i in range(len(f)):
        neigh = neighbors[i]

        # Make sure enough neighbors to compare
        if len(neigh) < n:
            neigh = np.argsort(dist[i, :])[1 : n + 1]

        # Check if optimum
        if np.all(f[i] > f[neigh]):
            optima.append(i)

    # Sort optima
    optima = sorted(set(optima), key=lambda i: f[i], reverse=True)

    # Some warning if anything goes wrong.
    if np.argmax(f) not in optima:
        warnings.warn("Optimum not identified", UserWarning)

    return optima
