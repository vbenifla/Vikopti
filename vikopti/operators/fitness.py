import numpy as np


def norm1D(arr):
    """
    Norm a 1D numpy array between 0 and 1

    Parameters
    ----------
    arr : np.ndarray
        Array to normalize.

    Returns
    -------
    np.ndarray
        Normalized array.
    """
    # Get min/max values
    min_val = arr.min()
    max_val = arr.max()

    # Normalize the array
    narr = (arr - min_val) / (max_val - min_val)

    return narr


def compute_fitness(
    obj: np.ndarray,
    const: np.ndarray,
    alpha: float = 1.0,
):
    """
    Compute fitness values.

    Parameters
    ----------
    obj : np.ndarray
        Objective values.
    const : np.ndarray
        Constraint values.
    alpha : float, optional
        Penalty factor, by default 1.0.

    Returns
    -------
    np.ndarray
        Fitness values.
    """
    # Consider only none NAN values
    is_nan = np.isnan(obj)
    _obj = obj[~is_nan]
    _f = np.zeros_like(_obj)

    # Constraint handling
    if const.size > 0:
        _const = const[~is_nan]
        _pen = compute_penalty(_const)
        _is_feasible = _pen == 0.0
        if np.any(_is_feasible):
            _f[_is_feasible] = _obj[_is_feasible]
            _f[~_is_feasible] = _obj[_is_feasible].max() + alpha * _pen[~_is_feasible]
        else:
            _f = _pen
    else:
        _pen = np.full(len(_obj), 0.0)
        _is_feasible = np.full(len(_obj), True)
        _f = _obj

    # The objective is minimized, so the fitness is maximized
    # Normalize values between 0 and 1
    f = np.zeros_like(obj)
    f[~is_nan] = norm1D(-_f)
    is_feasible = np.full(len(obj), False)
    is_feasible[~is_nan] = _is_feasible
    pen = np.zeros_like(obj)
    pen[~is_nan] = _pen

    # Kill Nan values
    f[is_nan] = 0
    pen[is_nan] = 0
    is_feasible[is_nan] = False

    return (f, is_feasible, pen)


def compute_penalty(const: np.ndarray):
    """
    Compute penalty values.

    Parameters
    ----------
    const : np.ndarray
        Constraints values.

    Returns
    -------
    np.ndarray
        Penalty values.
    """
    # Compute constraints violations
    v = const.copy()
    v[v <= 0.0] = 0.0

    # Normalize the violations
    vmax = np.max(v, axis=0)
    mask = vmax != 0
    v[:, mask] /= vmax[mask]

    # Compute the penalty term
    pen = np.sum(v, axis=-1)

    return pen


def scale_fitness(f: np.ndarray, dist: np.ndarray, is_feasible, d: float = 0.1, n: int = 4):
    """
    Perform the fitness scaling operation used in the CMNGA.
    Scales the fitness values to each local optima identified.

    Parameters
    ----------
    f : np.ndarray
        Fitness values of the population, between 0 and 1.
    dist : np.ndarray
        Distance matrix with the pair-wise distance to each individual.
    d : float, optional
        Minimum distance threshold, by default 0.1.
    n : int, optional
        Minimum number of neighbors, by default 4.

    Returns
    -------
    tuple
        Scaled fitness values and indices of local optima.
    """
    # Get local optima
    ids = get_optima(f, dist, is_feasible, d, n)

    # Scale fitness to each local optima
    fmaxs = np.zeros((f.shape[0], len(ids)))
    for i, id in enumerate(ids):
        fmaxs[:, i] = f / f[id]
        fmaxs[:, i][fmaxs[:, i] > 1.0] = 1.0
        m = np.median(fmaxs[:, i][fmaxs[:, i] > 0.0])
        m = min(m, 0.999)
        m = max(m, 0.001)
        p = np.log(0.5) / np.log(m)
        fmaxs[:, i] = np.power(fmaxs[:, i], p)

    # Compute local optima proximity term
    dist_mins = dist[:, ids]
    inv_dist = 1.0 / (0.0001 + dist_mins)
    prox_matrix = inv_dist / np.sum(inv_dist, axis=1, keepdims=True)

    # Compute the weighted sum as scaled fitness
    fs = np.einsum("ij,ij->i", prox_matrix, fmaxs)

    return fs, ids


def get_optima(f: np.ndarray, dist: np.ndarray, is_feasible, d: float = 0.1, n: int = 4):
    """
    Identify local optima based on neighbors' fitness values.

    Parameters
    ----------
    f : np.ndarray
        Fitness values of the population, between 0 and 1.
    dist : np.ndarray
        Distance matrix with the pair-wise distance to each individual.
    d : float, optional
        Minimum distance threshold, by default 0.1.
    n : int, optional
        Minimum number of neighbors, by default 4.

    Returns
    -------
    list
        Indices of local optima.
    """
    # Get mask of neighbors within d_mins for each individual
    mask = dist <= d
    np.fill_diagonal(mask, False)
    neighbors = [np.where(row)[0] for row in mask]

    # Loop on individuals
    max_ids = []
    for i in range(len(f)):

        # Ensure at least n neighbors for comparison
        if len(neighbors[i]) < n:
            neighbor = np.argsort(dist[i, :])[1 : n + 1]
        else:
            neighbor = neighbors[i]

        # Compare fitness values and identify local minima
        if np.all(f[i] > f[neighbor]):
            if is_feasible[i]:
                max_ids.append(i)

    # Make sure maximum is identified first
    max_id = int(np.argmax(f))
    if max_id not in max_ids:
        max_ids.insert(0, max_id)
    else:
        max_ids.remove(max_id)
        max_ids.insert(0, max_id)

    return max_ids
