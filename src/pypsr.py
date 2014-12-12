from operator import sub

import numpy as np
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from toolz import curry


def global_false_nearest_neighbors(x, lag, min_dims=1, max_dims=10, **cutoffs):
    """
    Across a range of embedding dimensions $d$, embeds $x(t)$ with lag $\tau$, finds all nearest neighbors,
    and computes the percentage of neighbors that that remain neighbors when an additional dimension is unfolded.
    See [1] for more information.

    Parameters
    ----------
    x : array-like
        Original signal $x(t).
    lag : int
        Time lag $\tau$ in units of the sampling time $h$ of $x(t)$.
    min_dims : int, optional
        The smallest embedding dimension $d$ to test.
    max_dims : int, optional
        The largest embedding dimension $d$ to test.
    relative_distance_cutoff : float, optional
        The cutoff for determining neighborliness,
        in distance increase relative to the original distance between neighboring points.
        The default, 15, is suggested in [1] (p. 41).
    relative_radius_cutoff : float, optional
        The cutoff for determining neighborliness,
        in distance increase relative to the radius of the attractor.
        The default, 2, is suggested in [1] (p. 42).

    Returns
    -------
    dims : ndarray
        The tested dimensions $d$.
    gfnn : ndarray
        The percentage of nearest neighbors that are false neighbors at each dimension.

    See Also
    --------
    reconstruct

    References
    ----------
    [1] Arbanel, H. D. (1996). *Analysis of Observed Chaotic Data* (pp. 40-43). New York: Springer.

    """
    x = _vector(x)

    dimensions = np.arange(min_dims, max_dims + 1)
    false_neighbor_pcts = np.array([_gfnn(x, lag, n_dims, **cutoffs) for n_dims in dimensions])
    return dimensions, false_neighbor_pcts


def _gfnn(x, lag, n_dims, **cutoffs):
    # Global false nearest neighbors at a particular dimension.
    # Returns percent of all nearest neighbors that are still neighbors when the next dimension is unfolded.
    # Neighbors that can't be embedded due to lack of data are not counted in the denominator.
    offset = lag*n_dims
    is_true_neighbor = _is_true_neighbor(x, _radius(x), offset)
    return np.mean([
        not is_true_neighbor(indices, distance, **cutoffs)
        for indices, distance in _nearest_neighbors(reconstruct(x, lag, n_dims))
        if (indices + offset < x.size).all()
    ])


def _radius(x):
    # Per Arbanel (p. 42):
    # "the nominal 'radius' of the attractor defined as the RMS value of the data about its mean."
    return np.sqrt(((x - x.mean())**2).mean())


@curry
def _is_true_neighbor(
        x, attractor_radius, offset, indices, distance,
        relative_distance_cutoff=15,
        relative_radius_cutoff=2
):
    distance_increase = np.abs(sub(*x[indices + offset]))
    return (distance_increase / distance < relative_distance_cutoff and
            distance_increase / attractor_radius < relative_radius_cutoff)


def _nearest_neighbors(y):
    """
    Wrapper for sklearn.neighbors.NearestNeighbors.
    Yields the indices of the neighboring points, and the distance between them.

    """
    distances, indices = NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(y).kneighbors(y)
    for distance, index in zip(distances, indices):
        yield index, distance[1]


def reconstruct(x, lag, n_dims):
    """Phase-space reconstruction.

    Given a signal $x(t)$, dimensionality $d$, and lag $\tau$, return the reconstructed signal
    \[
        \mathbf{y}(t) = [x(t), x(t + \tau), \ldots, x(t + (d - 1)\tau)].
    \]

    Parameters
    ----------
    x : array-like
        Original signal $x(t)$.
    lag : int
        Time lag $\tau$ in units of the sampling time $h$ of $x(t)$.
    n_dims : int
        Embedding dimension $d$.

    Returns
    -------
    ndarray
        $\mathbf{y}(t)$ as an array with $d$ columns.

    """
    x = _vector(x)

    if lag * (n_dims - 1) >= x.shape[0] // 2:
        raise ValueError('longest lag cannot be longer than half the length of x(t)')

    lags = lag * np.arange(n_dims)
    return np.vstack(x[lag:lag - lags[-1] or None] for lag in lags).transpose()


def ami(x, y=None, n_bins=10):
    """Calculate the average mutual information between $x(t)$ and $y(t)$.

    Parameters
    ----------
    x : array-like
    y : array-like, optional
        $x(t)$ and $y(t)$.
        If only `x` is passed, it must have two columns;
        the first column defines $x(t)$ and the second $y(t)$.
    n_bins : int
        The number of bins to use when computing the joint histogram.

    Returns
    -------
    scalar
        Average mutual information between $x(t)$ and $y(t)$, in nats (natural log equivalent of bits).

    See Also
    --------
    lagged_ami

    References
    ----------
    Arbanel, H. D. (1996). *Analysis of Observed Chaotic Data* (p. 28). New York: Springer.

    """
    x, y = _vector_pair(x, y)
    if x.shape[0] != y.shape[0]:
        raise ValueError('timeseries must have the same length')

    return metrics.mutual_info_score(None, None, contingency=np.histogram2d(x, y, bins=n_bins)[0])


def lagged_ami(x, min_lag=0, max_lag=None, lag_step=1, n_bins=10):
    """Calculate the average mutual information between $x(t)$ and $x(t + \tau)$, at multiple values of $\tau$.

    Parameters
    ----------
    x : array-like
        $x(t)$.
    min_lag : int, optional
        The shortest lag to evaluate, in units of the sampling period $h$ of $x(t)$.
    max_lag : int, optional
        The longest lag to evaluate, in units of $h$.
    lag_step : int, optional
        The step between lags to evaluate, in units of $h$.
    n_bins : int
        The number of bins to use when computing the joint histogram in order to calculate mutual information.
        See |ami|.

    Returns
    -------
    lags : ndarray
        The evaluated lags $\tau_i$, in units of $h$.
    amis : ndarray
        The average mutual information between $x(t)$ and $x(t + \tau_i)$.

    See Also
    --------
    ami

    """
    if max_lag is None:
        max_lag = x.shape[0]//2
    lags = np.arange(min_lag, max_lag, lag_step)

    amis = [ami(reconstruct(x, lag, 2), n_bins=n_bins) for lag in lags]
    return lags, np.array(amis)


def _vector_pair(a, b):
    a = np.squeeze(a)
    if b is None:
        if a.ndim != 2 or a.shape[1] != 2:
            raise ValueError('with one input, array must have be 2D with two columns')
        a, b = a[:, 0], a[:, 1]
    return a, np.squeeze(b)


def _vector(x):
    x = np.squeeze(x)
    if x.ndim != 1:
        raise ValueError('x(t) must be a 1-dimensional signal')
    return x
