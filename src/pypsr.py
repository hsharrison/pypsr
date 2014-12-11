import numpy as np
from sklearn import metrics


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
    x = np.squeeze(x)
    if x.ndim != 1:
        raise ValueError('x(t) must be a 1-dimensional signal')

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
