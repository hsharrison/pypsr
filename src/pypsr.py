import numpy as np
from sklearn.metrics import mutual_info_score


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

    return mutual_info_score(None, None, contingency=np.histogram2d(x, y, bins=n_bins)[0])


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

    amis = [ami(x[lag:], np.roll(x, lag)[lag:], n_bins=n_bins) for lag in lags]
    return lags, np.array(amis)


def _vector_pair(a, b):
    a = np.asarray(a)
    if b is None:
        a = np.squeeze(a)
        if a.ndim != 2 or a.shape[1] != 2:
            raise ValueError('with one input, array must have be 2D with two columns')
        a, b = a[:, 0], a[:, 1]
    return a, np.asarray(b)
