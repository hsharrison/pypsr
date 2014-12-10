import numpy as np

import pypsr


def test_ami():
    a = np.arange(100)
    assert np.isclose(np.exp(pypsr.ami(a, a)), 10)
    assert np.isclose(pypsr.ami(a, np.ones_like(a)), 0)


def test_lagged_ami():
    lags, ami = pypsr.lagged_ami(np.arange(10), min_lag=0, max_lag=5)
    assert lags.dtype == 'int64'
    assert ami.dtype == 'float64'
    assert np.all(lags == np.arange(5))
    assert np.all(np.isclose(np.exp(ami), np.arange(10, 5, -1)))
