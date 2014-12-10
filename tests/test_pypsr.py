import numpy as np
import pytest

import pypsr


def test_ami():
    a = np.arange(100)
    assert np.isclose(np.exp(pypsr.ami(a, a)), 10)
    assert np.isclose(pypsr.ami(a, np.ones_like(a)), 0)


def test_ami_two_column_input():
    a = np.arange(100)
    assert np.isclose(np.exp(pypsr.ami(np.vstack((a, a)).transpose())), 10)


def test_ami_different_length_signals():
    a = np.arange(10)
    with pytest.raises(ValueError):
        pypsr.ami(a, a[:-1])


def test_ami_only_one_signal():
    with pytest.raises(ValueError):
        pypsr.ami(np.arange(10))


def test_lagged_ami():
    lags, ami = pypsr.lagged_ami(np.arange(10), min_lag=0, max_lag=5)
    assert lags.dtype == 'int64'
    assert ami.dtype == 'float64'
    assert np.all(lags == np.arange(5))
    assert np.all(np.isclose(np.exp(ami), np.arange(10, 5, -1)))


def test_ami_unsqueezed_vector():
    a = np.arange(10)[:, np.newaxis]
    assert np.isclose(np.exp(pypsr.ami(a, a)), 10)
    assert np.isclose(pypsr.ami(a, np.ones_like(a)), 0)


def test_lagged_ami_unsqueezed_vector():
    a = np.arange(10)[:, np.newaxis]
    lags, ami = pypsr.lagged_ami(a)
    assert np.all(lags == np.arange(5))
    assert np.all(np.isclose(np.exp(ami), np.arange(10, 5, -1)))


def test_reconstruction():
    assert np.all(
        pypsr.reconstruct(np.arange(10), 1, 2) == np.vstack((np.arange(9), np.arange(1, 10))).transpose()
    )
    assert np.all(
        pypsr.reconstruct(np.arange(10), 2, 3)
        == np.vstack((np.arange(6), np.arange(2, 8), np.arange(4, 10))).transpose()
    )


def test_reconstruction_wrong_dimension_input():
    with pytest.raises(ValueError):
        pypsr.reconstruct(np.ones((10, 10)), 1, 2)


def test_reconstruction_too_long_lag():
    with pytest.raises(ValueError):
        pypsr.reconstruct(np.ones(10), 5, 2)
    with pytest.raises(ValueError):
        pypsr.reconstruct(np.ones(10), 2, 5)
