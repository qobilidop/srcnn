import numpy as np

from toolbox.array import periodic_shuffling
from toolbox.array import inverse_periodic_shuffling


def test_periodic_shuffling():
    S, H, W, C, r = 11, 7, 5, 3, 2
    a = np.random.rand(S, H, W, C * r ** 2)
    b = periodic_shuffling(a, r)
    assert b.shape == (S, H * r, W * r, C)
    c = inverse_periodic_shuffling(b, r)
    assert np.all(np.isclose(a, c))
