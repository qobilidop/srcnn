"""Array operators."""
import numpy as np


def periodic_shuffling(array, r):
    """Periodic shuffling operator.

    See https://arxiv.org/abs/1609.05158 equation 4.
    """
    H, W, Crr = array.shape[-3:]
    if Crr % (r ** 2) != 0:
        raise ValueError
    C = Crr // r // r
    new_shape = (H * r, W * r, C)
    xv, yv, cv = np.meshgrid(*[range(i) for i in new_shape], indexing='ij')
    xv, yv, cv = xv // r, yv // r, C * r * (yv % r) + C * (xv % r) + cv
    return array[..., xv, yv, cv]


def inverse_periodic_shuffling(array, r):
    """Inverse periodic shuffling operator.

    We derived it ourselves.
    """
    Hr, Wr, C = array.shape[-3:]
    if Hr % r !=0 or Wr % r != 0:
        raise ValueError
    H = Hr // r
    W = Wr // r
    Crr = C * r * r
    new_shape = (H, W, Crr)
    xv, yv, cv = np.meshgrid(*[range(i) for i in new_shape], indexing='ij')
    xv, yv, cv = xv * r + cv // C % r, yv * r + cv // C // r, cv % C
    return array[..., xv, yv, cv]
