"""Image preprocessing tools."""
import numpy as np
from PIL.Image import BICUBIC


def bicubic_resize(image, size):
    if isinstance(size, float):
        size = (np.array(image.size) * size).astype(int)
    return image.resize(size, resample=BICUBIC)


def modcrop(image, scale):
    size = np.array(image.size)
    size -= size % scale
    return image.crop([0, 0, *size])
