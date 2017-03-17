"""Image processing tools."""
import numpy as np
from PIL import Image


def array_to_img(x, mode='YCbCr'):
    return Image.fromarray(x.astype('uint8'), mode=mode)


def bicubic_resize(image, size):
    if isinstance(size, (float, int)):
        size = (np.array(image.size) * size).astype(int)
    return image.resize(size, resample=Image.BICUBIC)


def modcrop(image, scale):
    size = np.array(image.size)
    size -= size % scale
    return image.crop([0, 0, *size])
