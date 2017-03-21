"""Image processing tools."""
import numpy as np
from PIL import Image


def array_to_img(x, mode='YCbCr'):
    return Image.fromarray(x.astype('uint8'), mode=mode)


def bicubic_rescale(image, scale):
    if isinstance(scale, (float, int)):
        size = (np.array(image.size) * scale).astype(int)
    return image.resize(size, resample=Image.BICUBIC)


def modcrop(image, scale):
    size = np.array(image.size)
    size -= size % scale
    return image.crop([0, 0, *size])
