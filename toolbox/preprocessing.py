"""Image preprocessing tools."""
import numpy as np


def modcrop(image, scale):
    size = np.array(image.shape)
    size -= size % scale
    image = image[:size[0], :size[1]]
    return image
