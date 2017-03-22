from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import InputLayer
from keras.models import Sequential
import tensorflow as tf

from toolbox.layers import ImageRescale
from toolbox.layers import Conv2DSubPixel


def bicubic(x, scale=3):
    model = Sequential()
    model.add(InputLayer(input_shape=x.shape[-3:]))
    model.add(ImageRescale(scale, method=tf.image.ResizeMethod.BICUBIC))
    return model


def srcnn(x, f=[9, 1, 5], n=[64, 32], scale=3):
    """Build an SRCNN model.

    See https://arxiv.org/abs/1501.00092
    """
    assert len(f) == len(n) + 1
    model = bicubic(x, scale=scale)
    c = x.shape[-1]
    for ni, fi in zip(n, f):
        model.add(Conv2D(ni, fi, padding='same',
                         kernel_initializer='he_normal', activation='relu'))
    model.add(Conv2D(c, f[-1], padding='same',
                     kernel_initializer='he_normal'))
    return model


def fsrcnn(x, d=56, s=12, m=4, scale=3):
    """Build an FSRCNN model.

    See https://arxiv.org/abs/1608.00367
    """
    model = Sequential()
    model.add(InputLayer(input_shape=x.shape[-3:]))
    c = x.shape[-1]
    f = [5, 1] + [3] * m + [1]
    n = [d, s] + [s] * m + [d]
    for ni, fi in zip(n, f):
        model.add(Conv2D(ni, fi, padding='same',
                         kernel_initializer='he_normal', activation='relu'))
    model.add(Conv2DTranspose(c, 9, strides=scale, padding='same',
                              kernel_initializer='he_normal'))
    return model


def nsfsrcnn(x, d=56, s=12, m=4, scale=3, pos=1):
    """Build an FSRCNN model, but change deconv position.

    See https://arxiv.org/abs/1608.00367
    """
    model = Sequential()
    model.add(InputLayer(input_shape=x.shape[-3:]))
    c = x.shape[-1]
    f1 = [5, 1] + [3] * pos
    n1 = [d, s] + [s] * pos
    f2 = [3] * (m - pos - 1) + [1]
    n2 = [s] * (m - pos - 1) + [d]
    f3 = 9
    n3 = c
    for ni, fi in zip(n1, f1):
        model.add(Conv2D(ni, fi, padding='same',
                         kernel_initializer='he_normal', activation='relu'))
    model.add(Conv2DTranspose(s, 3, strides=scale, padding='same',
                              kernel_initializer='he_normal'))
    for ni, fi in zip(n2, f2):
        model.add(Conv2D(ni, fi, padding='same',
                         kernel_initializer='he_normal', activation='relu'))
    model.add(Conv2D(n3, f3, padding='same',
                         kernel_initializer='he_normal'))
    return model


def espcn(x, f=[5, 3, 3], n=[64, 32], scale=3):
    """Build an ESPCN model.

    See https://arxiv.org/abs/1609.05158
    """
    assert len(f) == len(n) + 1
    model = Sequential()
    model.add(InputLayer(input_shape=x.shape[1:]))
    c = x.shape[-1]
    for ni, fi in zip(n, f):
        model.add(Conv2D(ni, fi, padding='same',
                         kernel_initializer='he_normal', activation='tanh'))
    model.add(Conv2D(c * scale ** 2, f[-1], padding='same',
                     kernel_initializer='he_normal'))
    model.add(Conv2DSubPixel(scale))
    return model


def get_model(name):
    return globals()[name]
