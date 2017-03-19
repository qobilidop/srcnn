from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import InputLayer
from keras.layers import PReLU
from keras.models import Sequential

from toolbox.metrics import psnr


def compile(model, optimizer='adam', loss='mse', metrics=(psnr,), **kwargs):
    """Compile a model with default settings."""
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)
    return model


def build_srcnn(x, f=(9, 1, 5), n=(64, 32)):
    """Build an SRCNN model.

    See https://arxiv.org/abs/1501.00092
    """
    assert len(f) == len(n) + 1
    model = Sequential()
    model.add(InputLayer(input_shape=x.shape[-3:]))
    c = x.shape[-1]
    for ni, fi in zip(n, f):
        model.add(Conv2D(ni, fi, padding='same',
                         kernel_initializer='he_normal', activation='relu'))
    model.add(Conv2D(c, f[-1], padding='same',
                     kernel_initializer='he_normal'))
    return model


def build_fsrcnn(x, d=56, s=12, m=4, k=3):
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
                         kernel_initializer='he_normal'))
        model.add(PReLU())
    model.add(Conv2DTranspose(c, 9, strides=k, padding='same',
                              kernel_initializer='he_normal'))
    return model


def build_espcn(x, f=(5, 3, 3), n=(64, 32), r=3):
    """Build an ESPCN model.

    See https://arxiv.org/abs/1609.05158
    """
    assert len(f) == len(n) + 1
    model = Sequential()
    model.add(InputLayer(input_shape=x.shape[-3:]))
    c = x.shape[-1]
    for ni, fi in zip(n, f):
        model.add(Conv2D(ni, fi, padding='same',
                         kernel_initializer='he_normal', activation='tanh'))
    model.add(Conv2D(c * r ** 2, f[-1], padding='same',
                     kernel_initializer='he_normal'))
    return model
