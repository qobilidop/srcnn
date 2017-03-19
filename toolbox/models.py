from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.models import Sequential

from toolbox.metrics import psnr


def compile(model, optimizer='adam', loss='mse', metrics=[psnr], **kwargs):
    """Compile a model with default settings."""
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)
    return model


def srcnn(c=1, f1=9, f2=1, f3=5, n1=64, n2=32):
    """Build an SRCNN model.

    See https://arxiv.org/abs/1501.00092
    """
    model = Sequential()
    model.add(Conv2D(n1, f1, padding='same', kernel_initializer='he_normal',
                     activation='relu', input_shape=(None, None, c)))
    model.add(Conv2D(n2, f2, padding='same', kernel_initializer='he_normal',
                     activation='relu'))
    model.add(Conv2D(c, f3, padding='same', kernel_initializer='he_normal'))
    return model


def fsrcnn(c=1, d=56, s=12, m=4, k=3):
    """Build an FSRCNN model.

    See http://link.springer.com/chapter/10.1007/978-3-319-46475-6_25
    """
    conv_params = dict(padding='same', kernel_initializer='he_normal',
                       activation='relu')
    model = Sequential()
    model.add(Conv2D(d, 5, input_shape=(None, None, c), **conv_params))
    model.add(Conv2D(s, 1, **conv_params))
    for i in range(m):
        model.add(Conv2D(3, s, **conv_params))
    model.add(Conv2D(d, 1, **conv_params))
    model.add(Conv2DTranspose(1, 9, strides=k, padding='same',
                              kernel_initializer='he_normal'))
    return model


def espcn(c=1, f1=5, f2=3, f3=3, n1=64, n2=32, r=3):
    """Build an ESPCN model.

    See https://arxiv.org/abs/1609.05158
    """
    model = Sequential()
    model.add(Conv2D(n1, f1, padding='same', kernel_initializer='he_normal',
                     activation='tanh', input_shape=(None, None, c)))
    model.add(Conv2D(n2, f2, padding='same', kernel_initializer='he_normal',
                     activation='tanh'))
    model.add(Conv2D(c * r ** 2, f3, padding='same',
                     kernel_initializer='he_normal'))
    return model
