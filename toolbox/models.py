from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import ZeroPadding2D
from keras.models import Sequential
from keras.layers.advanced_activations import PReLU

from toolbox.metrics import psnr


def compile_srcnn(c=1, f1=9, f2=1, f3=5, n1=64, n2=32):
    """Compile an SRCNN model.

    See https://arxiv.org/abs/1501.00092.
    """
    model = Sequential()
    model.add(Conv2D(n1, f1, padding='same', kernel_initializer='he_normal',
                     activation='relu', input_shape=(None, None, c)))
    model.add(Conv2D(n2, f2, padding='same', kernel_initializer='he_normal',
                     activation='relu'))
    model.add(Conv2D(c, f3, padding='same', kernel_initializer='he_normal'))
    model.compile(optimizer='adam', loss='mse', metrics=[psnr])
    return model

def compile_fsrcnn(input_shape, c=1, f1=5, f2=1, f3=3, f4=9, n1=56, n2=12, n3=1, s1=3):
    """Compile an FSRCNN model.
    See http://link.springer.com/chapter/10.1007/978-3-319-46475-6_25
    """
    model = Sequential()
    model.add(Conv2D(n1, f1, padding='valid', kernel_initializer='he_normal',input_shape=input_shape))
    model.add(PReLU())
    model.add(Conv2D(n2, f2, kernel_initializer='he_normal'))
    model.add(PReLU())
    model.add(Conv2D(n2, f3, padding='same', kernel_initializer='he_normal'))
    model.add(PReLU())
    model.add(Conv2D(n2, f3, padding='same', kernel_initializer='he_normal'))
    model.add(PReLU())
    model.add(Conv2D(n2, f3, padding='same', kernel_initializer='he_normal'))
    model.add(PReLU())
    model.add(Conv2D(n2, f3, padding='same', kernel_initializer='he_normal'))
    model.add(PReLU())
    model.add(Conv2D(n1, f2, kernel_initializer='he_normal'))
    model.add(PReLU())
    model.add(Conv2DTranspose(n3, f4, strides=s1, padding='same', kernel_initializer='he_normal'))
    model.compile(optimizer='adam', loss='mse', metrics=[psnr])
    return model
