from keras.backend import tf
from keras.layers import Conv2D
from keras.models import Sequential
import numpy as np

from toolbox.metrics import psnr


def bicubic_interpolation(x, scale=3):
    new_size = np.array(x.shape)[[1, 2]].astype(int) * scale
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        return tf.image.resize_bicubic(x, size=new_size).eval()


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
