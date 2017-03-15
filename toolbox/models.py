from keras.layers import Conv2D
from keras.models import Sequential

from toolbox.metrics import psnr


def compile_srcnn(input_shape, c=1, f1=9, f2=1, f3=5, n1=64, n2=32):
    """Compile an SRCNN model.

    See https://arxiv.org/abs/1501.00092.
    """
    model = Sequential()
    model.add(Conv2D(nb_filter=n1, nb_row=f1, nb_col=f1, init='he_normal',
                     activation='relu', input_shape=input_shape))
    model.add(Conv2D(nb_filter=n2, nb_row=f2, nb_col=f2, init='he_normal',
                     activation='relu'))
    model.add(Conv2D(nb_filter=c, nb_row=f3, nb_col=f3, init='he_normal'))
    model.compile(optimizer='adam', loss='mse', metrics=[psnr])
    return model
