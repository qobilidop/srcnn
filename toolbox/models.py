from keras.layers import Conv2D
from keras.models import Sequential


def srcnn(input_shape, c=1, f1=9, f2=1, f3=5, n1=64, n2=32):
    model = Sequential()
    model.add(Conv2D(nb_filter=n1, nb_row=f1, nb_col=f1, activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(nb_filter=n2, nb_row=f2, nb_col=f2, activation='relu'))
    model.add(Conv2D(nb_filter=c, nb_row=f3, nb_col=f3))
    return model
