import keras.backend as K
from keras.metrics import mse
import numpy as np


def psnr(y_true, y_pred):
    return 20 * K.log(255 / K.sqrt(mse(y_true, y_pred))) / np.log(10)
