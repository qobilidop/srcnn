import keras.backend as K
from keras.metrics import mse
import numpy as np


def psnr(y_true, y_pred):
    """Peak signal-to-noise ratio."""
    return 20 * K.log(255 / K.sqrt(mse(y_true, y_pred))) / np.log(10)



def ssim(y_true, y_pred):
    """structural similarity measurement system."""
    ## K1, K2 are two constants, much smaller than 1
    K1 = 0.04
    K2 = 0.06
    
    ## mean, std, correlation
    mu_x = np.mean(y_pred)
    mu_y = np.mean(y_true)

    sig_x = np.std(y_pred)
    sig_y = np.std(y_true)
    sig_xy = np.sqrt(sig_x * sig_y)

    L = len(y_true)
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    ssim = (2 * mu_x * mu_y + C1) * (2 * sig_xy * C2) * 1.0 / ((mu_x ** 2 + mu_y ** 2 + C1) * (sig_x ** 2 + sig_y ** 2 + C2))
    return ssim
