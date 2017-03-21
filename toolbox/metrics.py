from keras import backend as K
import numpy as np


def psnr(y_true, y_pred):
    """Peak signal-to-noise ratio averaged over samples and channels."""
    mse = K.mean(K.square(y_true - y_pred), axis=(-3, -2))
    return K.mean(20 * K.log(255 / K.sqrt(mse)) / np.log(10))


def ssim(y_true, y_pred):
    """structural similarity measurement system."""
    ## K1, K2 are two constants, much smaller than 1
    K1 = 0.04
    K2 = 0.06
    
    ## mean, std, correlation
    mu_x = K.mean(y_pred)
    mu_y = K.mean(y_true)
    
    sig_x = K.std(y_pred)
    sig_y = K.std(y_true)
    sig_xy = (sig_x * sig_y) ** 0.5

    ## L, number of pixels, C1, C2, two constants
    L =  33
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    ssim = (2 * mu_x * mu_y + C1) * (2 * sig_xy * C2) * 1.0 / ((mu_x ** 2 + mu_y ** 2 + C1) * (sig_x ** 2 + sig_y ** 2 + C2))
    return ssim
