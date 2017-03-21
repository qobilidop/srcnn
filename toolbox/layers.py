from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf


custom_layers = {}


class ImageRescale(Layer):
    def __init__(self, scale, method=tf.image.ResizeMethod.BICUBIC,
                 trainable=False, **kwargs):
        self.scale = scale
        self.method = method
        super().__init__(trainable=trainable, **kwargs)

    def compute_size(self, shape):
        size = np.array(shape)[[1, 2]] * self.scale
        return tuple(size.astype(int))

    def call(self, x):
        size = self.compute_size(x.shape.as_list())
        return tf.image.resize_images(x, size, method=self.method)

    def compute_output_shape(self, input_shape):
        size = self.compute_size(input_shape)
        return (input_shape[0], *size, input_shape[3])

    def get_config(self):
        config = super().get_config()
        config['scale'] = self.scale
        config['method'] = self.method
        return config


custom_layers['ImageRescale'] = ImageRescale


class Conv2DSubPixel(Layer):
    """Sub-pixel convolution layer.

    See https://arxiv.org/abs/1609.05158
    """
    def __init__(self, scale, trainable=False, **kwargs):
        self.scale = scale
        super().__init__(trainable=trainable, **kwargs)

    def call(self, t):
        r = self.scale
        new_shape = self.compute_output_shape(t.shape.as_list())
        C = new_shape[-1]
        xv, yv, cv = np.meshgrid(*[range(size) for size in new_shape[1:]],
                                 indexing='ij')
        # See equation 4
        xv, yv, cv = xv // r, yv // r, C * r * (yv % r) + C * (xv % r) + cv
        indices = list(zip(xv.flatten(), yv.flatten(), cv.flatten()))
        t = tf.transpose(t, perm=[1, 2, 3, 0])
        t = tf.gather_nd(t, indices)
        t = tf.transpose(t, perm=[1, 0])
        return tf.reshape(t, (-1,) + new_shape[1:])

    def compute_output_shape(self, input_shape):
        r = self.scale
        H, W, Crr = np.array(input_shape[1:])
        if Crr % (r ** 2) != 0:
            raise ValueError
        return (input_shape[0], H * r, W * r, Crr // (r ** 2))

    def get_config(self):
        config = super().get_config()
        config['scale'] = self.scale
        return config


custom_layers['Conv2DSubPixel'] = Conv2DSubPixel
