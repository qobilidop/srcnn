from keras.engine.topology import Layer
import tensorflow as tf


class ImageResize(Layer):
    def __init__(self, size, method, **kwargs):
        self.size = size
        self.method = method
        super().__init__(**kwargs)

    def call(self, x):
        return tf.image.resize_images(x, size=self.size, method=self.method)

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-3], *self.size, input_shape[-1])
