"""General utilities."""
import tensorflow as tf


def tf_eval(variable):
    """Evaluate a TensorFlow Variable.

    See https://www.tensorflow.org/versions/master/api_docs/python/state_ops/variables#Variable.eval
    """
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        return variable.eval()
