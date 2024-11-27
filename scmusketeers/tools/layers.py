############################
#      A NETTOYER
############################


# from keras.engine.base_layer import InputSpec
import keras.backend as K
import tensorflow as tf

# from keras.engine.topology import Layer
from keras.layers import Dense, Lambda

nan2zeroLayer = Lambda(lambda x: tf.where(tf.is_nan(x), tf.zeros_like(x), x))
ColwiseMultLayer = Lambda(
    lambda l: l[0] * tf.reshape(l[1], (-1, 1)), name="reconstruction"
)


def reverse_gradient(X, hp_lambda):
    """Flips the sign of the incoming gradient during training."""
    try:
        reverse_gradient.num_calls += 1
    except AttributeError:
        reverse_gradient.num_calls = 1

    grad_name = "GradientReversal%d" % reverse_gradient.num_calls

    @tf.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
        return [tf.negative(grad) * hp_lambda]

    g = K.get_session().graph
    with g.gradient_override_map({"Identity": grad_name}):
        y = tf.identity(X)

    return y


@tf.custom_gradient
def grad_reverse(x):
    y = tf.identity(x)

    def custom_grad(dy):
        return -dy

    return y, custom_grad


class GradReverse(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        return grad_reverse(x)
