import tensorflow as tf

from sac.util import parametric_relu

ACTIVATIONS = dict(
    relu=tf.nn.relu,
    leaky=tf.nn.leaky_relu,
    elu=tf.nn.elu,
    selu=tf.nn.selu,
    prelu=parametric_relu,
    sigmoid=tf.sigmoid,
    tanh=tf.tanh,
    none=None,
)