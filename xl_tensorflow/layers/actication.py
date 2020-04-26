#!usr/bin/env python3
# -*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow.keras import layers,backend
from tensorflow.python.keras.utils import tf_utils


class HSwish(layers.Layer):
    def __init__(self, **kwargs):
        super(HSwish, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        # alpha is used for leaky relu slope in activations instead of
        # negative_slope.
        return tf.multiply(backend.sigmoid(inputs), tf.nn.relu6(inputs + 3) / 6)

    def get_config(self):
        base_config = super(HSwish, self).get_config()
        return base_config

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

class Swish(layers.Layer):
    def __init__(self, **kwargs):
        super(Swish, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        # alpha is used for leaky relu slope in activations instead of
        # negative_slope.
        return tf.multiply(backend.sigmoid(inputs), inputs)

    def get_config(self):
        base_config = super(Swish, self).get_config()
        return base_config

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape



def get_swish(**kwargs):
    def swish(x):
        """Swish activation function: x * sigmoid(x).
        Reference: [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
        """
        return tf.nn.swish(x)

    return swish