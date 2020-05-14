#!usr/bin/env python3
# -*- coding: UTF-8 -*-
from tensorflow.keras import layers, backend
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='Text')
class GlobalAveragePooling2DKeepDim(layers.GlobalAveragePooling2D):
    """Global average pooling operation for spatial data, this class keep dim for output

    Arguments:
        data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, height, width, channels)` while `channels_first`
          corresponds to inputs with shape
          `(batch, channels, height, width)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".

    Input shape:
      - If `data_format='channels_last'`:
        4D tensor with shape `(batch_size, rows, cols, channels)`.
      - If `data_format='channels_first'`:
        4D tensor with shape `(batch_size, channels, rows, cols)`.

    Output shape:
      4D tensor with shape `(batch_size,1,1, channels)`.
    """

    def __init__(self, **kwargs):
        super(GlobalAveragePooling2DKeepDim, self).__init__(**kwargs)

    def call(self, inputs):
        if self.data_format == 'channels_last':
            return backend.mean(inputs, axis=[1, 2], keepdims=True)
        else:
            return backend.mean(inputs, axis=[2, 3], keepdims=True)

    def get_config(self):
        config = super(GlobalAveragePooling2DKeepDim, self).get_config()
        return config


@tf.keras.utils.register_keras_serializable(package='Text')
class SEConvEfnet2D(layers.Layer):
    """
    This  Squeeze and Excitation layer for efficientnet
    Args:
        input_channels: 输入通道数
        se_ratio: squeeze ratio
    """

    def __init__(self, input_channels, se_ratio, name="SEConvEfnet2D", **kwargs):
        super(SEConvEfnet2D, self).__init__(name=name, **kwargs)
        num_reduced_filters = max(1, int(input_channels * se_ratio))
        self.se_ratio = se_ratio
        self.global_pooling = GlobalAveragePooling2DKeepDim()
        self.conv_kernel_initializer = {
            'class_name': 'VarianceScaling',
            'config': {
                'scale': 2.0,
                'mode': 'fan_out',
                'distribution': 'normal'
            }}
        self._se_reduce = layers.Conv2D(num_reduced_filters, 1, strides=[1, 1],
                                        kernel_initializer=self.conv_kernel_initializer,
                                        activation=None, padding="same", use_bias=True)
        self.activation = layers.ReLU()
        self._se_expand = layers.Conv2D(input_channels, 1, strides=[1, 1],
                                        kernel_initializer=self.conv_kernel_initializer,
                                        activation="hard_sigmoid", padding="same",
                                        use_bias=True)
        self._multiply = layers.Multiply()

    def call(self, inputs, **kwargs):
        se_tensor = self.global_pooling(inputs)
        se_tensor = self._se_expand(self.activation(self._se_reduce(se_tensor)))
        x = self._multiply([se_tensor, inputs])
        return x

    def get_config(self):
        config = super(SEConvEfnet2D, self).get_config()
        config.update({'se_ratio': self.se_ratio})
        return config
