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


@tf.keras.utils.register_keras_serializable(package='Text')
class Base64ImageProcessLayer(tf.keras.layers.Layer):
    """
    Layer for deal with base64 input
    """

    def __init__(
            self,
            target_size=(512, 512),
            mean=tf.constant([0.485, 0.456, 0.406]),
            std=tf.constant([0.229, 0.224, 0.225]),
            **kwargs):
        """

        Args:
            target_size: target size to resize image
            mean: mean value for rgb, if None， value only devide by 255
            std: std value for rgb, if None， value only devide by 255
            **kwargs:
        """
        self.target_size = target_size
        self.mean = mean
        self.std = std
        super(Base64ImageProcessLayer, self).__init__(**kwargs)

    def preprocess_and_decode(self, img_str):
        img = tf.io.decode_base64(img_str)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.cast(img, tf.float32)
        img = img / 255.0
        if (self.mean is not None) and (self.std is not None):
            img = (img - self.mean) / self.std

        image_size = tf.cast(tf.shape(input=img)[0:2], tf.float32)
        scale = tf.minimum(
            self.target_size[0] / image_size[0], self.target_size[1] / image_size[1])
        scaled_size = tf.round(image_size * scale)
        image_scale = scaled_size / image_size
        scaled_image = tf.image.resize(
            img, tf.cast(scaled_size, tf.int32), method=tf.image.ResizeMethod.BILINEAR)

        output_image = tf.image.pad_to_bounding_box(scaled_image, 0, 0,
                                                    self.target_size[0], self.target_size[0])
        return [output_image, image_scale, image_size]

    def call(self, inputs, **kwargs):
        """
        Constructs the NMS graph.

        Args
            inputs : List of [boxes, classification, other[0], other[1], ...] tensors.
        """
        with tf.device("/cpu:0"):
            ouput_tensor, scales, image_sizes = tf.map_fn(lambda im: self.preprocess_and_decode(im[0]),
                                                          inputs, parallel_iterations=32,
                                                          dtype=["float32", "float32", "float32"])

        return [ouput_tensor, scales, image_sizes]

    def compute_output_shape(self, input_shape):
        """
        Computes the output shapes given the input shapes.

        Args
            input_shape : List of input shapes [boxes, classification].

        Returns
            List of tuples representing the output shapes:
            [filtered_boxes.shape, filtered_scores.shape, filtered_labels.shape, filtered_other[0].shape, filtered_other[1].shape, ...]
        """

        return [
            (input_shape[0], self.target_size[0], self.target_size[0], 3),
            (input_shape[0], 2),
            (input_shape[0], 2)
        ]

    def get_config(self):
        """
        Gets the configuration of this layer.

        Returns
            Dictionary containing the parameters of this layer.
        """
        config = super(Base64ImageProcessLayer, self).get_config()
        config.update({
            'target_size': self.target_size,
        })


@tf.keras.utils.register_keras_serializable(package='Text')
class ResizeImageProcessLayer(tf.keras.layers.Layer):
    """
    Layer for resize image and
    """

    def __init__(
            self,
            target_size=(512, 512),
            mean=tf.constant([0.485, 0.456, 0.406]),
            std=tf.constant([0.229, 0.224, 0.225]),
            **kwargs):
        """

        Args:
            target_size: target size to resize image
            mean: mean value for rgb, if None， value only devide by 255
            std: std value for rgb, if None， value only devide by 255
            **kwargs:
        """
        self.target_size = target_size
        self.mean = mean
        self.std = std
        super(ResizeImageProcessLayer, self).__init__(**kwargs)

    def preprocess_and_decode(self, img):
        img = tf.cast(img, tf.float32)
        img = img / 255.0
        if (self.mean is not None) and (self.std is not None):
            img = (img - self.mean) / self.std

        image_size = tf.cast(tf.shape(input=img)[0:2], tf.float32)
        scale = tf.minimum(
            self.target_size[0] / image_size[0], self.target_size[0] / image_size[1])
        scaled_size = tf.round(image_size * scale)
        image_scale = scaled_size / image_size
        scaled_image = tf.image.resize(
            img, tf.cast(scaled_size, tf.int32), method=tf.image.ResizeMethod.BILINEAR)

        output_image = tf.image.pad_to_bounding_box(scaled_image, 0, 0,
                                                    self.target_size[0], self.target_size[0])
        return [output_image, image_scale, image_size]

    def call(self, inputs, **kwargs):
        """
        Constructs the NMS graph.

        Args
            inputs : List of [boxes, classification, other[0], other[1], ...] tensors.
        """
        # with tf.device("/cpu:0"):
        ouput_tensor, scales, image_sizes = tf.map_fn(lambda im: self.preprocess_and_decode(im),
                                                      inputs, parallel_iterations=32,
                                                      dtype=["float32", "float32", "float32"])

        return [ouput_tensor, scales, image_sizes]

    def compute_output_shape(self, input_shape):
        """
        Computes the output shapes given the input shapes.

        Args
            input_shape : List of input shapes [boxes, classification].

        Returns
            List of tuples representing the output shapes:
            [filtered_boxes.shape, filtered_scores.shape, filtered_labels.shape, filtered_other[0].shape, filtered_other[1].shape, ...]
        """

        return [
            (input_shape[0], self.target_size[0], self.target_size[0], 3),
            (input_shape[0], 2),
            (input_shape[0], 2),
        ]

    def get_config(self):
        """
        Gets the configuration of this layer.

        Returns
            Dictionary containing the parameters of this layer.
        """
        config = super(ResizeImageProcessLayer, self).get_config()
        config.update({
            'target_size': self.target_size,
        })
