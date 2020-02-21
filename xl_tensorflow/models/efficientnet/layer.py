#!usr/bin/env python3
# -*- coding: UTF-8 -*-
import collections
import functools
import logging
import math

from tensorflow.keras import layers, backend
import tensorflow as tf
import numpy as np

GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate', 'data_format',
    'num_classes', 'width_coefficient', 'depth_coefficient', 'depth_divisor',
    'min_depth', 'survival_prob', 'relu_fn', 'batch_norm', 'use_se',
    'local_pooling', 'condconv_num_experts', 'clip_projection_output',
    'blocks_args'
])
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio', 'conv_type', 'fused_conv',
    'super_pixel', 'condconv'
])
# defaults will be a public argument for namedtuple in Python 3.7
# https://docs.python.org/3/library/collections.html#collections.namedtuple
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


def conv_kernel_initializer(shape, dtype=None, partition_info=None):
    """Initialization for convolutional kernels.
    The main difference with tf.variance_scaling_initializer is that
    tf.variance_scaling_initializer uses a truncated normal with an uncorrected
    standard deviation, whereas here we use a normal distribution. Similarly,
    tf.initializers.variance_scaling uses a truncated normal with
    a corrected standard deviation.
    Args:
      shape: shape of variable
      dtype: dtype of variable
      partition_info: unused
    Returns:
      an initialization for the variable
    """
    del partition_info
    kernel_height, kernel_width, _, out_filters = shape
    fan_out = int(kernel_height * kernel_width * out_filters)
    return tf.random_normal(
        shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)


def dense_kernel_initializer(shape, dtype=None, partition_info=None):
    """Initialization for dense kernels.
    This initialization is equal to
      tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_out',
                                      distribution='uniform').
    It is written out explicitly here for clarity.
    Args:
      shape: shape of variable
      dtype: dtype of variable
      partition_info: unused
    Returns:
      an initialization for the variable
    """
    del partition_info
    init_range = 1.0 / np.sqrt(shape[1])
    return tf.random_uniform(shape, -init_range, init_range, dtype=dtype)


def superpixel_kernel_initializer(shape, dtype='float32', partition_info=None):
    """Initializes superpixel kernels.
    This is inspired by space-to-depth transformation that is mathematically
    equivalent before and after the transformation. But we do the space-to-depth
    via a convolution. Moreover, we make the layer trainable instead of direct
    transform, we can initialization it this way so that the model can learn not
    to do anything but keep it mathematically equivalent, when improving
    performance.
    Args:
      shape: shape of variable
      dtype: dtype of variable
      partition_info: unused
    Returns:
      an initialization for the variable
    """
    del partition_info
    #  use input depth to make superpixel kernel.
    depth = shape[-2]
    filters = np.zeros([2, 2, depth, 4 * depth], dtype=dtype)
    i = np.arange(2)
    j = np.arange(2)
    k = np.arange(depth)
    mesh = np.array(np.meshgrid(i, j, k)).T.reshape(-1, 3).T
    filters[
        mesh[0],
        mesh[1],
        mesh[2],
        4 * mesh[2] + 2 * mesh[0] + mesh[1]] = 1
    return filters


def round_filters(filters, global_params):
    """Round number of filters based on depth multiplier."""
    orig_f = filters
    multiplier = global_params.width_coefficient
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    logging.info('round_filter input=%s output=%s', orig_f, new_filters)
    return int(new_filters)


def round_repeats(repeats, global_params):
    """Round number of filters based on depth multiplier."""
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))

class SEConvBlock(tf.keras.layers)


class MBConvBlock(tf.keras.layers.Layer):
    """A class of MBConv: Mobile Inverted Residual Bottleneck.
    Attributes:
      endpoints: dict. A list of internal tensors.
    """

    def __init__(self, block_args, global_params):
        """Initializes a MBConv block.
        Args:
          block_args: BlockArgs, arguments to create a Block.
          global_params: GlobalParams, a set of global parameters.
        """
        super(MBConvBlock, self).__init__()
        self._block_args = block_args
        self._batch_norm_momentum = global_params.batch_norm_momentum
        self._batch_norm_epsilon = global_params.batch_norm_epsilon
        self._batch_norm = global_params.batch_norm
        self._condconv_num_experts = global_params.condconv_num_experts
        self._data_format = global_params.data_format
        if self._data_format == 'channels_first':
            self._channel_axis = 1
            self._spatial_dims = [2, 3]
        else:
            self._channel_axis = -1
            self._spatial_dims = [1, 2]

        self._relu_fn = global_params.relu_fn or tf.nn.swish
        self._has_se = (
                global_params.use_se and self._block_args.se_ratio is not None and
                0 < self._block_args.se_ratio <= 1)

        self._clip_projection_output = global_params.clip_projection_output

        self.endpoints = None

        self.conv_cls = tf.layers.Conv2D
        self.depthwise_conv_cls = tf.keras.layers.DepthwiseConv2D
        if self._block_args.condconv:
            self.conv_cls = functools.partial(
                condconv_layers.CondConv2D, num_experts=self._condconv_num_experts)
            self.depthwise_conv_cls = functools.partial(
                condconv_layers.DepthwiseCondConv2D,
                num_experts=self._condconv_num_experts)

        # Builds the block accordings to arguments.
        self._build()

    def block_args(self):
        return self._block_args

    def _build(self):
        """Builds block according to the arguments."""
        if self._block_args.super_pixel == 1:
            self._superpixel = tf.layers.Conv2D(
                self._block_args.input_filters,
                kernel_size=[2, 2],
                strides=[2, 2],
                kernel_initializer=conv_kernel_initializer,
                padding='same',
                data_format=self._data_format,
                use_bias=False)
            self._bnsp = self._batch_norm(
                axis=self._channel_axis,
                momentum=self._batch_norm_momentum,
                epsilon=self._batch_norm_epsilon)

        if self._block_args.condconv:
            # Add the example-dependent routing function
            self._avg_pooling = tf.keras.layers.GlobalAveragePooling2D(
                data_format=self._data_format)
            self._routing_fn = tf.layers.Dense(
                self._condconv_num_experts, activation=tf.nn.sigmoid)

        filters = self._block_args.input_filters * self._block_args.expand_ratio
        kernel_size = self._block_args.kernel_size

        # Fused expansion phase. Called if using fused convolutions.
        self._fused_conv = self.conv_cls(
            filters=filters,
            kernel_size=[kernel_size, kernel_size],
            strides=self._block_args.strides,
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            data_format=self._data_format,
            use_bias=False)

        # Expansion phase. Called if not using fused convolutions and expansion
        # phase is necessary.
        self._expand_conv = self.conv_cls(
            filters=filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            data_format=self._data_format,
            use_bias=False)
        self._bn0 = self._batch_norm(
            axis=self._channel_axis,
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon)

        # Depth-wise convolution phase. Called if not using fused convolutions.
        self._depthwise_conv = self.depthwise_conv_cls(
            kernel_size=[kernel_size, kernel_size],
            strides=self._block_args.strides,
            depthwise_initializer=conv_kernel_initializer,
            padding='same',
            data_format=self._data_format,
            use_bias=False)

        self._bn1 = self._batch_norm(
            axis=self._channel_axis,
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon)

        if self._has_se:
            num_reduced_filters = max(
                1, int(self._block_args.input_filters * self._block_args.se_ratio))
            # Squeeze and Excitation layer.
            self._se_reduce = tf.layers.Conv2D(
                num_reduced_filters,
                kernel_size=[1, 1],
                strides=[1, 1],
                kernel_initializer=conv_kernel_initializer,
                padding='same',
                data_format=self._data_format,
                use_bias=True)
            self._se_expand = tf.layers.Conv2D(
                filters,
                kernel_size=[1, 1],
                strides=[1, 1],
                kernel_initializer=conv_kernel_initializer,
                padding='same',
                data_format=self._data_format,
                use_bias=True)

        # Output phase.
        filters = self._block_args.output_filters
        self._project_conv = self.conv_cls(
            filters=filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            data_format=self._data_format,
            use_bias=False)
        self._bn2 = self._batch_norm(
            axis=self._channel_axis,
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon)

    def _call_se(self, input_tensor):
        """Call Squeeze and Excitation layer.
        Args:
          input_tensor: Tensor, a single input tensor for Squeeze/Excitation layer.
        Returns:
          A output tensor, which should have the same shape as input.
        """
        se_tensor = tf.reduce_mean(input_tensor, self._spatial_dims, keepdims=True)
        se_tensor = self._se_expand(self._relu_fn(self._se_reduce(se_tensor)))
        logging.info('Built Squeeze and Excitation with tensor shape: %s',
                     (se_tensor.shape))
        return tf.sigmoid(se_tensor) * input_tensor

    def call(self, inputs, training=True, survival_prob=None):
        """Implementation of call().
        Args:
          inputs: the inputs tensor.
          training: boolean, whether the model is constructed for training.
          survival_prob: float, between 0 to 1, drop connect rate.
        Returns:
          A output tensor.
        """
        logging.info('Block input: %s shape: %s', inputs.name, inputs.shape)
        logging.info('Block input depth: %s output depth: %s',
                     self._block_args.input_filters,
                     self._block_args.output_filters)

        x = inputs

        fused_conv_fn = self._fused_conv
        expand_conv_fn = self._expand_conv
        depthwise_conv_fn = self._depthwise_conv
        project_conv_fn = self._project_conv

        if self._block_args.condconv:
            pooled_inputs = self._avg_pooling(inputs)
            routing_weights = self._routing_fn(pooled_inputs)
            # Capture routing weights as additional input to CondConv layers
            fused_conv_fn = functools.partial(
                self._fused_conv, routing_weights=routing_weights)
            expand_conv_fn = functools.partial(
                self._expand_conv, routing_weights=routing_weights)
            depthwise_conv_fn = functools.partial(
                self._depthwise_conv, routing_weights=routing_weights)
            project_conv_fn = functools.partial(
                self._project_conv, routing_weights=routing_weights)

        # creates conv 2x2 kernel
        if self._block_args.super_pixel == 1:
            with tf.variable_scope('super_pixel'):
                x = self._relu_fn(
                    self._bnsp(self._superpixel(x), training=training))
            logging.info(
                'Block start with SuperPixel: %s shape: %s', x.name, x.shape)

        if self._block_args.fused_conv:
            # If use fused mbconv, skip expansion and use regular conv.
            x = self._relu_fn(self._bn1(fused_conv_fn(x), training=training))
            logging.info('Conv2D: %s shape: %s', x.name, x.shape)
        else:
            # Otherwise, first apply expansion and then apply depthwise conv.
            if self._block_args.expand_ratio != 1:
                x = self._relu_fn(self._bn0(expand_conv_fn(x), training=training))
                logging.info('Expand: %s shape: %s', x.name, x.shape)

            x = self._relu_fn(self._bn1(depthwise_conv_fn(x), training=training))
            logging.info('DWConv: %s shape: %s', x.name, x.shape)

        if self._has_se:
            with tf.variable_scope('se'):
                x = self._call_se(x)

        self.endpoints = {'expansion_output': x}

        x = self._bn2(project_conv_fn(x), training=training)
        # Add identity so that quantization-aware training can insert quantization
        # ops correctly.
        x = tf.identity(x)
        if self._clip_projection_output:
            x = tf.clip_by_value(x, -6, 6)
        if self._block_args.id_skip:
            if all(
                    s == 1 for s in self._block_args.strides
            ) and self._block_args.input_filters == self._block_args.output_filters:
                # Apply only if skip connection presents.
                if survival_prob:
                    x = utils.drop_connect(x, training, survival_prob)
                x = tf.add(x, inputs)
        logging.info('Project: %s shape: %s', x.name, x.shape)
        return x
