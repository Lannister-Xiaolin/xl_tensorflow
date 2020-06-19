#!usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import Text, Tuple, Union
import tensorflow as tf
from xl_tensorflow.layers.actication import Swish


def parse_image_size(image_size: Union[Text, int, Tuple[int, int]]):
    """Parse the image size and return (height, width).

    Args:
      image_size: A integer, a tuple (H, W), or a string with HxW format.

    Returns:
      A tuple of integer (height, width).
    """
    if isinstance(image_size, int):
        # image_size is integer, with the same width and height.
        return (image_size, image_size)

    if isinstance(image_size, str):
        # image_size is a string with format WxH
        width, height = image_size.lower().split('x')
        return (int(height), int(width))

    if isinstance(image_size, tuple):
        return image_size

    raise ValueError('image_size must be an int, WxH string, or (height, width)'
                     'tuple. Was %r' % image_size)


def get_feat_sizes(image_size: Union[Text, int, Tuple[int, int]],
                   max_level: int):
    """Get feat widths and heights for all levels.

    Args:
      image_size: A integer, a tuple (H, W), or a string with HxW format.
      max_level: maximum feature level.

    Returns:
      feat_sizes: a list of tuples (height, width) for each level.
    """
    image_size = parse_image_size(image_size)
    feat_sizes = [{'height': image_size[0], 'width': image_size[1]}]
    feat_size = image_size
    for _ in range(1, max_level + 1):
        feat_size = ((feat_size[0] - 1) // 2 + 1, (feat_size[1] - 1) // 2 + 1)
        feat_sizes.append({'height': feat_size[0], 'width': feat_size[1]})
    return feat_sizes


def activation_fn(features: tf.Tensor, act_type: Text):
    """Customized non-linear activation type."""
    if act_type == 'swish':
        # t = lambda x:tf.nn.swish(x)
        return  tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(features)
        # 上面代码有点问题
        # return Swish()(features)
    elif act_type == 'swish_native':
        return features * tf.sigmoid(features)
    elif act_type == 'relu':
        return tf.nn.relu(features)
    elif act_type == 'relu6':
        return tf.nn.relu6(features)
    else:
        raise ValueError('Unsupported act_type {}'.format(act_type))
