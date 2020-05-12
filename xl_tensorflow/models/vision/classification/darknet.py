#!usr/bin/env python3
# -*- coding: UTF-8 -*-
from functools import wraps
from tensorflow.keras.regularizers import l2
from xl_tensorflow.utils.common import compose
from tensorflow.keras import layers, models, backend
import os
from keras_applications.imagenet_utils import _obtain_input_shape
from keras_applications.imagenet_utils import preprocess_input as _preprocess_input
from xl_tensorflow.utils.common import DENSE_KERNEL_INITIALIZER
from xl_tensorflow.layers.actication import Mish, get_mish

BASE_WEIGTHS_PATH = (
    'https://github.com/keras-team/keras-applications/'
    'releases/download/densenet/')
DENSENET121_WEIGHT_PATH = (
        BASE_WEIGTHS_PATH +
        'densenet121_weights_tf_dim_ordering_tf_kernels.h5')


@wraps(layers.Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return layers.Conv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.1))


def DarknetConv2D_BN_Mish(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and Mish activation."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        layers.BatchNormalization(),
        # get_mish()
        Mish())


def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = compose(
            DarknetConv2D_BN_Leaky(num_filters // 2, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters, (3, 3)))(x)
        x = layers.Add()([x, y])
    return x


def short_cut(x, num_filters, num_blocks, filter_half=False):
    """shot cut connection for yolov4"""
    for i in range(num_blocks):
        y = compose(
            DarknetConv2D_BN_Mish(num_filters // 2 if filter_half else num_filters, (1, 1)),
            DarknetConv2D_BN_Mish(num_filters, (3, 3)))(x)
        x = layers.Add()([x, y])
    return x


def dense_block_body(x, num_filters, num_blocks, filter_half=False):
    """

    Args:
        x:
        num_filters:
        num_blocks:
        filter_half:

    Returns:

    """
    x = layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Mish(num_filters * 2 if num_blocks > 1 else num_filters, (3, 3), strides=(2, 2))(x)
    dense_x = DarknetConv2D_BN_Mish(num_filters, (1, 1))(x)
    dense_x = short_cut(dense_x, num_filters, num_blocks, filter_half)
    dense_x = DarknetConv2D_BN_Mish(num_filters, (1, 1))(dense_x)
    x = DarknetConv2D_BN_Mish(num_filters, (1, 1))(x)
    x = layers.Concatenate()([dense_x, x])
    x = DarknetConv2D_BN_Mish(num_filters * 2 if num_blocks > 1 else num_filters, (1, 1))(x)
    return x


def darknet_body(x):
    '''
        Darknent body having 52 Convolution2D layers
    '''
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x


def cspdarknet_body(x):
    """
    csp darknet
    Args:
        x:

    Returns:

    """
    x = DarknetConv2D_BN_Mish(32, (3, 3))(x)
    x = dense_block_body(x, 64, 1, True)
    x = dense_block_body(x, 64, 2, False)
    x = dense_block_body(x, 128, 8, False)
    x = dense_block_body(x, 256, 8, False)
    x = dense_block_body(x, 512, 4, False)
    return x


def DarkNet(
        model_name,
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        dropout_rate=None,
        **kwargs):
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Todo 完善不同模型
    if model_name == "darknet53":
        x = darknet_body(img_input)
    elif model_name == "cspdarknet53":
        x = cspdarknet_body(img_input)
    else:
        x = img_input

    # top
    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        if dropout_rate and dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name='top_dropout')(x)
        x = layers.Dense(classes,
                         activation='softmax',
                         kernel_initializer=DENSE_KERNEL_INITIALIZER,
                         name='probs')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)

        # Ensure that the model takes into account

    inputs = img_input

    # Create model.
    model = models.Model(inputs, x, name=model_name)

    # Load weights.
    if weights == 'imagenet':
        # if include_top:
        #     file_name = model_name.replace("lite", "") + '_weights_tf_dim_ordering_tf_kernels_autoaugment.h5'
        #     file_hash = WEIGHTS_HASHES[model_name.replace("lite", "")][0]
        # else:
        #     file_name = model_name.replace("lite", "") + '_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'
        #     file_hash = WEIGHTS_HASHES[model_name.replace("lite", "")][1]
        # weights_path = utils.get_file(file_name,
        #                               BASE_WEIGHTS_PATH + file_name,
        #                               cache_subdir='models',
        #                               file_hash=file_hash)
        # model.load_weights(weights_path)
        pass
    elif weights is not None:
        model.load_weights(weights)

    return model


def DarkNet53(include_top=True,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              classes=1000,
              **kwargs):
    return DarkNet('darknet53',
                   include_top, weights,
                   input_tensor, input_shape,
                   pooling, classes,
                   **kwargs)


def CspDarkNet53(include_top=True,
                 weights='imagenet',
                 input_tensor=None,
                 input_shape=None,
                 pooling=None,
                 classes=1000,
                 **kwargs):
    return DarkNet('cspdarknet53',
                   include_top, weights,
                   input_tensor, input_shape,
                   pooling, classes,
                   **kwargs)
