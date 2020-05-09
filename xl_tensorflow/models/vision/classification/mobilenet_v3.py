#!usr/bin/env python3
# -*- coding: UTF-8 -*-
import os

from keras_applications.imagenet_utils import _obtain_input_shape
from keras_applications import correct_pad
from tensorflow.keras import backend, layers, models
from xl_tensorflow.layers import SEConvEfnet2D, HSwish, GlobalAveragePooling2DKeepDim, \
    CONV_KERNEL_INITIALIZER, DENSE_KERNEL_INITIALIZER


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _inverted_res_se_block(inputs, expansion=1, stride=1, alpha=1.0, filters=3,
                           block_id=0,
                           has_se=False, activation="relu", kernel_size=3):
    """
    inverted resnet with squeeze and excitation block, se ratio is 0.25
    """
    channel_axis = -1

    in_channels = backend.int_shape(inputs)[channel_axis]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'block_{}_'.format(block_id)

    if block_id:
        # Expand
        x = layers.Conv2D(int(expansion * in_channels),
                          kernel_size=1,
                          padding='same',
                          use_bias=False,
                          activation=None,
                          name=prefix + 'expand')(x)
        x = layers.BatchNormalization(axis=channel_axis,
                                      name=prefix + 'Expand_BN')(x)
        x = layers.ReLU(6., name=prefix + 'Expand_Relu')(x)
    else:
        prefix = 'Expanded_Conv_'

    # # Depthwise
    if stride == 2:
        x = layers.ZeroPadding2D(padding=correct_pad(backend, x, kernel_size),
                                 name=prefix + 'Pad')(x)
    x = layers.DepthwiseConv2D(kernel_size=kernel_size,
                               strides=stride,
                               activation=None,
                               use_bias=False,
                               padding='same' if stride == 1 else 'valid',
                               name=prefix + 'depthwise')(x)
    x = layers.BatchNormalization(epsilon=1e-3,
                                  momentum=0.99, name=prefix + 'depthwise_bn')(x)
    if activation == "relu":
        x = layers.ReLU(6., name=prefix + 'depthwise_relu')(x)
    else:
        x = HSwish(name=prefix + "depthwise_swish")(x)
    # Squeeze and excitation
    # comment: global average pooling is unvalid for lite gpu so we use general avgpooling instead
    if has_se:
        x = SEConvEfnet2D(expansion * in_channels if block_id else in_channels, se_ratio=0.25,
                          name=prefix + "seconv")(x)
    # Project
    x = layers.Conv2D(pointwise_filters,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      activation=None,
                      name=prefix + 'project_conv2d',
                      )(x)
    x = layers.BatchNormalization(name=prefix + 'project_bn')(x)

    if in_channels == pointwise_filters and stride == 1:
        return layers.Add(name=prefix + 'Add')([inputs, x])
    return x


def MobileNetV3(size, input_shape=None,
                alpha=1.0,
                include_top=True,
                weights='imagenet',
                input_tensor=None,
                pooling=None,
                classes=1000,
                name="mobilenetv3large",
                non_custom=False,
                force_relu=False,
                dropout_rate=0.2,
                **kwargs):
    """
    create model for mobilenet_v3
    hit: if you want using random input size, please set non_custom=False
    Args:
        size:
        input_shape:
        alpha:
        include_top:
        weights:
        input_tensor:
        pooling:
        classes:
        name: model name
        non_custom: using custom layers if True, otherwhise using foundation layers in tensorflow
        force_relu: force to using relu as activation function if True
        **kwargs:

    Returns:
        keras style funtional model
    """
    V3_Settings = {"small": [(
        dict(filters=16, alpha=alpha, stride=2, has_se=False, activation="relu",
             expansion=1, block_id=0, kernel_size=3),
        dict(filters=24, alpha=alpha, stride=2, has_se=False, activation="relu",
             expansion=4.5, block_id=1, kernel_size=3),
        dict(filters=24, alpha=alpha, stride=1, has_se=False, activation="relu",
             expansion=3.5, block_id=2, kernel_size=3),
        dict(filters=40, alpha=alpha, stride=2, has_se=True, activation="swish",
             expansion=4, block_id=3, kernel_size=5),
        dict(filters=40, alpha=alpha, stride=1, has_se=True, activation="wish",
             expansion=6, block_id=4, kernel_size=5),
        dict(filters=40, alpha=alpha, stride=1, has_se=True, activation="relu",
             expansion=6, block_id=5, kernel_size=5),
        dict(filters=48, alpha=alpha, stride=1, has_se=True, activation="swish",
             expansion=3, block_id=6, kernel_size=5),
        dict(filters=48, alpha=alpha, stride=1, has_se=True, activation="wish",
             expansion=3, block_id=7, kernel_size=5),
        dict(filters=96, alpha=alpha, stride=2, has_se=True, activation="relu",
             expansion=6, block_id=8, kernel_size=5),
        dict(filters=96, alpha=alpha, stride=1, has_se=True, activation="swish",
             expansion=6, block_id=9, kernel_size=5),
        dict(filters=96, alpha=alpha, stride=1, has_se=True, activation="wish",
             expansion=6, block_id=10, kernel_size=5),

    ), 576, 1024],

        "large": [(
            dict(filters=16, alpha=alpha, stride=1, has_se=False, activation="relu",
                 expansion=1, block_id=0, kernel_size=3),
            dict(filters=24, alpha=alpha, stride=2, has_se=False, activation="relu",
                 expansion=4, block_id=1, kernel_size=3),
            dict(filters=24, alpha=alpha, stride=1, has_se=False, activation="relu",
                 expansion=3, block_id=2, kernel_size=3),
            dict(filters=40, alpha=alpha, stride=2, has_se=True, activation="relu",
                 expansion=3, block_id=3, kernel_size=5),
            dict(filters=40, alpha=alpha, stride=1, has_se=True, activation="relu",
                 expansion=3, block_id=4, kernel_size=5),
            dict(filters=40, alpha=alpha, stride=1, has_se=True, activation="relu",
                 expansion=3, block_id=5, kernel_size=5),
            dict(filters=80, alpha=alpha, stride=2, has_se=False, activation="swish",
                 expansion=6, block_id=6, kernel_size=3),
            dict(filters=80, alpha=alpha, stride=1, has_se=False, activation="swish",
                 expansion=2.5, block_id=7, kernel_size=3),
            dict(filters=80, alpha=alpha, stride=1, has_se=False, activation="swish",
                 expansion=2.3, block_id=8, kernel_size=3),
            dict(filters=80, alpha=alpha, stride=1, has_se=False, activation="swish",
                 expansion=2.3, block_id=9, kernel_size=3),
            dict(filters=112, alpha=alpha, stride=1, has_se=True, activation="swish",
                 expansion=6, block_id=10, kernel_size=3),
            dict(filters=112, alpha=alpha, stride=1, has_se=True, activation="swish",
                 expansion=6, block_id=11, kernel_size=3),
            dict(filters=160, alpha=alpha, stride=2, has_se=True, activation="swish",
                 expansion=6, block_id=12, kernel_size=5),
            dict(filters=160, alpha=alpha, stride=1, has_se=True, activation="swish",
                 expansion=6, block_id=13, kernel_size=5),
            dict(filters=160, alpha=alpha, stride=1, has_se=True, activation="swish",
                 expansion=6, block_id=14, kernel_size=5),

        ), 960, 1280]}

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top` '
                         'as true, `classes` should be 1000')

    # If input_shape is None, infer shape from input_tensor
    if input_shape is None:
        default_size = 224
    # If input_shape is not None, assume default size
    else:
        if backend.image_data_format() == 'channels_first':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            rows = input_shape[0]
            cols = input_shape[1]
        if rows == cols and rows in [96, 128, 160, 192, 224]:
            default_size = rows
        else:
            default_size = 224
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=default_size,
                                      min_size=32,
                                      data_format=backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    img_input = layers.Input(shape=input_shape)
    channel_axis = -1
    first_block_filters = _make_divisible(16 * alpha, 8)
    x = layers.ZeroPadding2D(padding=correct_pad(backend, img_input, 3),
                             name='conv1_pad')(img_input)
    x = layers.Conv2D(first_block_filters, kernel_size=3, strides=(2, 2), padding='valid',
                      use_bias=False, name='conv1_first', )(x)
    x = layers.BatchNormalization(name='bn_conv1')(x)

    x = HSwish(name="conv1_swish")(x)
    for args in V3_Settings[size][0]:
        x = _inverted_res_se_block(x, **args)

    if alpha > 1.0:
        last_block_filters = _make_divisible(V3_Settings[size][1] * alpha, 8)
    else:
        last_block_filters = V3_Settings[size][1]
    x = layers.Conv2D(last_block_filters, kernel_size=1,
                      use_bias=False, name='conv2d_last',
                      )(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  name='bn_last_conv1')(x)

    x = HSwish(name="conv2d_last_swish")(x)
    x = GlobalAveragePooling2DKeepDim()(x)
    x = layers.Conv2D(V3_Settings[size][2], kernel_size=1, use_bias=False, name='conv2d_1x1_last',
                      )(x)
    x = HSwish(name="globalpooling_last_swish")(x)
    x = layers.Reshape(target_shape=(V3_Settings[size][2],))(x)

    if include_top:
        if dropout_rate and dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name='top_dropout')(x)
        x = layers.Dense(classes, activation='softmax',
                         use_bias=True, name='logits')(x)
    inputs = img_input
    model = models.Model(inputs, x,
                         name=name)
    return model


def MobileNetV3Large(input_shape=None,
                     alpha=1.0,
                     include_top=True,
                     weights='imagenet',
                     input_tensor=None,
                     pooling=None,
                     classes=1000,
                     dropout_rate=0.2,
                     name="mobilenetv3large",
                     **kwargs):
    return MobileNetV3("large", input_shape=input_shape,
                       alpha=alpha,
                       include_top=include_top,
                       weights=weights,
                       input_tensor=input_tensor,
                       pooling=pooling,
                       classes=classes,
                       name=name,
                       dropout_rate=dropout_rate,
                       **kwargs)


def MobileNetV3Small(input_shape=None,
                     alpha=1.0,
                     include_top=True,
                     weights='imagenet',
                     input_tensor=None,
                     pooling=None,
                     classes=1000,
                     dropout_rate=0.2,
                     name="mobilenetv3small",
                     **kwargs):
    return MobileNetV3("small", input_shape=input_shape,
                       alpha=alpha,
                       include_top=include_top,
                       weights=weights,
                       input_tensor=input_tensor,
                       pooling=pooling,
                       classes=classes,
                       name=name,
                       dropout_rate=dropout_rate,
                       **kwargs)


setattr(MobileNetV3Large, '__doc__', MobileNetV3.__doc__)
setattr(MobileNetV3Small, '__doc__', MobileNetV3.__doc__)


def main():
    import tensorflow as tf
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = MobileNetV3Small(weights=None, input_shape=(240, 240, 3), classes=1000, force_relu=False,
                                 non_custom=True)
    # model.save(f"./{model.name}.h5")
    print(model.summary())


if __name__ == '__main__':
    main()
