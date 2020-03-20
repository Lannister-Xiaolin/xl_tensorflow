#!usr/bin/env python3
# -*- coding: UTF-8 -*-
from functools import wraps

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from ..efficientnet import EfficientNetB4, EfficientNetB3, \
    EfficientNetLiteB4, mb_conv_block, BlockArgs, get_swish, EfficientNetLiteB1, EfficientNetB0
from .utils import compose


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = compose(
            DarknetConv2D_BN_Leaky(num_filters // 2, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters, (3, 3)))(x)
        x = Add()([x, y])
    return x


def darknet_body(x):
    '''Darknent body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x


def make_last_layers(x, num_filters, out_filters):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)
    y = compose(
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D(out_filters, (1, 1)))(x)
    return x, y


def yolo_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras."""
    darknet = Model(inputs, darknet_body(inputs))
    x, y1 = make_last_layers(darknet.output, 512, num_anchors * (num_classes + 5))

    x = compose(
        DarknetConv2D_BN_Leaky(256, (1, 1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, num_anchors * (num_classes + 5))

    x = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, num_anchors * (num_classes + 5))

    return Model(inputs, [y1, y2, y3])


def make_last_layers_efficientnet(x, block_args, prefix):
    num_filters = block_args.input_filters * block_args.expand_ratio
    x = compose(
        tf.keras.layers.Conv2D(num_filters,
                               kernel_size=1,
                               padding='same',
                               use_bias=False),
        tf.keras.layers.BatchNormalization(
            epsilon=1e-3,
            momentum=0.9),
        tf.keras.layers.ReLU(6.),
        lambda x: mb_conv_block(x, block_args, activation=get_swish(), drop_rate=0.2, prefix="1_" + prefix),
        tf.keras.layers.Conv2D(num_filters,
                               kernel_size=1,
                               padding='same',
                               use_bias=False),
        tf.keras.layers.BatchNormalization(
            epsilon=1e-3,
            momentum=0.9),
        tf.keras.layers.ReLU(6.),
        lambda x: mb_conv_block(x, block_args, activation=get_swish(), drop_rate=0.2, prefix="2_" + prefix),
        tf.keras.layers.Conv2D(num_filters,
                               kernel_size=1,
                               padding='same',
                               use_bias=False),
        tf.keras.layers.BatchNormalization(
            epsilon=1e-3,
            momentum=0.9),
        tf.keras.layers.ReLU(6.))(x)
    y = compose(
        lambda x: mb_conv_block(x, block_args, activation=get_swish(), drop_rate=0.2, prefix="3_" + prefix),
        tf.keras.layers.Conv2D(block_args.output_filters,
                               kernel_size=1,
                               padding='same',
                               use_bias=False))(x)
    return x, y


def yolo_efficientnetb4_body(inputs, number_anchors, number_classes):
    block_args = BlockArgs(kernel_size=3,
                           num_repeat=1,
                           input_filters=512,
                           output_filters=number_anchors * (number_classes + 5),
                           expand_ratio=1,
                           id_skip=True,
                           se_ratio=0.25,
                           strides=[1, 1])
    backbone = EfficientNetB4(include_top=False, input_tensor=inputs, weights=None)
    x, y1 = make_last_layers_efficientnet(backbone.output, block_args, "y1_")
    x = compose(
        tf.keras.layers.Conv2D(256,
                               kernel_size=1,
                               padding='same',
                               use_bias=False,
                               name='block_20_conv'),
        tf.keras.layers.BatchNormalization(momentum=0.9,
                                           name='block_20_BN'),
        tf.keras.layers.ReLU(6., name='block_20_relu6'),
        tf.keras.layers.UpSampling2D(2))(x)
    block_args = block_args._replace(input_filters=256)
    x = tf.keras.layers.Concatenate()(
        [x, backbone.get_layer('block6a_expand_activation').output])
    x, y2 = make_last_layers_efficientnet(x, block_args, "y2_")
    x = compose(
        tf.keras.layers.Conv2D(128,
                               kernel_size=1,
                               padding='same',
                               use_bias=False,
                               name='block_24_conv'),
        tf.keras.layers.BatchNormalization(
            momentum=0.9,
            name='block_24_BN'),
        tf.keras.layers.ReLU(6., name='block_24_relu6'),
        tf.keras.layers.UpSampling2D(2))(x)
    block_args = block_args._replace(input_filters=128)
    x = tf.keras.layers.Concatenate()(
        [x, backbone.get_layer('block4a_expand_activation').output])
    x, y3 = make_last_layers_efficientnet(x, block_args, "y3_")
    return Model(inputs, [y1, y2, y3])


def yolo_efficientnetliteb4_body(inputs, number_anchors, number_classes):
    block_args = BlockArgs(kernel_size=3,
                           num_repeat=1,
                           input_filters=512,
                           output_filters=number_anchors * (number_classes + 5),
                           expand_ratio=1,
                           id_skip=True,
                           se_ratio=None,
                           strides=[1, 1])
    backbone = EfficientNetLiteB4(include_top=False, input_tensor=inputs, weights=None)
    x, y1 = make_last_layers_efficientnet(backbone.output, block_args, "y1_")
    x = compose(
        tf.keras.layers.Conv2D(256,
                               kernel_size=1,
                               padding='same',
                               use_bias=False,
                               name='block_20_conv'),
        tf.keras.layers.BatchNormalization(momentum=0.9,
                                           name='block_20_BN'),
        tf.keras.layers.ReLU(6., name='block_20_relu6'),
        tf.keras.layers.UpSampling2D(2))(x)
    block_args = block_args._replace(input_filters=256)
    x = tf.keras.layers.Concatenate()(
        [x, backbone.get_layer('block6a_expand_activation').output])
    x, y2 = make_last_layers_efficientnet(x, block_args, "y2_")
    x = compose(
        tf.keras.layers.Conv2D(128,
                               kernel_size=1,
                               padding='same',
                               use_bias=False,
                               name='block_24_conv'),
        tf.keras.layers.BatchNormalization(
            momentum=0.9,
            name='block_24_BN'),
        tf.keras.layers.ReLU(6., name='block_24_relu6'),
        tf.keras.layers.UpSampling2D(2))(x)
    block_args = block_args._replace(input_filters=128)
    x = tf.keras.layers.Concatenate()(
        [x, backbone.get_layer('block4a_expand_activation').output])
    x, y3 = make_last_layers_efficientnet(x, block_args, "y3_")
    return Model(inputs, [y1, y2, y3])


def yolo_efficientnetliteb1_body(inputs, number_anchors, number_classes):
    block_args = BlockArgs(kernel_size=3,
                           num_repeat=1,
                           input_filters=512,
                           output_filters=number_anchors * (number_classes + 5),
                           expand_ratio=1,
                           id_skip=True,
                           se_ratio=None,
                           strides=[1, 1])
    backbone = EfficientNetLiteB1(include_top=False, input_tensor=inputs, weights=None)
    x, y1 = make_last_layers_efficientnet(backbone.output, block_args, "y1_")
    x = compose(
        tf.keras.layers.Conv2D(256,
                               kernel_size=1,
                               padding='same',
                               use_bias=False,
                               name='block_20_conv'),
        tf.keras.layers.BatchNormalization(momentum=0.9,
                                           name='block_20_BN'),
        tf.keras.layers.ReLU(6., name='block_20_relu6'),
        tf.keras.layers.UpSampling2D(2))(x)
    block_args = block_args._replace(input_filters=256)
    x = tf.keras.layers.Concatenate()(
        [x, backbone.get_layer('block6a_expand_activation').output])
    x, y2 = make_last_layers_efficientnet(x, block_args, "y2_")
    x = compose(
        tf.keras.layers.Conv2D(128,
                               kernel_size=1,
                               padding='same',
                               use_bias=False,
                               name='block_24_conv'),
        tf.keras.layers.BatchNormalization(
            momentum=0.9,
            name='block_24_BN'),
        tf.keras.layers.ReLU(6., name='block_24_relu6'),
        tf.keras.layers.UpSampling2D(2))(x)
    block_args = block_args._replace(input_filters=128)
    x = tf.keras.layers.Concatenate()(
        [x, backbone.get_layer('block4a_expand_activation').output])
    x, y3 = make_last_layers_efficientnet(x, block_args, "y3_")
    return Model(inputs, [y1, y2, y3])


def yolo_efficientnetb3_body(inputs, number_anchors, number_classes):
    block_args = BlockArgs(kernel_size=3,
                           num_repeat=1,
                           input_filters=512,
                           output_filters=number_anchors * (number_classes + 5),
                           expand_ratio=1,
                           id_skip=True,
                           se_ratio=0.25,
                           strides=[1, 1])
    backbone = EfficientNetB3(include_top=False, input_tensor=inputs, weights=None)
    x, y1 = make_last_layers_efficientnet(backbone.output, block_args, "y1_")
    x = compose(
        tf.keras.layers.Conv2D(256,
                               kernel_size=1,
                               padding='same',
                               use_bias=False,
                               name='block_20_conv'),
        tf.keras.layers.BatchNormalization(momentum=0.9,
                                           name='block_20_BN'),
        tf.keras.layers.ReLU(6., name='block_20_relu6'),
        tf.keras.layers.UpSampling2D(2))(x)
    block_args = block_args._replace(input_filters=256)
    x = tf.keras.layers.Concatenate()(
        [x, backbone.get_layer('block6a_expand_activation').output])
    x, y2 = make_last_layers_efficientnet(x, block_args, "y2_")
    x = compose(
        tf.keras.layers.Conv2D(128,
                               kernel_size=1,
                               padding='same',
                               use_bias=False,
                               name='block_24_conv'),
        tf.keras.layers.BatchNormalization(
            momentum=0.9,
            name='block_24_BN'),
        tf.keras.layers.ReLU(6., name='block_24_relu6'),
        tf.keras.layers.UpSampling2D(2))(x)
    block_args = block_args._replace(input_filters=128)
    x = tf.keras.layers.Concatenate()(
        [x, backbone.get_layer('block4a_expand_activation').output])
    x, y3 = make_last_layers_efficientnet(x, block_args, "y3_")
    return Model(inputs, [y1, y2, y3])


def yolo_efficientnetb0_body(inputs, number_anchors, number_classes):
    block_args = BlockArgs(kernel_size=3,
                           num_repeat=1,
                           input_filters=512,
                           output_filters=number_anchors * (number_classes + 5),
                           expand_ratio=1,
                           id_skip=True,
                           se_ratio=0.25,
                           strides=[1, 1])
    backbone = EfficientNetB0(include_top=False, input_tensor=inputs, weights=None)
    x, y1 = make_last_layers_efficientnet(backbone.output, block_args, "y1_")
    x = compose(
        tf.keras.layers.Conv2D(256,
                               kernel_size=1,
                               padding='same',
                               use_bias=False,
                               name='block_20_conv'),
        tf.keras.layers.BatchNormalization(momentum=0.9,
                                           name='block_20_BN'),
        tf.keras.layers.ReLU(6., name='block_20_relu6'),
        tf.keras.layers.UpSampling2D(2))(x)
    block_args = block_args._replace(input_filters=256)
    x = tf.keras.layers.Concatenate()(
        [x, backbone.get_layer('block6a_expand_activation').output])
    x, y2 = make_last_layers_efficientnet(x, block_args, "y2_")
    x = compose(
        tf.keras.layers.Conv2D(128,
                               kernel_size=1,
                               padding='same',
                               use_bias=False,
                               name='block_24_conv'),
        tf.keras.layers.BatchNormalization(
            momentum=0.9,
            name='block_24_BN'),
        tf.keras.layers.ReLU(6., name='block_24_relu6'),
        tf.keras.layers.UpSampling2D(2))(x)
    block_args = block_args._replace(input_filters=128)
    x = tf.keras.layers.Concatenate()(
        [x, backbone.get_layer('block4a_expand_activation').output])
    x, y3 = make_last_layers_efficientnet(x, block_args, "y3_")
    return Model(inputs, [y1, y2, y3])


def tiny_yolo_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 model CNN body in keras.'''
    x1 = compose(
        DarknetConv2D_BN_Leaky(16, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(32, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(64, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(128, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(256, (3, 3)))(inputs)
    x2 = compose(
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(512, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'),
        DarknetConv2D_BN_Leaky(1024, (3, 3)),
        DarknetConv2D_BN_Leaky(256, (1, 1)))(x1)
    y1 = compose(
        DarknetConv2D_BN_Leaky(512, (3, 3)),
        DarknetConv2D(num_anchors * (num_classes + 5), (1, 1)))(x2)

    x2 = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        UpSampling2D(2))(x2)
    y2 = compose(
        Concatenate(),
        DarknetConv2D_BN_Leaky(256, (3, 3)),
        DarknetConv2D(num_anchors * (num_classes + 5), (1, 1)))([x2, x1])

    return Model(inputs, [y1, y2])


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """计算grid和预测box的坐标和长宽
        Args:
            feats, 即yolobody的输出，未经过未经过sigmoid函数处理
            anchors: anchor box
            num_classes: number class
            input_shape: input shape of yolobody, like 416,320
            calc_loss: where to caculate loss, used for training
        Returns:
            box_xy  相对整图的大小，0至1
            box_wh   相对input shape即416的大小，0，+无穷大
            box_confidence
            box_class_probs
    """
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3]  # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                    [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])  # shape like 26，26，1，2
    grid = K.cast(grid, K.dtype(feats))
    # shape like batch,26,26,3,85
    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    # batch,26,26,3,2 相对整图的位置，即0，1范围
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    # 相对于416的大小
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''
    获取正确的box坐标，相对整图的坐标
    Args:
        box_xy  xy坐标值
        box_wh  宽高
        input_shape  模型输入尺寸
        image_shape  图片原始尺寸，用于还原重建坐标，Height * Width

    '''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape  # 相对整图的偏移量
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    # 注此处y和x对换
    boxes = K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    # 获取输出，即相对整图的长宽以及置信度与概率值
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
                                                                anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes.
    只适用于一张图片的处理，不适合批处理
    Args:
        image_shape: 原始输入图片尺寸, 高X宽
    Returns:
        boxes,其中box格式为[y1,x1,y2,x2]
    """
    num_layers = len(yolo_outputs)
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]  # default setting
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    for l in range(num_layers):
        # 此处的处理会去除batch的信息，完全展开，因此该计算图只能用于单张图片处理，不能批处理
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
                                                    anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    # 此处更改是为了直接把后处理写入模型中
    # boxes_ = K.expand_dims(K.concatenate(boxes_, axis=0), axis=0)
    # scores_ = K.expand_dims(K.concatenate(scores_, axis=0), axis=0)
    # classes_ = K.expand_dims(K.concatenate(classes_, axis=0), axis=0)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_


def yolo_eval_serving(yolo_outputs,
                      anchors,
                      num_classes,
                      image_shape,
                      max_boxes=20,
                      score_threshold=.6,
                      iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes.
    只适用于一张图片的处理，不适合批处理
    Args:
        image_shape: 原始输入图片尺寸, 高X宽
    Returns:
        boxes,其中box格式为[y1,x1,y2,x2]
    """
    num_layers = len(yolo_outputs)
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]  # default setting
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    for l in range(num_layers):
        # 此处的处理会去除batch的信息，完全展开，因此该计算图只能用于单张图片处理，不能批处理
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
                                                    anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    # 此处更改是为了直接把后处理写入模型中
    boxes_ = K.expand_dims(K.concatenate(boxes_, axis=0), axis=0)
    scores_ = K.expand_dims(K.concatenate(scores_, axis=0), axis=0)
    classes_ = K.expand_dims(K.concatenate(classes_, axis=0), axis=0)
    # boxes_ = K.concatenate(boxes_, axis=0)
    # scores_ = K.concatenate(scores_, axis=0)
    # classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_


def box_iou(b1, b2):
    '''Return iou tensor, 即所有预测box与真实box的iou值
    Parameters
    ----------
    b1: predict box tensor, shape=(i1,...,iN, 4), xywh, shape like 26*26*3*4
    b2: true box tensor, tensor, shape=(j, 4), xywh, j mean the real box number for image
    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)
    '''

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def yolo_loss(data, anchors, num_classes, ignore_thresh=.5, print_loss=True):
    '''Return yolo_loss tensor
    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), 2 refer to wh, N refer to number of achors
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss
    Returns
    -------
    loss: tensor, shape=(1,)
    '''
    num_layers = len(anchors) // 3  # default setting
    yolo_outputs = data[:num_layers]
    y_true = data[num_layers:]  # list batch * 26* 26 * 85
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    # 13， 26， 52
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    loss = 0
    m = K.shape(yolo_outputs[0])[0]  # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    for l in range(num_layers):
        # object score for real image, guess 0 or 1
        object_mask = y_true[l][..., 4:5]
        # true class probs； 0 or 1
        true_class_probs = y_true[l][..., 5:]
        # grid values like 0,1,...25 or 0,...51 or 0,...13
        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
                                                     anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])

        # Darknet raw box to calculate loss.相对grid的坐标
        raw_true_xy = y_true[l][..., :2] * grid_shapes[l][::-1] - grid
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))  # avoid log(0)=-inf
        # box_loss_scale used for scale imbalance large value for small object and small value for large object
        box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]
        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')

        def loop_body(b, ignore_mask):
            # 每张图片的真实box,应为二维tensor
            true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
            # pred_box batch,26,26,3,4，iou shape like  (26,26,3,real_box_num)
            iou = box_iou(pred_box[b], true_box)
            # 26,26,3，所有grid的预测值与真实box最好的iou值
            best_iou = K.max(iou, axis=-1)
            # 小于阙值的为真，即确定当前值是否参与计算
            ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_thresh, K.dtype(true_box)))
            return b + 1, ignore_mask

        _, ignore_mask = tf.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)
        """
        损失函数组成：
            1、中心定位误差，采用交叉熵，只计算有真实目标位置的损失
            2、宽度高度误差，使用L2损失，只计算有真实目标位置的损失
            3、是否包含目标的置信度计算，包含两部分，一个是真实目标位置的binary crossentropy, 
                一个是其他位置的binary crossentropy，不包含与真实目标iou大于0.5的位置
            4、各个类别的损失函数，只计算有真实目标位置的损失
        """
        # K.binary_crossentropy is helpful to avoid exp overflow.     相等时为极值点
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2],
                                                                       from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[..., 2:4])
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) + \
                          (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5],
                                                                    from_logits=True) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True)

        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        loss += xy_loss + wh_loss + confidence_loss + class_loss
        if print_loss:
            loss = tf.compat.v1.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)],
                                      message='loss: ')
    # tensor must have 1 dimension for tensorflow 2.0 above
    loss = tf.expand_dims(loss, 0)
    return loss

# if __name__ == '__main__':
