#!usr/bin/env python3
# -*- coding: UTF-8 -*-
from ..configs.anchors import YOLOV3_ANCHORS

# !usr/bin/env python3
# -*- coding: UTF-8 -*-
from functools import wraps
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Model, layers
from ..utils.yolo_utils import compose
from tensorflow.keras.applications import MobileNetV2
from xl_tensorflow.models.vision.classification.darknet import DarknetConv2D_BN_Leaky, \
    DarknetConv2D, darknet_body, cspdarknet_body
from xl_tensorflow.models.vision.classification.efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2, \
    EfficientNetLiteB1, \
    EfficientNetLiteB2, EfficientNetLiteB3, EfficientNetLiteB4
from .common import node_aggregate
from .aggregation import pan_network, fpn_network
from ..configs.yolo_config import get_yolo_config


def spatial_pyramid_block(feature):
    """
    SSP layer for yolo
    Args:
        feature:

    Returns:

    """
    pre_convs = [
        DarknetConv2D_BN_Leaky(filters=512, kernel_size=1, strides=1),
        DarknetConv2D_BN_Leaky(filters=1024, kernel_size=3, strides=1),
        DarknetConv2D_BN_Leaky(filters=512, kernel_size=1, strides=1),
    ]
    feature_1 = compose(*pre_convs)(feature)
    feature_5 = layers.MaxPooling2D(pool_size=5, padding="same", name="spp_5", strides=1)(feature_1)
    feature_9 = layers.MaxPooling2D(pool_size=9, padding="same", name="spp_9", strides=1)(feature_1)
    feature_13 = layers.MaxPooling2D(pool_size=13, padding="same", name="spp_13", strides=1)(feature_1)
    new_feature = node_aggregate([feature_13, feature_9, feature_5, feature_1], method="concat")
    new_feature = DarknetConv2D_BN_Leaky(filters=512, kernel_size=1, strides=1)(new_feature)
    new_feature = DarknetConv2D_BN_Leaky(filters=1024, kernel_size=3, strides=1)(new_feature)
    return new_feature


def output_wrapper(func):
    """将backbone的输出形状(batch,anchor,anchor,3*(5+num_classes))
     reshape为(batch,anchor,anchor,3，(5+num_classes)),主要用于自定义的yololoss"""

    @wraps(func)
    def wrapper(inputs, num_anchors, num_classes, architecture, reshape_y=False):
        model = func(inputs, num_anchors, num_classes, architecture)
        if reshape_y:
            y1, y2, y3 = model.outputs
            # print(y1, y2, y3)
            y1 = tf.keras.layers.Reshape((y1.shape[1], y1.shape[2], num_anchors, num_classes + 5))(y1)
            y2 = tf.keras.layers.Reshape((y2.shape[1], y2.shape[2], num_anchors, num_classes + 5))(y2)
            y3 = tf.keras.layers.Reshape((y3.shape[1], y3.shape[2], num_anchors, num_classes + 5))(y3)
            # print("   ", y1, y2, y3)
            model = Model(inputs, [y1, y2, y3])
        return model

    return wrapper


@output_wrapper
def yolo_body(inputs, num_anchors, num_classes, architecture="yolov4"):
    """Create YOLO_V3 model CNN body in Keras."""
    if architecture == "yolov4":
        config = get_yolo_config("yolov4", num_anchors, num_classes)
        outputs = spatial_pyramid_block(cspdarknet_body(inputs)) if config.spp else cspdarknet_body(inputs)
        body = Model(inputs, outputs)
        features = [body.layers[131].output, body.layers[204].output, body.output]  # mish_37  58
    elif architecture == "yolov4_efficientnetb0":
        config = get_yolo_config("yolov4_efficientnetb0", num_anchors, num_classes)
        backbone = EfficientNetB0(include_top=False, weights=None, input_tensor=inputs)

        outputs = spatial_pyramid_block(
            backbone.get_layer("top_activation").output) if config.spp else backbone.get_layer("top_activation").output
        body = Model(inputs, outputs)
        features = [body.get_layer("block4a_expand_activation").output,
                    body.get_layer("block6a_expand_activation").output,
                    body.output]
    elif architecture == "yolov4_efficientnetb1":
        config = get_yolo_config("yolov4_efficientnetb0", num_anchors, num_classes)
        backbone = EfficientNetB1(include_top=False, weights=None, input_tensor=inputs)
        outputs = spatial_pyramid_block(
            backbone.get_layer("top_activation").output) if config.spp else backbone.get_layer("top_activation").output
        body = Model(inputs, outputs)
        features = [body.get_layer("block4a_expand_activation").output,
                    body.get_layer("block6a_expand_activation").output,
                    body.output]
    elif architecture == "yolov4_efficientnetb2":
        config = get_yolo_config("yolov4_efficientnetb0", num_anchors, num_classes)
        backbone = EfficientNetB2(include_top=False, weights=None, input_tensor=inputs)
        outputs = spatial_pyramid_block(
            backbone.get_layer("top_activation").output) if config.spp else backbone.get_layer("top_activation").output
        body = Model(inputs, outputs)
        features = [body.get_layer("block4a_expand_activation").output,
                    body.get_layer("block6a_expand_activation").output,
                    body.output]
    elif architecture == "yolov4_efficientnetliteb1":
        config = get_yolo_config("yolov4_efficientnetb0", num_anchors, num_classes)
        backbone = EfficientNetLiteB1(include_top=False, weights=None, input_tensor=inputs)
        outputs = spatial_pyramid_block(
            backbone.get_layer("top_activation").output) if config.spp else backbone.get_layer("top_activation").output
        body = Model(inputs, outputs)
        features = [body.get_layer("block4a_expand_activation").output,
                    body.get_layer("block6a_expand_activation").output,
                    body.output]
    elif architecture == "yolov4_efficientnetliteb2":
        config = get_yolo_config("yolov4_efficientnetb0", num_anchors, num_classes)
        backbone = EfficientNetLiteB2(include_top=False, weights=None, input_tensor=inputs)
        outputs = spatial_pyramid_block(
            backbone.get_layer("top_activation").output) if config.spp else backbone.get_layer("top_activation").output
        body = Model(inputs, outputs)
        features = [body.get_layer("block4a_expand_activation").output,
                    body.get_layer("block6a_expand_activation").output,
                    body.output]
    elif architecture == "yolov4_efficientnetliteb3":
        config = get_yolo_config("yolov4_efficientnetb0", num_anchors, num_classes)
        backbone = EfficientNetLiteB3(include_top=False, weights=None, input_tensor=inputs)
        outputs = spatial_pyramid_block(
            backbone.get_layer("top_activation").output) if config.spp else backbone.get_layer("top_activation").output
        body = Model(inputs, outputs)
        features = [body.get_layer("block4a_expand_activation").output,
                    body.get_layer("block6a_expand_activation").output,
                    body.output]
    elif architecture == "yolov4_efficientnetliteb4":
        config = get_yolo_config("yolov4_efficientnetb0", num_anchors, num_classes)
        backbone = EfficientNetLiteB4(include_top=False, weights=None, input_tensor=inputs)
        outputs = spatial_pyramid_block(
            backbone.get_layer("top_activation").output) if config.spp else backbone.get_layer("top_activation").output
        body = Model(inputs, outputs)
        features = [body.get_layer("block4a_expand_activation").output,
                    body.get_layer("block6a_expand_activation").output,
                    body.output]
    elif architecture == "yolov4_mobilenetv2":
        config = get_yolo_config("yolov4_efficientnetb0", num_anchors, num_classes)
        backbone = MobileNetV2(include_top=False, weights=None, input_tensor=inputs)
        outputs = spatial_pyramid_block(
            backbone.get_layer("out_relu").output) if config.spp else backbone.get_layer("out_relu").output
        body = Model(inputs, outputs)
        features = [body.get_layer("block_6_expand_relu").output,
                    body.get_layer("block_13_expand_relu").output,
                    body.output]
    elif architecture == "yolov3_efficientnetliteb4_spp":
        config = get_yolo_config("yolov3", num_anchors, num_classes)
        config.agg_method = "fpn"
        backbone = EfficientNetLiteB4(include_top=False, weights=None, input_tensor=inputs)
        outputs = spatial_pyramid_block(backbone.get_layer("top_activation").output)
        body = Model(inputs, outputs)
        features = [body.get_layer("block4a_expand_activation").output,
                    body.get_layer("block6a_expand_activation").output,
                    body.output]
    elif architecture == "yolov3_efficientnetliteb0_spp":
        config = get_yolo_config("yolov3", num_anchors, num_classes)
        config.agg_method = "fpn"
        backbone = EfficientNetB0(include_top=False, weights=None, input_tensor=inputs)
        outputs = spatial_pyramid_block(backbone.get_layer("top_activation").output)
        body = Model(inputs, outputs)
        features = [body.get_layer("block4a_expand_activation").output,
                    body.get_layer("block6a_expand_activation").output,
                    body.output]
    elif architecture == "yolov3_mobilenetv2_spp":
        config = get_yolo_config("yolov3", num_anchors, num_classes)
        config.agg_method = "fpn"
        backbone = MobileNetV2(include_top=False, weights=None, input_tensor=inputs)
        outputs = spatial_pyramid_block(
            backbone.get_layer("out_relu").output) if config.spp else backbone.get_layer("out_relu").output
        body = Model(inputs, outputs)
        features = [body.get_layer("block_6_expand_relu").output,
                    body.get_layer("block_13_expand_relu").output,
                    body.output]
    else:
        config = get_yolo_config("yolov3", num_anchors, num_classes)
        # print(config)
        outputs = spatial_pyramid_block(darknet_body(inputs)) if config.spp else darknet_body(inputs)
        body = Model(inputs, outputs)
        features = [body.layers[92].output, body.layers[152].output, body.output]
        pass
    # print(config.agg_method)
    if config.agg_method == "panet":
        new_features = pan_network(features, config)
        y1, y2, y3 = new_features[::-1]
    else:
        new_features = fpn_network(features, config)
        y1, y2, y3 = new_features[::-1]

    return Model(inputs, [y1, y2, y3])


def box_iou(b1, b2, method="iou", as_loss=False):
    '''Return iou tensor, 即所有预测box与真实box的iou值
    Parameters
    ----------
    b1: predict box tensor, shape=(i1,...,iN, 4), xywh, shape like 26*26*3*4
    b2: true box tensor, tensor, shape=(j, 4), xywh, j mean the real box number for image
    method: must be one of "iou giou ciou diou"
    as_loss: whether to use iou as loss
    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)
    '''

    # Expand dim to apply broadcasting.
    if not as_loss:
        b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = tf.maximum(0.0, b1[..., 2:4])
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    if not as_loss:
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
    union_area = b1_area + b2_area - intersect_area
    iou = tf.math.divide_no_nan(intersect_area, union_area)
    if method == "iou":
        return iou
    elif method in ("giou", "ciou", "diou"):
        enclose_ymin = tf.minimum(b1_mins[..., 1], b2_mins[..., 1])
        enclose_xmin = tf.minimum(b1_mins[..., 0], b2_mins[..., 0])
        enclose_ymax = tf.maximum(b1_maxes[..., 1], b2_maxes[..., 1])
        enclose_xmax = tf.maximum(b1_maxes[..., 0], b2_maxes[..., 0])
        enclose_width = tf.maximum(0.0, enclose_xmax - enclose_xmin)
        enclose_height = tf.maximum(0.0, enclose_ymax - enclose_ymin)
        enclose_area = enclose_width * enclose_height
        if method == "giou":
            giou = iou - tf.math.divide_no_nan(
                (enclose_area - union_area), enclose_area)
            return giou
        elif method == "diou":
            diou_term = tf.math.divide_no_nan(tf.reduce_sum(tf.math.pow((b1_xy - b2_xy), 2), axis=-1),
                                              tf.math.pow((enclose_width * enclose_height), 2))
            diou = iou - diou_term
            return diou
            pass
        elif method == "ciou":
            d = tf.math.divide_no_nan(tf.reduce_sum(tf.math.pow((b1_xy - b2_xy), 2), axis=-1),
                                      tf.math.pow((enclose_width * enclose_height), 2))
            ar_gt = tf.atan(tf.math.divide_no_nan(b2_wh[..., 0], b2_wh[..., 1]))
            ar_pred = tf.atan(tf.math.divide_no_nan(b1_wh[..., 0], b1_wh[..., 1]))
            ar_loss = 0.4052847 * tf.pow(ar_gt - ar_pred, 2)
            alpha = ar_loss / (1 - iou + ar_loss + 0.000001)
            ciou_term = d + alpha * ar_loss
            ciou = iou - ciou_term
            return ciou
        else:
            return iou
    return iou


def yolo_head(feats, anchors, input_shape, calc_loss=False):
    """计算grid和预测box的坐标和长宽(专门用于指定的tf loss类)
        Args:
            feats, 即yolobody的输出，未经过未经过sigmoid函数处理,输出为batch 26,26,3,85
            anchors: anchor box
            num_classes: number class
            input_shape: input shape of yolobody, like 416,320
            calc_loss: where to caculate loss, used for training
        Returns:
            box_xy  相对整图的大小，0至1,shape like (batch, gridx,gridy,3,2)
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
    # # shape like batch,26,26,3,85
    # Adjust preditions to each spatial grid point and anchor size.
    # coordinates normalized to 0,1,relative to grid , batch,26,26,3,2
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    # box_xy = ((K.sigmoid(feats[..., :2]) * scale - (scale-1)/2) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    # size relative to input shape(ie:416)
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])
    if calc_loss == True:
        # grid, shape like 26，26，1，2
        # feats, shape like batch,26,26,3,85
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


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, origin_image_shape):
    '''Process Conv layer output'''
    # 获取输出，即相对整图的长宽以及置信度与概率值
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats=feats,
                                                                anchors=anchors, input_shape=input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, origin_image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              origin_image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5, return_xy=True):
    """Evaluate YOLO model on given input and return filtered boxes.
    只适用于一张图片的处理，不适合批处理
    Args:
        origin_image_shape: 原始输入图片尺寸, 高X宽
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
                                                    anchors[anchor_mask[l]], num_classes, input_shape,
                                                    origin_image_shape)
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
    if return_xy:
        boxes_ = K.concatenate([
            boxes_[..., 1:2],  # y_min
            boxes_[..., 0:1],  # x_min
            boxes_[..., 3:],  # y_max
            boxes_[..., 2:3]  # x_max
        ])
    # boxes_ = K.concatenate(boxes_, axis=0)
    # scores_ = K.concatenate(scores_, axis=0)
    # classes_ = K.concatenate(classes_, axis=0)
    return boxes_, scores_, classes_


# Todo lite 的配置暂时不管
def yolo_head_lite(feats, anchors, num_classes, input_shape, calc_loss=False):
    """计算grid和预测box的坐标和长宽
        Args:
            feats, 即yolobody的输出，未经过未经过sigmoid函数处理,输出为batch 26,26,255
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
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3]  # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                    [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])  # shape like 26，26，1，2
    grid = K.cast(grid, K.dtype(feats))
    # shape like batch,26,26,3,85
    feats = K.reshape(
        feats, [grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    # batch,26,26,3,2 相对整图的位置，即0，1范围
    box_xy = (K.sigmoid(feats[:, :, :, :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    # 相对于416的大小
    box_wh = K.exp(feats[:, :, :, 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[:, :, :, 4:5])
    box_class_probs = K.sigmoid(feats[:, :, :, 5:])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes_lite(box_xy, box_wh, input_shape, image_shape):
    '''
    获取正确的box坐标，相对整图的坐标
    Args:
        box_xy  xy坐标值
        box_wh  宽高
        input_shape  模型输入尺寸
        image_shape  图片原始尺寸，用于还原重建坐标，Height * Width

    '''
    box_yx = box_xy[:, :, :, ::-1]
    box_hw = box_wh[:, :, :, ::-1]
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
        box_mins[:, :, :, 0:1],  # y_min
        box_mins[:, :, :, 1:2],  # x_min
        box_maxes[:, :, :, 0:1],  # y_max
        box_maxes[:, :, :, 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def yolo_boxes_and_scores_lite(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    # 获取输出，即相对整图的长宽以及置信度与概率值
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head_lite(feats,
                                                                     anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes_lite(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def yolo_eval_lite(yolo_outputs,
                   anchors,
                   num_classes,
                   image_shape,
                   return_xy=True):
    """Todo 未测试和完善
    """
    num_layers = len(yolo_outputs)
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]  # default setting
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    for l in range(num_layers):
        # 此处的处理会去除batch的信息，完全展开，因此该计算图只能用于单张图片处理，不能批处理
        _boxes, _box_scores = yolo_boxes_and_scores_lite(yolo_outputs[l],
                                                         anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)
    if return_xy:
        boxes = K.concatenate([
            boxes[..., 1:2],  # y_min
            boxes[..., 0:1],  # x_min
            boxes[..., 3:],  # y_max
            boxes[..., 2:3]  # x_max
        ])
    box_scores = K.concatenate(box_scores, axis=0)

    # Todo 此处需要注意是为了方便java
    box_scores = tf.transpose(box_scores)
    return K.expand_dims(boxes, 0), K.expand_dims(box_scores, 0)
