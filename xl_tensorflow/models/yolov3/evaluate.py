#!usr/bin/env python3
# -*- coding: UTF-8 -*-
from .training import body_dict
from .utils import get_anchors
from tensorflow.keras import Input, Model
from .model import yolo_eval


def single_inference_model_serving(model_name, weights,
                                   num_classes,
                                   image_shape=(416, 416),
                                   anchors=None,
                                   input_shape=(416, 416),
                                   score_threshold=.1,
                                   iou_threshold=.5):
    """
    用于部署在serving端的模型，固定输入尺寸和图片尺寸，会对iou值和置信度进行过滤0.1
    暂时不将尺寸和阙值写入模型，因此返回框的尺寸和位置需要根据图片进行重新调整（与resize方式有关）
    Args:
        image_shape: 宽高
    """
    # Todo 把iou和置信度,以及输入图片尺寸（高宽）， 写入模型
    if anchors == None:
        anchors = get_anchors("./config/yolo_anchors.txt")
    yolo_model = body_dict[model_name](Input(shape=(*input_shape, 3)),
                                       len(anchors) // 3, num_classes)
    yolo_model.load_weights(weights)
    boxes_, scores_, classes_ = yolo_eval(yolo_model.outputs,
                                          anchors, num_classes, image_shape, 20,
                                          score_threshold,
                                          iou_threshold)
    model = Model(inputs=yolo_model.inputs, outputs=(boxes_, scores_, classes_))
    return model


def map_evaluate(image_files, gt_xml_files, model_name, weights,
                 num_classes,
                 image_shape=(416, 416),
                 anchors=None,
                 input_shape=(416, 416),
                 score_threshold=.1,
                 iou_threshold=.5):
    model = single_inference_model_serving(model_name=model_name, weights=weights)
