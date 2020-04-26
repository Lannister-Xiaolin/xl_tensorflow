#!usr/bin/env python3
# -*- coding: UTF-8 -*-
import os

from .training import body_dict
from .utils import get_anchors, letterbox_image
from tensorflow.keras import Input, Model
from .model import yolo_eval, DEFALT_ANCHORS, yolo_eval_lite
from PIL import Image
import numpy as np
from xl_tensorflow.metrics.rafaelpadilla import voc2ratxt, map_raf_from_lists
from xl_tool.data.image.annonation import get_bndbox


def single_inference_model_serving(model_name, weights,
                                   num_classes,
                                   image_shape=(416, 416),
                                   input_shape=(416, 416),
                                   anchors=None,
                                   score_threshold=.1,
                                   iou_threshold=.5,
                                   max_detections=20,
                                   dynamic_shape=False):
    """
    用于部署在serving端的模型，固定输入尺寸和图片尺寸，会对iou值和置信度进行过滤0.1
    暂时不将尺寸和阙值写入模型，因此返回框的尺寸和位置需要根据图片进行重新调整（与resize方式有关）
    Args:
        image_shape: 宽高
    Returns:
        tf.keras.Model object, 预测图片的绝对值坐标x1,y1,x2,y2
    """
    # Todo 把iou和置信度,以及输入图片尺寸（高宽）， 写入模型
    if anchors == None:
        anchors = DEFALT_ANCHORS
    yolo_model = body_dict[model_name](Input(shape=(*input_shape, 3)),
                                       len(anchors) // 3, num_classes)
    if weights:
        yolo_model.load_weights(weights)
    if dynamic_shape:
        shape_input = Input(shape=(2,))
        boxes_, scores_, classes_ = yolo_eval(yolo_model.outputs,
                                              anchors, num_classes, shape_input, max_detections,
                                              score_threshold,
                                              iou_threshold, return_xy=True)
        model = Model(inputs=yolo_model.inputs + [shape_input], outputs=(boxes_, scores_, classes_))
    else:
        boxes_, scores_, classes_ = yolo_eval(yolo_model.outputs,
                                              anchors, num_classes, image_shape, max_detections,
                                              score_threshold,
                                              iou_threshold, return_xy=True)
        model = Model(inputs=yolo_model.inputs, outputs=(boxes_, scores_, classes_))
    return model


def single_inference_model_lite(model_name, weights, num_classes, image_shape=(480, 640),
                                anchors=None,
                                input_shape=(416, 416),
                                score_threshold=.6,
                                iou_threshold=.5):
    """
    专门用于移动端处理，无5维tensor和掩码
    """
    # Todo 把iou和置信度,以及输入图片尺寸（高宽）， 写入模型
    if anchors == None:
        anchors = DEFALT_ANCHORS
    yolo_model = body_dict[model_name](Input(shape=(*input_shape, 3)), len(anchors) // 3, num_classes)
    yolo_model.load_weights(weights)

    boxes_, scores_ = yolo_eval_lite(yolo_model.outputs, anchors, num_classes, image_shape, 20,
                                     score_threshold,
                                     iou_threshold)
    model = Model(inputs=yolo_model.inputs, outputs=(boxes_, scores_))
    return model


def map_evaluate(image_files, gt_xml_files, model_name, weights,
                 gt_path,
                 dt_path,
                 num_classes,
                 index2label,
                 anchors=None,
                 input_shape=(416, 416),
                 score_threshold=.1,
                 iou_threshold=.5):
    model = single_inference_model_serving(model_name=model_name, weights=weights, num_classes=num_classes,
                                           dynamic_shape=True)

    for xml_file in gt_xml_files:
        bndboxes = voc2ratxt(xml_file, box_format="xyxy")
        with open(f"{gt_path}/{os.path.basename(xml_file).split('.')[0]}.txt") as f:
            f.write("\n".join([" ".join([str(j) for j in i]) for i in bndboxes]))
    for image_file in image_files:
        image = Image.open(image_file)
        image_id = os.path.basename(image_file).split('.')[0]
        boxed_image = letterbox_image(image, (416, 416))
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)
        boxes_, scores_, classes_ = model.predict([image_data, np.array([[*image.size][::-1]])])
        boxes_, scores_, classes_ = boxes_[0], scores_[0], classes_[0]
        if len(scores_) > 0:
            dt_boxes = []
            for i in range(scores_):
                dt_boxes.append(f"{index2label(classes_[i])} {scores_[i]:.2f} {' '.join([str(i) for i in boxes_[i]])}")
            with open(f"{dt_path}/{image_id}.txt") as f:
                f.write("\n".join(dt_boxes))
        else:
            pass

            # Todo 导出符合规格的box
            pass
