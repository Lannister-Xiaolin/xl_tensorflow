#!usr/bin/env python3
# -*- coding: UTF-8 -*-

# 后处理步骤
# 提供input_anchor = anchor.Anchor(
#             self._min_level, self._max_level, self._num_scales,
#             self._aspect_ratios, self._anchor_size, (image_height, image_width))
# 参考inference.py 233
# 使用faster_rcnn_box_coder.decode即可将坐标还原
import functools

import xl_tensorflow.models.vision.detection.configs.factory as config_factory
from xl_tensorflow.models.vision.detection.body.efficientdet_model import EfficientDetModel
from xl_tensorflow.models.vision.detection.dataloader.utils import input_utils, box_list, faster_rcnn_box_coder
from typing import Text, Dict, Any, List, Tuple, Union
import tensorflow as tf
from xl_tensorflow.layers.conv import Base64ImageProcessLayer, ResizeImageProcessLayer
from xl_tensorflow.utils.deploy import serving_model_export
import os
import pathlib
import numpy as np
from xl_tensorflow.utils.deploy import quantize_model


# todo 推理部署 - 保证所有检测接口保持一致，高可用，高性能（参考谷歌官方，端到端，高效，快速）

def image_preprocess(image, image_size: Union[int, Tuple[int, int]]):
    """Preprocess image for inference.

    Args:
      image: input image, can be a tensor or a numpy arary.
      image_size: single integer of image size for square image or tuple of two
        integers, in the format of (image_height, image_width).

    Returns:
      (image, scale): a tuple of processed image and its scale.
    """
    image = input_utils.normalize_image(image)
    image, image_info = input_utils.resize_and_crop_image(
        image,
        image_size,
        padded_size=input_utils.compute_padded_size(
            image_size, 2 ** 1),
        aug_scale_min=1.0,
        aug_scale_max=1.0)
    image_scale = image_info[2, :]
    return image, image_scale


def batch_image_preprocess(raw_images,
                           image_size: Union[int, Tuple[int, int]],
                           batch_size: int = None):
    """Preprocess batched images for inference.

  Args:
    raw_images: a list of images, each image can be a tensor or a numpy arary.
    image_size: single integer of image size for square image or tuple of two
      integers, in the format of (image_height, image_width).
    batch_size: if None, use map_fn to deal with dynamic batch size.

  Returns:
    (image, scale): a tuple of processed images and scales.
  """
    # hint； images must in the same shape if batch_size is none
    if not batch_size:
        # map_fn is a little bit slower due to some extra overhead.
        map_fn = functools.partial(image_preprocess, image_size=image_size)
        images, scales = tf.map_fn(
            map_fn, raw_images, dtype=(tf.float32, tf.float32), back_prop=False)
        return images, scales
    # If batch size is known, use a simple loop.
    scales, images = [], []
    for i in range(batch_size):
        image, scale = image_preprocess(raw_images[i], image_size)
        scales.append(scale)
        images.append(image)
    images = tf.stack(images)
    scales = tf.stack(scales)
    return images, scales


def efficiendet_inference_model(model_name="efficientdet-d0",
                                input_shape=(512, 512),
                                inference_mode="fixed",
                                num_classes=85,
                                weights=None,
                                mean=tf.constant([0.485, 0.456, 0.406]),
                                std=tf.constant([0.229, 0.224, 0.225]),
                                serving_export=False,
                                version=1,
                                max_detections=20,
                                auto_incre_version=True,
                                serving_path=None, lite_path=None, quantize_method=None, allow_custom_ops=False):
    """
    Hint: 自定义base64模型不能保存为keras格式模型，请使用tf格式
    Args:
        model_name:
        input_shape:
        inference_mode:
            base64:  end to end, no need to preprocess and post process
            dynamic: fixed size inputs and shape input:  just read image with array, no need to add extral preprocess and post process
            fixed: fixed size inputs as target size：
        preprocessing:
        shape_input:

    Returns:

    """
    params = config_factory.config_generator(model_name)
    params.architecture.num_classes = num_classes
    params.postprocess.max_total_size = max_detections

    if input_shape:
        params.efficientdet_parser.output_size = list(input_shape)
    input_shape = params.efficientdet_parser.output_size
    if inference_mode == "base64":
        inputs = tf.keras.layers.Input(shape=(1,), dtype="string", name="image_b64")
        with tf.name_scope("preprocess"):
            ouput_tensor, scales, image_sizes = Base64ImageProcessLayer(target_size=input_shape, name="preprocess")(
                inputs)
    elif inference_mode == "dynamic":
        inputs = tf.keras.layers.Input(shape=(None, None, 3), name="image_tensor")
        with tf.name_scope("preprocess"):
            ouput_tensor, scales, image_sizes = ResizeImageProcessLayer(target_size=input_shape, name="preprocess")(
                inputs)
    else:
        inputs = tf.keras.layers.Input(shape=(input_shape[0], input_shape[1], 3), name="image_tensor")
        ouput_tensor = tf.cast(inputs, tf.float32) / 255.0
        if (mean is not None) and (std is not None):
            ouput_tensor = (ouput_tensor - mean) / std
        shape_inputs = tf.keras.layers.Input(shape=(2,), name="shape_input")
        width_scales = input_shape[0] / shape_inputs[:, 0:1]
        hight_scales = input_shape[1] / shape_inputs[:, 1:]
        scales = tf.where(tf.keras.backend.greater(width_scales, hight_scales), hight_scales, width_scales)
        scaled_size = tf.round(shape_inputs * scales)
        scales = scaled_size / shape_inputs
        image_sizes = shape_inputs

    model_fn = EfficientDetModel(params)
    model, inference_model, lite_model = model_fn.build_model(params, inference_mode=True)
    if weights:
        model.load_weights(weights)
    outputs = inference_model(ouput_tensor)
    boxes, scores, classes, valid_detections = outputs[0], outputs[1], outputs[2], outputs[3]
    scales = tf.expand_dims(tf.concat([scales, scales], 1), 1)
    image_sizes = tf.expand_dims(tf.concat([image_sizes, image_sizes], axis=1), 1)
    boxes = boxes / scales
    boxes = tf.keras.backend.clip(boxes, 0.0, image_sizes)
    if inference_mode == "base64":
        model = tf.keras.Model(inputs, [boxes, scores, classes, valid_detections])
    elif inference_mode == "dynamic":
        model = tf.keras.Model(inputs, [boxes, scores, classes, valid_detections])
    else:
        model = tf.keras.Model([inputs, shape_inputs], [boxes, scores, classes, valid_detections])
    model.output_names[0] = "boxes"
    model.output_names[1] = "scores"
    model.output_names[2] = "labels"
    model.output_names[3] = "valid_detections"
    model.outputs[0]._name = "boxes"
    model.outputs[1]._name = "scores"
    model.outputs[2]._name = "labels"
    model.outputs[3]._name = "valid_detections"
    lite_inputs = tf.keras.layers.Input(shape=(input_shape[0], input_shape[1], 3), name="image_tensor")
    lite_ouput_tensor = tf.cast(lite_inputs, tf.float32) / 255.0
    if (mean is not None) and (std is not None):
        lite_ouput_tensor = (lite_ouput_tensor - mean) / std
    lite_ouput_tensor = lite_model(lite_ouput_tensor)
    boxes_lite, scores_lite = lite_ouput_tensor
    scores_lite = tf.keras.layers.Permute([2, 1], name="all_scores")(scores_lite)
    boxes_lite = tf.identity(boxes_lite, "all_boxes")
    scores_lite = tf.identity(scores_lite, "all_scores")

    lite_model_with_pre = tf.keras.Model(lite_inputs, [boxes_lite, scores_lite])
    lite_model_with_pre.output_names[0] = "all_boxes"
    lite_model_with_pre.output_names[1] = "all_scores"

    if serving_export and serving_path:
        os.makedirs(serving_path, exist_ok=True)
        serving_model_export(model, serving_path, version=version, auto_incre_version=auto_incre_version)
    if lite_path:
        print("保存lite模型")
        converter = tf.lite.TFLiteConverter.from_keras_model(lite_model_with_pre)
        if quantize_method:
            converter = quantize_model(converter, quantize_method, (100, *input_shape, 3))
        if allow_custom_ops:
            converter.allow_custom_ops = True
            converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS, tf.lite.OpsSet.TFLITE_BUILTINS]
        pathlib.Path(f"{lite_path}/{model_name}.tflite").write_bytes(converter.convert())
    return model, lite_model_with_pre
