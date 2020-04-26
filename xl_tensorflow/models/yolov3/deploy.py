#!usr/bin/env python3
# -*- coding: UTF-8 -*-
import os

from .evaluate import single_inference_model_serving
import tensorflow as tf
from xl_tensorflow.utils.deploy import serving_model_export  # ,tf_saved_model_to_lite
import numpy as np


def b64_yolo_serving_export(path, model_name, num_classes, pre_weights,
                            score_threshold=0.1, iou_threshold=.5
                            , max_detections=20, version=1,
                            auto_incre_version=True,
                            image_shape=(416, 416),
                            input_shape=(416, 416)):
    def preprocess_and_decode(img_str):
        img = tf.io.decode_base64(img_str)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.cast(img, tf.float32)

        img = tf.image.resize_with_pad(img, 416, 416,
                                       method=tf.image.ResizeMethod.BILINEAR)
        img = img / 255.0
        return img

    input64 = tf.keras.layers.Input(shape=(1,), dtype="string", name="image_b64")
    ouput_tensor = tf.keras.layers.Lambda(
        lambda img: tf.map_fn(lambda im: preprocess_and_decode(im[0]), img, dtype="float32"))(input64)
    model = single_inference_model_serving(model_name,
                                           pre_weights,
                                           num_classes, input_shape=input_shape,
                                           image_shape=image_shape,
                                           dynamic_shape=True, iou_threshold=iou_threshold,
                                           score_threshold=score_threshold, max_detections=max_detections)
    if pre_weights:
        model.load_weights(pre_weights)
    boxes, scores, classes = model(ouput_tensor)
    new_model = tf.keras.Model(input64, [boxes, scores, classes])
    new_model.output_names[0] = "boxes"
    new_model.output_names[1] = "scores"
    new_model.output_names[2] = "classes"
    import base64
    print(new_model.predict(
        [[base64.urlsafe_b64encode(open(r"E:\Temp\test\mix_corn_potato.jpg", "rb").read()).decode()]]))
    os.makedirs(path, exist_ok=True)
    print(model.summary())
    serving_model_export(new_model, path, version=version, auto_incre_version=auto_incre_version)
# b64_yolo_serving_export(r"E:\Programming\Python\Notes_ML\model_test\object_Detection\efficientdetd1",
#                                 i, 80, None,
#                                 class_specific_filter=True, auto_incre_version=True)
