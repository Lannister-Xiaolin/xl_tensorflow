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
from xl_tensorflow.models.vision.detection.dataloader.efficientdet_parser import anchor
from xl_tensorflow.models.vision.detection.dataloader.utils import input_utils, box_list, faster_rcnn_box_coder
from typing import Text, Dict, Any, List, Tuple, Union
import tensorflow as tf
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





def efficiendet_inference_model(model_name="efficientdet-d0",input_shape=(512, 512)):
    params = config_factory.config_generator(model_name)
    model_fn = EfficientDetModel(params)
    model, inference_model = model_fn.build_model(params, inference_mode=True)
    def preprocess_and_decode(img_str, input_shape=input_shape):
        img = tf.io.decode_base64(img_str)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.cast(img, tf.float32)
        img = tf.image.resize_with_pad(img, input_shape[0], input_shape[1],
                                       method=tf.image.ResizeMethod.BILINEAR)
        return img

    def preprocess_and_decode_shape(img_str):
        img = tf.io.decode_base64(img_str)
        img = tf.image.decode_jpeg(img, channels=3)
        shape = tf.keras.backend.shape(img)[:2]
        shape = tf.keras.backend.cast(shape, tf.float32)
        return shape

    def batch_decode_on_cpu(image_files):
        with tf.device("/cpu:0"):
            ouput_tensor = tf.map_fn(lambda im: preprocess_and_decode(im[0]), image_files, dtype="float32")
        return ouput_tensor

    def batch_decode_shape_on_cpu(image_files):
        with tf.device("/cpu:0"):
            shape_input = tf.map_fn(lambda im: preprocess_and_decode_shape(im[0]), image_files, dtype="float32")
        return shape_input
