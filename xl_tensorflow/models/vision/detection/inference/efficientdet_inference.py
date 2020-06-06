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


def efficiendet_inference_model(model_name="efficientdet-d0"):
    params = config_factory.config_generator(model_name)
    model_fn = EfficientDetModel(params)
    model = model_fn.build_model(params)


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
            image_size, 32),
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


# def det_post_process(params: Dict[Any, Any], cls_outputs: Dict[int, tf.Tensor],
#                      box_outputs: Dict[int, tf.Tensor], scales: List[float],
#                      min_score_thresh, max_boxes_to_draw):
#   """Post preprocessing the box/class predictions.
#
#   Args:
#     params: a parameter dictionary that includes `min_level`, `max_level`,
#       `batch_size`, and `num_classes`.
#     cls_outputs: an OrderDict with keys representing levels and values
#       representing logits in [batch_size, height, width, num_anchors].
#     box_outputs: an OrderDict with keys representing levels and values
#       representing box regression targets in [batch_size, height, width,
#       num_anchors * 4].
#     scales: a list of float values indicating image scale.
#     min_score_thresh: A float representing the threshold for deciding when to
#       remove boxes based on score.
#     max_boxes_to_draw: Max number of boxes to draw.
#
#   Returns:
#     detections_batch: a batch of detection results. Each detection is a tensor
#       with each row as [image_id, ymin, xmin, ymax, xmax, score, class].
#   """
#   if not params['batch_size']:
#     # Use combined version for dynamic batch size.
#     return det_post_process_combined(params, cls_outputs, box_outputs, scales,
#                                      min_score_thresh, max_boxes_to_draw)
#
#   # TODO(tanmingxing): refactor the code to make it more explicity.
#   outputs = {
#       'cls_outputs_all': [None],
#       'box_outputs_all': [None],
#       'indices_all': [None],
#       'classes_all': [None]
#   }
#   det_model_fn.add_metric_fn_inputs(params, cls_outputs, box_outputs, outputs,
#                                     -1)
#
#   # Create anchor_label for picking top-k predictions.
#   eval_anchors = anchor.Anchor(params['min_level'], params['max_level'],
#                                  params['num_scales'], params['aspect_ratios'],
#                                  params['anchor_scale'], params['image_size'])
#   anchor_labeler = anchor.AnchorLabeler(eval_anchors, params['num_classes'])
#
#   # Add all detections for each input image.
#   detections_batch = []
#   for index in range(params['batch_size']):
#     cls_outputs_per_sample = outputs['cls_outputs_all'][index]
#     box_outputs_per_sample = outputs['box_outputs_all'][index]
#     indices_per_sample = outputs['indices_all'][index]
#     classes_per_sample = outputs['classes_all'][index]
#     detections = anchor_labeler.generate_detections(
#         cls_outputs_per_sample,
#         box_outputs_per_sample,
#         indices_per_sample,
#         classes_per_sample,
#         image_id=[index],
#         image_scale=[scales[index]],
#         image_size=params['image_size'],
#         min_score_thresh=min_score_thresh,
#         max_boxes_to_draw=max_boxes_to_draw,
#         disable_pyfun=params.get('disable_pyfun'))
#     if params['batch_size'] > 1:
#       # pad to fixed length if batch size > 1.
#       padding_size = max_boxes_to_draw - tf.shape(detections)[0]
#       detections = tf.pad(detections, [[0, padding_size], [0, 0]])
#     detections_batch.append(detections)
#   return tf.stack(detections_batch, name='detections')


def det_post_process_combined(params, cls_outputs, box_outputs, scales,
                              min_score_thresh, max_boxes_to_draw):
    """A combined version of det_post_process with dynamic batch size support."""
    # todo 待校验，基本完成，就用这个，非极大抑制过程还可优化，即先过滤score再还原坐标
    batch_size = tf.shape(list(cls_outputs.values())[0])[0]
    cls_outputs_all = []
    box_outputs_all = []
    # Concatenates class and box of all levels into one tensor.
    for level in range(params.architecture.min_level, params.architecture.max_level + 1):
        if params.data_format == 'channels_first':
            cls_outputs[level] = tf.transpose(cls_outputs[level], [0, 2, 3, 1])
            box_outputs[level] = tf.transpose(box_outputs[level], [0, 2, 3, 1])

        cls_outputs_all.append(
            tf.reshape(cls_outputs[level], [batch_size, -1, params.architecture.num_classes]))
        box_outputs_all.append(tf.reshape(box_outputs[level], [batch_size, -1, 4]))
    cls_outputs_all = tf.concat(cls_outputs_all, 1)
    box_outputs_all = tf.concat(box_outputs_all, 1)

    # Create anchor_label for picking top-k predictions.
    eval_anchors = anchor.Anchor(params.architecture.min_level, params.architecture.max_level,
                                 params.anchor.num_scales, params.anchor.aspect_ratios,
                                 params.anchor.anchor_size, params.efficientdet_parser.output_size)
    anchor_boxes = eval_anchors.boxes
    scores = tf.math.sigmoid(cls_outputs_all)
    # apply bounding box regression to anchors
    # anchor_boxlist = box_list.BoxList(anchor_boxes)
    # box_coder = faster_rcnn_box_coder.FasterRcnnBoxCoder()
    boxes = faster_rcnn_box_coder.decode_box_outputs_tf(box_outputs_all, anchor_boxes)

    boxes = tf.expand_dims(boxes, axis=2)
    # scales = tf.expand_dims(scales, axis=-1)
    nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections = (
        tf.image.combined_non_max_suppression(
            boxes,
            scores,
            max_boxes_to_draw,
            max_boxes_to_draw,
            score_threshold=min_score_thresh,
            clip_boxes=False))
    del valid_detections  # to be used in futue.

    image_ids = tf.cast(
        tf.tile(
            tf.expand_dims(tf.range(batch_size), axis=1), [1, max_boxes_to_draw]),
        dtype=tf.float32)
    image_size = params.efficientdet_parser.output_size
    ymin = tf.clip_by_value(nmsed_boxes[..., 0], 0, image_size[0]) * scales[:, :1]
    xmin = tf.clip_by_value(nmsed_boxes[..., 1], 0, image_size[1]) * scales[:, 1:2]
    ymax = tf.clip_by_value(nmsed_boxes[..., 2], 0, image_size[0]) * scales[:, :1]
    xmax = tf.clip_by_value(nmsed_boxes[..., 3], 0, image_size[1]) * scales[:, 1:2]
    classes = tf.cast(nmsed_classes + 1, tf.float32)
    detection_list = [image_ids, ymin, xmin, ymax, xmax, nmsed_scores, classes]
    detections = tf.stack(detection_list, axis=2, name='detections')
    return detections
