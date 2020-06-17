# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Data parser and processing.

Parse image and ground truths in a dataset to training targets and package them
into (image, labels) tuple for RetinaNet.

T.-Y. Lin, P. Goyal, R. Girshick, K. He,  and P. Dollar
Focal Loss for Dense Object Detection. arXiv:1708.02002
"""

import tensorflow as tf
import numpy as np
from xl_tensorflow.models.vision.detection.dataloader.utils.mode_keys import TRAIN, PREDICT_WITH_GT, EVAL, PREDICT
from .utils.anchors_yolo import YOLOV3_ANCHORS, YOLOV4_ANCHORS
from .utils import tf_example_decoder
from .utils import input_utils, box_utils


def process_source_id(source_id):
    """Processes source_id to the right format."""
    if source_id.dtype == tf.string:
        source_id = tf.cast(tf.strings.to_number(source_id), tf.int32)
    with tf.control_dependencies([source_id]):
        source_id = tf.cond(
            pred=tf.equal(tf.size(input=source_id), 0),
            true_fn=lambda: tf.cast(tf.constant(-1), tf.int32),
            false_fn=lambda: tf.identity(source_id))
    return source_id


def pad_groundtruths_to_fixed_size(gt, n):
    """Pads the first dimension of groundtruths labels to the fixed size."""
    gt['boxes'] = input_utils.pad_to_fixed_size(gt['boxes'], n, -1)
    gt['is_crowds'] = input_utils.pad_to_fixed_size(gt['is_crowds'], n, 0)
    gt['areas'] = input_utils.pad_to_fixed_size(gt['areas'], n, -1)
    gt['classes'] = input_utils.pad_to_fixed_size(gt['classes'], n, -1)
    return gt


def anchor_grid_align(true_boxes, input_shape, anchors, num_classes):
    '''Preprocess true boxes to training input format
       Parameters
       ----------
       true_boxes: array, shape=(m, T, 5)
           Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
       input_shape: array-like, hw, multiples of 32
       anchors: array, shape=(N, 2), 2 refer to wh, N refer to number of achors
       num_classes: integer
       Returns
       -------
       y_true: list of array, shape like yolo_outputs, xywh are reletive value（即相对值，相对整图比例）
       '''
    num_stages = len(anchors) // 3
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_stages == 3 else [[3, 4, 5], [1, 2, 3]]
    true_boxes = tf.cast(true_boxes, tf.float32)
    # todo 待验证，此处修正位置
    true_boxes = tf.concat([true_boxes[..., 0:2][..., ::-1], true_boxes[..., 2:4][..., ::-1]], -1)
    input_shape = tf.cast(input_shape, tf.int32)
    input_shape_float = tf.cast(input_shape, tf.float32)
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    xy = boxes_xy / input_shape_float[::-1]
    wh = boxes_wh / input_shape_float[::-1]
    true_boxes = tf.concat([xy, wh], -1)
    m = true_boxes.shape[0]
    grid_shapes = [input_shape * (2 ** l) // 32 for l in range(num_stages)]
    y_true = [tf.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes),
                       dtype='float32') for l in range(num_stages)]
    # Expand dim to apply broadcasting.
    anchors = tf.expand_dims(anchors, 0)
    anchors = tf.cast(anchors, tf.float32)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0
    # for b in range(m):
    # Discard zero rows.
    wh = boxes_wh[valid_mask]
    if len(wh) == 0: return y_true
    # Expand dim to apply broadcasting.
    wh = tf.expand_dims(wh, -2)
    box_maxes = wh / 2.
    box_mins = -box_maxes
    # min和max这两个对于单个box来说是正负两个数
    intersect_mins = tf.maximum(box_mins, anchor_mins)
    intersect_maxes = tf.minimum(box_maxes, anchor_maxes)  # 选择box与anchor中较小的值
    # print(intersect_mins.shape, tf.keras.backend.sum(intersect_maxes == (-intersect_mins)))
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    # 计算box与每个anchor的交叉区域，已确认对于单个box无误
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    box_area = wh[..., 0] * wh[..., 1]
    anchor_area = anchors[..., 0] * anchors[..., 1]
    iou = intersect_area / (box_area + anchor_area - intersect_area)
    # Find best anchor for each true box
    best_anchor = tf.argmax(iou, axis=-1)
    if len(best_anchor) > 1:
        print("fucky")
    for t, n in enumerate(best_anchor):
        for l in range(num_stages):
            if n in anchor_mask[l]:
                # i,j代表grid坐标
                i = tf.floor(true_boxes[t, 0] * grid_shapes[l][1]).astype('int32')
                j = tf.floor(true_boxes[t, 1] * grid_shapes[l][0]).astype('int32')
                k = anchor_mask[l].index(n)
                c = true_boxes[t, 4].astype('int32')
                y_true[l][j, i, k, 0:4] = true_boxes[t, 0:4]
                y_true[l][j, i, k, 4] = 1
                y_true[l][j, i, k, 5 + c] = 1

    return y_true


def anchor_grid_align_py(true_boxes, input_shape, anchors, num_classes):
    num_layers = len(anchors) // 3  # default setting
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2][..., ::-1] + true_boxes[..., 2:4][..., ::-1]) // 2
    boxes_wh = true_boxes[..., 2:4][..., ::-1] - true_boxes[..., 0:2][..., ::-1]
    # 相对值
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    grid_shapes = [input_shape * (2 ** l) // 32 for l in range(num_layers)]
    # 输入值batch * 26 *26 *3 * (5+classes)
    y_true = [np.zeros((grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes),
                       dtype='float32') for l in range(num_layers)]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 1e-5

    wh = boxes_wh[valid_mask]
    if len(wh) == 0: return y_true
    # Expand dim to apply broadcasting.
    wh = np.expand_dims(wh, -2)
    box_maxes = wh / 2.
    box_mins = -box_maxes
    # min和max这两个对于单个box来说是正负两个数
    intersect_mins = np.maximum(box_mins, anchor_mins)
    intersect_maxes = np.minimum(box_maxes, anchor_maxes)  # 选择box与anchor中较小的值
    # print(intersect_mins.shape, np.sum(intersect_maxes == (-intersect_mins)))
    intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
    # 计算box与每个anchor的交叉区域，已确认对于单个box无误
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    box_area = wh[..., 0] * wh[..., 1]
    anchor_area = anchors[..., 0] * anchors[..., 1]
    iou = intersect_area / (box_area + anchor_area - intersect_area)
    # Find best anchor for each true box

    best_anchor = np.argmax(iou, axis=-1)
    # print(iou.shape, best_anchor)
    for t, n in enumerate(best_anchor):
        for l in range(num_layers):
            if n in anchor_mask[l]:
                # i,j代表grid坐标
                i = np.floor(true_boxes[t, 0] * grid_shapes[l][1]).astype('int32')
                j = np.floor(true_boxes[t, 1] * grid_shapes[l][0]).astype('int32')
                k = anchor_mask[l].index(n)
                c = true_boxes[t, 4].astype('int32')
                y_true[l][j, i, k, 0:4] = true_boxes[t, 0:4]
                y_true[l][j, i, k, 4] = 1
                y_true[l][j, i, k, 5 + c] = 1
    y_true = tuple(y_true)
    return y_true


class Parser(object):
    """Parser to parse an image and its annotations into a dictionary of tensors."""

    def __init__(self,
                 output_size,
                 num_classes=20,
                 min_level=1,
                 max_level=7,
                 anchor=YOLOV3_ANCHORS,
                 match_threshold=0.5,
                 unmatched_threshold=0.5,
                 aug_rand_hflip=True,
                 aug_scale_min=1.0,
                 aug_scale_max=1.0,
                 use_autoaugment=False,
                 autoaugment_policy_name='v0',
                 autoaugment_ratio=0.8,
                 skip_crowd_during_training=False,
                 max_num_instances=100,
                 use_bfloat16=False,
                 mode=TRAIN):
        """Initializes parameters for parsing annotations in the dataset.

        Args:
          output_size: `Tensor` or `list` for [height, width] of output image. The
            output_size should be divided by the largest feature stride 2^max_level.
          min_level: `int` number of minimum level of the output feature pyramid.
          max_level: `int` number of maximum level of the output feature pyramid.
          num_scales: `int` number representing intermediate scales added
            on each level. For instances, num_scales=2 adds one additional
            intermediate anchor scales [2^0, 2^0.5] on each level.
          aspect_ratios: `list` of float numbers representing the aspect raito
            anchors added on each level. The number indicates the ratio of width to
            height. For instances, aspect_ratios=[1.0, 2.0, 0.5] adds three anchors
            on each scale level.
          anchor_size: `float` number representing the scale of size of the base
            anchor to the feature stride 2^level.
          match_threshold: `float` number between 0 and 1 representing the
            lower-bound threshold to assign positive labels for anchors. An anchor
            with a score over the threshold is labeled positive.
          unmatched_threshold: `float` number between 0 and 1 representing the
            upper-bound threshold to assign negative labels for anchors. An anchor
            with a score below the threshold is labeled negative.
          aug_rand_hflip: `bool`, if True, augment training with random
            horizontal flip.
          aug_scale_min: `float`, the minimum scale applied to `output_size` for
            data augmentation during training.
          aug_scale_max: `float`, the maximum scale applied to `output_size` for
            data augmentation during training.
          use_autoaugment: `bool`, if True, use the AutoAugment augmentation policy
            during training.
          autoaugment_policy_name: `string` that specifies the name of the
            AutoAugment policy that will be used during training.
          skip_crowd_during_training: `bool`, if True, skip annotations labeled with
            `is_crowd` equals to 1.
          max_num_instances: `int` number of maximum number of instances in an
            image. The groundtruth data will be padded to `max_num_instances`.
          use_bfloat16: `bool`, if True, cast output image to tf.bfloat16.
          mode: a ModeKeys. Specifies if this is training, evaluation, prediction
            or prediction with groundtruths in the outputs.
        """
        self._mode = mode
        self._max_num_instances = max_num_instances
        self._skip_crowd_during_training = skip_crowd_during_training
        self._is_training = (mode == TRAIN)

        self._example_decoder = tf_example_decoder.TfExampleDecoder(
            include_mask=False)

        # Anchor.
        self._output_size = output_size
        self._min_level = min_level
        self._max_level = max_level
        # self._num_scales = num_scales
        # self._aspect_ratios = aspect_ratios
        self._anchor = anchor
        self._match_threshold = match_threshold
        self._unmatched_threshold = unmatched_threshold
        self._num_classes = num_classes
        # Data augmentation.
        self._aug_rand_hflip = aug_rand_hflip
        self._aug_scale_min = aug_scale_min
        self._aug_scale_max = aug_scale_max
        self._autoaugment_ratio = autoaugment_ratio
        # Data Augmentation with AutoAugment.
        self._use_autoaugment = use_autoaugment
        self._autoaugment_policy_name = autoaugment_policy_name

        # Device.
        self._use_bfloat16 = use_bfloat16

        # Data is parsed depending on the model Modekey.
        if mode == TRAIN:
            self._parse_fn = self._parse_train_data
        elif mode == EVAL:
            self._parse_fn = self._parse_eval_data
        elif mode == PREDICT or mode == PREDICT_WITH_GT:
            self._parse_fn = self._parse_predict_data
        else:
            raise ValueError('mode is not defined.')

    def __call__(self, value):
        """Parses data to an image and associated training labels.

        Args:
          value: a string tensor holding a serialized tf.Example proto.

        Returns:
          image: image tensor that is preproessed to have normalized value and
            dimension [output_size[0], output_size[1], 3]
          labels:
            cls_targets: ordered dictionary with keys
              [min_level, min_level+1, ..., max_level]. The values are tensor with
              shape [height_l, width_l, anchors_per_location]. The height_l and
              width_l represent the dimension of class logits at l-th level.
            box_targets: ordered dictionary with keys
              [min_level, min_level+1, ..., max_level]. The values are tensor with
              shape [height_l, width_l, anchors_per_location * 4]. The height_l and
              width_l represent the dimension of bounding box regression output at
              l-th level.
            num_positives: number of positive anchors in the image.
            anchor_boxes: ordered dictionary with keys
              [min_level, min_level+1, ..., max_level]. The values are tensor with
              shape [height_l, width_l, 4] representing anchor boxes at each level.
            image_info: a 2D `Tensor` that encodes the information of the image and
              the applied preprocessing. It is in the format of
              [[original_height, original_width], [scaled_height, scaled_width],
               [y_scale, x_scale], [y_offset, x_offset]].
            groundtruths:
              source_id: source image id. Default value -1 if the source id is empty
                in the groundtruth annotation.
              boxes: groundtruth bounding box annotations. The box is represented in
                [y1, x1, y2, x2] format. The tennsor is padded with -1 to the fixed
                dimension [self._max_num_instances, 4].
              classes: groundtruth classes annotations. The tennsor is padded with
                -1 to the fixed dimension [self._max_num_instances].
              areas: groundtruth areas annotations. The tennsor is padded with -1
                to the fixed dimension [self._max_num_instances].
              is_crowds: groundtruth annotations to indicate if an annotation
                represents a group of instances by value {0, 1}. The tennsor is
                padded with 0 to the fixed dimension [self._max_num_instances].
        """

        with tf.name_scope('parser'):
            # 1 - 加载原始数据tfexample
            data = self._example_decoder.decode(value)
            # 2  - 预处理
            return self._parse_fn(data)

    # @tf.autograph.experimental.do_not_convert
    def _parse_train_data(self, data):
        """Parses data for training and evaluation."""
        #  1 - 读取基本数据
        classes = data['groundtruth_classes']
        boxes = data['groundtruth_boxes']  # [ymin, xmin, ymax, xmax]
        image = data['image']
        # Gets original image and its size
        image_shape = tf.shape(input=image)[0:2]
        source_id = data['source_id']
        # XL:暂不处理__start
        is_crowds = data['groundtruth_is_crowd']
        # Skips annotations with `is_crowd` = True.
        if self._skip_crowd_during_training and self._is_training:
            num_groundtrtuhs = tf.shape(input=classes)[0]
            with tf.control_dependencies([num_groundtrtuhs, is_crowds]):
                indices = tf.cond(
                    pred=tf.greater(tf.size(input=is_crowds), 0),
                    true_fn=lambda: tf.where(tf.logical_not(is_crowds))[:, 0],
                    false_fn=lambda: tf.cast(tf.range(num_groundtrtuhs), tf.int64))
            classes = tf.gather(classes, indices)
            boxes = tf.gather(boxes, indices)
        # XL:暂不处理__end

        # 2 - 是否使用数据增强
        if self._use_autoaugment and self._is_training:
            from .augment import autoaugment  # pylint: disable=g-import-not-at-top
            image, boxes = autoaugment.distort_image_with_autoaugment(
                image, boxes, self._autoaugment_policy_name, False, ratio=self._autoaugment_ratio)

        # 3 - 图片归一化
        # Normalizes image with mean and std pixel values.
        image = input_utils.normalize_image(image)

        # Flips image randomly during training.
        if self._aug_rand_hflip:
            image, boxes = input_utils.random_horizontal_flip(image, boxes)
        # Converts boxes from normalized coordinates to pixel coordinates.
        boxes = box_utils.denormalize_boxes(boxes, image_shape)

        # Resizes and crops image.
        image, image_info = input_utils.resize_and_crop_image(
            image,
            self._output_size,
            padded_size=self._output_size,
            aug_scale_min=self._aug_scale_min,
            aug_scale_max=self._aug_scale_max)
        image_height, image_width, _ = image.get_shape().as_list()
        # Resizes and crops boxes.
        image_scale = image_info[2, :]
        offset = image_info[3, :]
        boxes = input_utils.resize_and_crop_boxes(
            boxes, image_scale, image_info[1, :], offset)
        indices = box_utils.get_non_empty_box_indices(boxes)
        boxes = tf.gather(boxes, indices)
        classes = tf.gather(classes, indices)
        classes = tf.reshape(tf.cast(classes, dtype=tf.float32), [-1, 1])
        true_boxes = tf.concat([boxes, classes], -1)
        y1, y2, y3 = tf.py_function(func=anchor_grid_align_py,
                                    inp=[true_boxes, [*self._output_size], self._anchor, self._num_classes],
                                    Tout=(tf.float32, tf.float32, tf.float32))
        y1.set_shape([None, None, 3, self._num_classes + 5])
        y2.set_shape([None, None, 3, self._num_classes + 5])
        y3.set_shape([None, None, 3, self._num_classes + 5])
        # If bfloat16 is used, casts input image to tf.bfloat16.
        if self._use_bfloat16:
            image = tf.cast(image, dtype=tf.bfloat16)

        # Packs labels for model_fn outputs.
        return image, (y1, y2, y3)

    # def _parse_eval_data(self, data):
    #     """Parses data for training and evaluation."""
    #     groundtruths = {}
    #     classes = data['groundtruth_classes']
    #     boxes = data['groundtruth_boxes']
    #
    #     # Gets original image and its size.
    #     image = data['image']
    #     image_shape = tf.shape(input=image)[0:2]
    #
    #     # Normalizes image with mean and std pixel values.
    #     image = input_utils.normalize_image(image)
    #
    #     # Converts boxes from normalized coordinates to pixel coordinates.
    #     boxes = box_utils.denormalize_boxes(boxes, image_shape)
    #
    #     # Resizes and crops image.
    #     image, image_info = input_utils.resize_and_crop_image(
    #         image,
    #         self._output_size,
    #         padded_size=input_utils.compute_padded_size(
    #             self._output_size, 2 ** self._max_level),
    #         aug_scale_min=1.0,
    #         aug_scale_max=1.0)
    #     image_height, image_width, _ = image.get_shape().as_list()
    #
    #     # Resizes and crops boxes.
    #     image_scale = image_info[2, :]
    #     offset = image_info[3, :]
    #     boxes = input_utils.resize_and_crop_boxes(
    #         boxes, image_scale, image_info[1, :], offset)
    #     # Filters out ground truth boxes that are all zeros.
    #     indices = box_utils.get_non_empty_box_indices(boxes)
    #     boxes = tf.gather(boxes, indices)
    #     classes = tf.gather(classes, indices)
    #
    #     # Assigns anchors.
    #     input_anchor = anchor.Anchor(
    #         self._min_level, self._max_level, self._num_scales,
    #         self._aspect_ratios, self._anchor_size, (image_height, image_width))
    #     anchor_labeler = anchor.AnchorLabeler(
    #         input_anchor, self._match_threshold, self._unmatched_threshold)
    #     (cls_targets, box_targets, num_positives) = anchor_labeler.label_anchors(
    #         boxes,
    #         tf.cast(tf.expand_dims(classes, axis=1), tf.float32))
    #
    #     # If bfloat16 is used, casts input image to tf.bfloat16.
    #     if self._use_bfloat16:
    #         image = tf.cast(image, dtype=tf.bfloat16)
    #
    #     # Sets up groundtruth data for evaluation.
    #     groundtruths = {
    #         'source_id': data['source_id'],
    #         'num_groundtrtuhs': tf.shape(data['groundtruth_classes']),
    #         'image_info': image_info,
    #         'boxes': box_utils.denormalize_boxes(
    #             data['groundtruth_boxes'], image_shape),
    #         'classes': data['groundtruth_classes'],
    #         'areas': data['groundtruth_area'],
    #         'is_crowds': tf.cast(data['groundtruth_is_crowd'], tf.int32),
    #     }
    #     groundtruths['source_id'] = process_source_id(groundtruths['source_id'])
    #     groundtruths = pad_groundtruths_to_fixed_size(
    #         groundtruths, self._max_num_instances)
    #
    #     # Packs labels for model_fn outputs.
    #     labels = {
    #         'cls_targets': cls_targets,
    #         'box_targets': box_targets,
    #         'anchor_boxes': input_anchor.multilevel_boxes,
    #         'num_positives': num_positives,
    #         'image_info': image_info,
    #         'groundtruths': groundtruths,
    #     }
    #     return image, labels
    #
    # def _parse_predict_data(self, data):
    #     """Parses data for prediction."""
    #     # Gets original image and its size.
    #     image = data['image']
    #     image_shape = tf.shape(input=image)[0:2]
    #
    #     # Normalizes image with mean and std pixel values.
    #     image = input_utils.normalize_image(image)
    #
    #     # Resizes and crops image.
    #     image, image_info = input_utils.resize_and_crop_image(
    #         image,
    #         self._output_size,
    #         padded_size=input_utils.compute_padded_size(
    #             self._output_size, 2 ** self._max_level),
    #         aug_scale_min=1.0,
    #         aug_scale_max=1.0)
    #     image_height, image_width, _ = image.get_shape().as_list()
    #
    #     # If bfloat16 is used, casts input image to tf.bfloat16.
    #     if self._use_bfloat16:
    #         image = tf.cast(image, dtype=tf.bfloat16)
    #
    #     # Compute Anchor boxes.
    #     input_anchor = anchor.Anchor(
    #         self._min_level, self._max_level, self._num_scales,
    #         self._aspect_ratios, self._anchor_size, (image_height, image_width))
    #
    #     labels = {
    #         'anchor_boxes': input_anchor.multilevel_boxes,
    #         'image_info': image_info,
    #     }
    #     # If mode is PREDICT_WITH_GT, returns groundtruths and training targets
    #     # in labels.
    #     if self._mode == ModeKeys.PREDICT_WITH_GT:
    #         # Converts boxes from normalized coordinates to pixel coordinates.
    #         boxes = box_utils.denormalize_boxes(
    #             data['groundtruth_boxes'], image_shape)
    #         groundtruths = {
    #             'source_id': data['source_id'],
    #             'num_detections': tf.shape(data['groundtruth_classes']),
    #             'boxes': boxes,
    #             'classes': data['groundtruth_classes'],
    #             'areas': data['groundtruth_area'],
    #             'is_crowds': tf.cast(data['groundtruth_is_crowd'], tf.int32),
    #         }
    #         groundtruths['source_id'] = process_source_id(groundtruths['source_id'])
    #         groundtruths = pad_groundtruths_to_fixed_size(
    #             groundtruths, self._max_num_instances)
    #         labels['groundtruths'] = groundtruths
    #
    #         # Computes training objective for evaluation loss.
    #         classes = data['groundtruth_classes']
    #
    #         image_scale = image_info[2, :]
    #         offset = image_info[3, :]
    #         boxes = input_utils.resize_and_crop_boxes(
    #             boxes, image_scale, image_info[1, :], offset)
    #         # Filters out ground truth boxes that are all zeros.
    #         indices = box_utils.get_non_empty_box_indices(boxes)
    #         boxes = tf.gather(boxes, indices)
    #
    #         # Assigns anchors.
    #         anchor_labeler = anchor.AnchorLabeler(
    #             input_anchor, self._match_threshold, self._unmatched_threshold)
    #         (cls_targets, box_targets, num_positives) = anchor_labeler.label_anchors(
    #             boxes,
    #             tf.cast(tf.expand_dims(classes, axis=1), tf.float32))
    #         labels['cls_targets'] = cls_targets
    #         labels['box_targets'] = box_targets
    #         labels['num_positives'] = num_positives
    #     return image, labels
