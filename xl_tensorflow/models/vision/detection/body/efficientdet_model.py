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
"""Model defination for the RetinaNet Model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.keras import backend
from ..evaluation import factory as eval_factory
from . import base_model
from ..loss import losses as losses
from ..architecture import factory
from ..ops import postprocess_ops
from ..dataloader.utils import anchor_rcnn as  anchor


class EfficientDetModel(base_model.Model):
    """EfficientDet Model  function.
    hint：classnet和boxnet不同level的卷积层是共享的
        classnet和boxnet都是卷积层直接输出未与anchor匹配和坐标还原
    数据加载
        官方损失函数会把输入的坐标直接还原成tx,ty,tw,th
        真实标签未经onehot处理

    """

    def __init__(self, params):
        super(EfficientDetModel, self).__init__(params)

        # For eval metrics.
        self._params = params

        # Architecture generators.
        self._backbone_fn = factory.backbone_generator(params)
        self._fpn_fn = factory.multilevel_features_generator(params)
        # hint: 与keras版本相比多boxnet和classnet多了激活函数,导致速度变慢
        self._head_fn = factory.efficientdet_head_generator(params)

        # Loss function.
        # class loss与automl完全一致，需要确认背景类
        self._cls_loss_fn = losses.EfficientnetClassLoss(
            params.efficientdet_loss, params.architecture.num_classes)
        self._box_loss_fn = losses.RetinanetBoxLoss(params.efficientdet_loss)
        self._box_loss_weight = params.efficientdet_loss.box_loss_weight
        self._keras_model = None
        self._inference_keras_model = None
        # Predict function.
        self._generate_detections_fn = postprocess_ops.MultilevelDetectionGeneratorWithScoreFilter(
            params.architecture.min_level,
            params.architecture.max_level,
            params.postprocess)
        self._input_anchor = anchor.Anchor(
            params.architecture.min_level, params.architecture.max_level, params.anchor.num_scales,
            params.anchor.aspect_ratios, params.anchor.anchor_size, (*params.efficientdet_parser.output_size,))
        self._transpose_input = params.train.transpose_input
        assert not self._transpose_input, 'Transpose input is not supportted.'
        # Input layer.
        self._input_image_size = params.efficientdet_parser.output_size
        input_shape = (
                params.efficientdet_parser.output_size +
                [params.efficientdet_parser.num_channels])
        self._input_layer = tf.keras.layers.Input(
            shape=input_shape, name='',
            dtype=tf.bfloat16 if self._use_bfloat16 else tf.float32)

    def build_outputs(self, inputs, mode, inference_mode=False):
        # If the input image is transposed (from NHWC to HWCN), we need to revert it
        # back to the original shape before it's used in the computation.
        if self._transpose_input:
            inputs = tf.transpose(inputs, [3, 0, 1, 2])

        backbone_features = self._backbone_fn(input_tensor=inputs, fpn_features=True, activation=self._params.act_type)
        fpn_features = self._fpn_fn(
            backbone_features, self._params)
        cls_outputs, box_outputs = self._head_fn(
            fpn_features, is_training=None)

        if self._use_bfloat16:
            levels = cls_outputs.keys()
            for level in levels:
                cls_outputs[level] = tf.cast(cls_outputs[level], tf.float32)
                box_outputs[level] = tf.cast(box_outputs[level], tf.float32)

        model_outputs = {
            'cls_outputs': cls_outputs,
            'box_outputs': box_outputs,
        }
        evaluate_outputs = self.post_processing_inference(model_outputs, inference_mode)
        return model_outputs, evaluate_outputs
    #todo
    def build_outputs_keras(self, inputs, mode, inference_mode=False):
        # If the input image is transposed (from NHWC to HWCN), we need to revert it
        # back to the original shape before it's used in the computation.
        if self._transpose_input:
            inputs = tf.transpose(inputs, [3, 0, 1, 2])

        backbone_features = self._backbone_fn(input_tensor=inputs, fpn_features=True, activation=self._params.act_type)
        fpn_features = self._fpn_fn(
            backbone_features, self._params)
        cls_outputs, box_outputs = self._head_fn(
            fpn_features, is_training=None)

        if self._use_bfloat16:
            levels = cls_outputs.keys()
            for level in levels:
                cls_outputs[level] = tf.cast(cls_outputs[level], tf.float32)
                box_outputs[level] = tf.cast(box_outputs[level], tf.float32)

        model_outputs = {
            'cls_outputs': cls_outputs,
            'box_outputs': box_outputs,
        }
        keys = cls_outputs.keys()
        classification = []
        regression = []
        for key in keys:
            classification.append(
                tf.keras.layers.Reshape((-1, self._params.architecture.num_classes))(cls_outputs[key]))
            regression.append(
                tf.keras.layers.Reshape((-1, 4))(box_outputs[key]))

        classification = tf.keras.layers.Concatenate(axis=1, name='classification')(classification)
        regression = tf.keras.layers.Concatenate(axis=1, name='regression')(regression)



        evaluate_outputs = self.post_processing_inference(model_outputs, inference_mode)
        return model_outputs, evaluate_outputs

    def build_loss_fn(self):
        if self._keras_model is None:
            raise ValueError('build_loss_fn() must be called after build_model().')

        filter_fn = self.make_filter_trainable_variables_fn()
        trainable_variables = filter_fn(self._keras_model.trainable_variables)

        def _total_loss_fn(labels, outputs):
            cls_loss = self._cls_loss_fn(outputs['cls_outputs'],
                                         labels['cls_targets'],
                                         labels['num_positives'])
            box_loss = self._box_loss_fn(outputs['box_outputs'],
                                         labels['box_targets'],
                                         labels['num_positives'])
            model_loss = cls_loss + self._box_loss_weight * box_loss
            l2_regularization_loss = self.weight_decay_loss(trainable_variables)
            total_loss = model_loss + l2_regularization_loss
            return {
                'total_loss': total_loss,
                'cls_loss': cls_loss,
                'box_loss': box_loss,
                'model_loss': model_loss,
                'l2_regularization_loss': l2_regularization_loss,
            }

        return _total_loss_fn

    # todo 变成keras形式
    def build_loss_fn_keras(self):
        if self._keras_model is None:
            raise ValueError('build_loss_fn() must be called after build_model().')

        filter_fn = self.make_filter_trainable_variables_fn()
        trainable_variables = filter_fn(self._keras_model.trainable_variables)

        def _total_loss_fn(labels, outputs):
            cls_loss = self._cls_loss_fn(outputs['cls_outputs'],
                                         labels['cls_targets'],
                                         labels['num_positives'])
            box_loss = self._box_loss_fn(outputs['box_outputs'],
                                         labels['box_targets'],
                                         labels['num_positives'])
            model_loss = cls_loss + self._box_loss_weight * box_loss
            l2_regularization_loss = self.weight_decay_loss(trainable_variables)
            total_loss = model_loss + l2_regularization_loss
            return {
                'total_loss': total_loss,
                'cls_loss': cls_loss,
                'box_loss': box_loss,
                'model_loss': model_loss,
                'l2_regularization_loss': l2_regularization_loss,
            }

        return _total_loss_fn

    def build_model(self, params, mode=None, inference_mode=False):
        if self._keras_model is None:
            with backend.get_graph().as_default():
                outputs, evaluate_outputs = self.model_outputs(self._input_layer, mode, inference_mode=inference_mode)

                model = tf.keras.models.Model(
                    inputs=self._input_layer, outputs=outputs, name=params.name)
                inference_model = tf.keras.models.Model(
                    inputs=self._input_layer, outputs=evaluate_outputs, name=params.name + "_inference")
                assert model is not None, 'Fail to build tf.keras.Model.'
                model.optimizer = self.build_optimizer()
                self._keras_model = model
                self._inference_keras_model = inference_model
        return self._keras_model, self._inference_keras_model

    def post_processing_inference(self, outputs, inference_mode=False):
        boxes, scores, classes, valid_detections = self._generate_detections_fn(
            outputs['box_outputs'], outputs['cls_outputs'],
            self._input_anchor.multilevel_boxes, self._input_image_size)
        # Discards the old output tensors to save memory. The `cls_outputs` and
        # `box_outputs` are pretty big and could potentiall lead to memory issue.
        if inference_mode:
            outputs = valid_detections, boxes, classes, scores
        else:
            outputs = {
                # 'source_id': labels['groundtruths']['source_id'],
                # 'image_info': labels['image_info'],
                'num_detections': valid_detections,
                'detection_boxes': boxes,
                'detection_classes': classes,
                'detection_scores': scores,
                'box_outputs': outputs['box_outputs'],
                'cls_outputs': outputs['cls_outputs']
            }

        return outputs

    def eval_metrics(self):
        return eval_factory.evaluator_generator(self._params.eval)
