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
"""Feature Pyramid Networks.

Feature Pyramid Networks were proposed in:
[1] Tsung-Yi Lin, Piotr Dollar, Ross Girshick, Kaiming He, Bharath Hariharan,
    , and Serge Belongie
    Feature Pyramid Networks for Object Detection. CVPR 2017.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import itertools
import logging

import tensorflow as tf

from tensorflow.python.keras import backend
from . import nn_ops
from ..ops import spatial_transform_ops
from ..utils.efficientdet_utils import get_feat_sizes, activation_fn
from xl_tensorflow.utils import hparams_config


@tf.keras.utils.register_keras_serializable(package='Text')
class WeightedAdd(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-4, activation="relu", **kwargs):
        """

        Args:
            epsilon:
            activation: relu and softmax
            **kwargs:
        """
        super(WeightedAdd, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.activation = tf.nn.softmax if activation == "softmax" else tf.nn.relu

    def build(self, input_shape):
        num_in = len(input_shape)
        self.w = self.add_weight(name=self.name,
                                 shape=(num_in,),
                                 initializer=tf.keras.initializers.constant(1 / num_in),
                                 trainable=True,
                                 dtype=tf.float32)

    def call(self, inputs, **kwargs):
        w = self.activation(self.w)
        weights_sum = tf.reduce_sum(self.w)
        x = tf.reduce_sum([(w[i] * inputs[i]) / (weights_sum + self.epsilon) for i in range(len(inputs))], axis=0)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(WeightedAdd, self).get_config()
        config.update({
            'epsilon': self.epsilon
        })
        return config



class BiFpn(object):
    """BiFeature pyramid networks.
    1、去掉training_bn参数
    2、以keras网络层为主，部分tf.nn层
    todo 把bifpn放到yolo种
    """

    def __init__(self,
                 min_level=3,
                 max_level=7,
                 ):
        """FPN initialization function.

        Args:
          min_level: `int` minimum level in FPN output feature maps.
          max_level: `int` maximum level in FPN output feature maps.
        """
        self._min_level = min_level
        self._max_level = max_level

    def get_fpn_config(self, fpn_name, min_level, max_level, weight_method):
        """Get fpn related configuration."""
        if not fpn_name:
            fpn_name = 'bifpn_fa'
        name_to_config = {
            'bifpn_sum': self.bifpn_sum_config(),
            'bifpn_fa': self.bifpn_fa_config(),
            'bifpn_dyn': self.bifpn_dynamic_config(min_level, max_level, weight_method)
        }
        return name_to_config[fpn_name]

    def fuse_features(self, nodes, weight_method):
        """Fuse features from different resolutions and return a weighted sum.

        Args:
          nodes: a list of tensorflow features at different levels
          weight_method: feature fusion method. One of:
            - "attn" - Softmax weighted fusion
            - "fastattn" - Fast normalzied feature fusion
            - "sum" - a sum of inputs

        Returns:
          A tensor denoting the fused feature.
        """
        dtype = nodes[0].dtype

        if weight_method == 'attn':
            new_node = WeightedAdd(activation="softmax")(nodes)
        elif weight_method == 'fastattn':
            new_node = WeightedAdd(activation="relu")(nodes)
        elif weight_method == 'sum':
            new_node = tf.add_n(nodes)
        else:
            raise ValueError(
                'unknown weight_method {}'.format(weight_method))

        return new_node

    def build_bifpn_layer(self, feats, feat_sizes, params):
        """Builds a feature pyramid given previous feature pyramid and config."""
        p = params  # use p to denote the network config.
        if p.fpn.fpn_config:
            fpn_config = p.fpn_config
        else:
            fpn_config = self.get_fpn_config(p.fpn.fpn_name, p.architecture.min_level, p.architecture.max_level,
                                             p.fpn.fpn_weight_method)

        num_output_connections = [0 for _ in feats]
        for i, fnode in enumerate(fpn_config.nodes):
            with tf.name_scope('fnode{}'.format(i)):
                logging.info('fnode %d : %s', i, fnode)
                new_node_height = feat_sizes[fnode['feat_level']]['height']
                new_node_width = feat_sizes[fnode['feat_level']]['width']
                nodes = []
                for idx, input_offset in enumerate(fnode['inputs_offsets']):
                    input_node = feats[input_offset]
                    num_output_connections[input_offset] += 1
                    input_node = spatial_transform_ops.resample_feature_map(
                        input_node, '{}_{}_{}'.format(idx, input_offset, len(feats)),
                        new_node_height, new_node_width, p.fpn.fpn_feat_dims,
                        p.fpn.apply_bn_for_resampling, p.is_training_bn,
                        p.fpn.conv_after_downsample,
                        p.fpn.use_native_resize_op,
                        p.fpn.pooling_type,
                        use_tpu=p.use_tpu,
                        data_format=params.data_format)
                    nodes.append(input_node)

                new_node = self.fuse_features(nodes, fpn_config.weight_method)
                with tf.name_scope('op_after_combine{}'.format(len(feats))):
                    if not p.fpn.conv_bn_act_pattern:
                        new_node = activation_fn(new_node, p.act_type)
                    if p.fpn.use_separable_conv:
                        conv_op = functools.partial(
                            tf.keras.layers.SeparableConv2D, depth_multiplier=1)
                    else:
                        conv_op = tf.keras.layers.Conv2D
                    new_node = conv_op(
                        filters=p.fpn.fpn_feat_dims,
                        kernel_size=(3, 3),
                        padding='same',
                        use_bias=not p.fpn.conv_bn_act_pattern,
                        data_format=params.data_format)(new_node)
                    # 拆分activation
                    act_type = None if not p.fpn.conv_bn_act_pattern else p.act_type
                    new_node = tf.keras.layers.BatchNormalization(
                        axis=1 if params.data_format == "channels_first" else -1,
                        momentum=p.norm_activation.batch_norm_momentum,
                        epsilon=p.norm_activation.batch_norm_epsilon)(new_node)
                    if act_type:
                        new_node = activation_fn(new_node, act_type)
                feats.append(new_node)
                num_output_connections.append(0)

        output_feats = {}
        for l in range(p.architecture.min_level, p.architecture.max_level + 1):
            for i, fnode in enumerate(reversed(fpn_config.nodes)):
                if fnode['feat_level'] == l:
                    output_feats[l] = feats[-1 - i]
                    break
        return output_feats

    def bifpn_sum_config(self):
        """BiFPN config with sum."""
        p = hparams_config.Config()
        p.nodes = [
            {'feat_level': 6, 'inputs_offsets': [3, 4]},
            {'feat_level': 5, 'inputs_offsets': [2, 5]},
            {'feat_level': 4, 'inputs_offsets': [1, 6]},
            {'feat_level': 3, 'inputs_offsets': [0, 7]},
            {'feat_level': 4, 'inputs_offsets': [1, 7, 8]},
            {'feat_level': 5, 'inputs_offsets': [2, 6, 9]},
            {'feat_level': 6, 'inputs_offsets': [3, 5, 10]},
            {'feat_level': 7, 'inputs_offsets': [4, 11]},
        ]
        p.weight_method = 'sum'
        return p

    def bifpn_fa_config(self):
        """BiFPN config with fast weighted sum."""
        p = self.bifpn_sum_config()
        p.weight_method = 'fastattn'
        return p

    def bifpn_dynamic_config(self, min_level, max_level, weight_method):
        """A dynamic bifpn config that can adapt to different min/max levels."""
        p = hparams_config.Config()
        p.weight_method = weight_method or 'fastattn'

        num_levels = max_level - min_level + 1
        node_ids = {min_level + i: [i] for i in range(num_levels)}

        level_last_id = lambda level: node_ids[level][-1]
        level_all_ids = lambda level: node_ids[level]
        id_cnt = itertools.count(num_levels)

        p.nodes = []
        for i in range(max_level - 1, min_level - 1, -1):
            # top-down path.
            p.nodes.append({
                'feat_level': i,
                'inputs_offsets': [level_last_id(i), level_last_id(i + 1)]
            })
            node_ids[i].append(next(id_cnt))

        for i in range(min_level + 1, max_level + 1):
            # bottom-up path.
            p.nodes.append({
                'feat_level': i,
                'inputs_offsets': level_all_ids(i) + [level_last_id(i - 1)]
            })
            node_ids[i].append(next(id_cnt))

        return p

    def __call__(self, multilevel_features, params):
        """Returns the FPN features for a given multilevel features.

        Args:
          multilevel_features: a `dict` containing `int` keys for continuous feature
            levels, e.g., [2, 3, 4, 5]. The values are corresponding features with
            shape [batch_size, height_l, width_l, num_filters].

        Returns:
          a `dict` containing `int` keys for continuous feature levels
          [min_level, min_level + 1, ..., max_level]. The values are corresponding
          FPN features with shape [batch_size, height_l, width_l, fpn_feat_dims].
        """
        # step 1: Build additional input features that are not from backbone.(ie. level 6 and 7)
        feats = []
        # with  tf.name_scope('bifpn'):
        with backend.get_graph().as_default(), tf.name_scope('bifpn'):

            for level in range(self._min_level, self._max_level + 1):
                if level in multilevel_features.keys():
                    feats.append(multilevel_features[level])
                else:
                    h_id, w_id = (1, 2)  # 不允许通道前置,即data_format必须等于channels_last
                    feats.append(
                        spatial_transform_ops.resample_feature_map(
                            feats[-1],
                            name='p%d' % level,
                            target_height=(feats[-1].shape[h_id] - 1) // 2 + 1,
                            target_width=(feats[-1].shape[w_id] - 1) // 2 + 1,
                            target_num_channels=params.fpn.fpn_feat_dims,
                            apply_bn=params.fpn.apply_bn_for_resampling,
                            is_training=params.is_training_bn,
                            conv_after_downsample=params.fpn.conv_after_downsample,
                            use_native_resize_op=params.fpn.use_native_resize_op,
                            pooling_type=params.fpn.pooling_type,
                            use_tpu=False,
                            data_format="channels_last"
                        ))
            feat_sizes = get_feat_sizes(params.efficientdet_parser.output_size[0], self._max_level)

            with tf.name_scope("bifpn_cells"):
                for rep in range(params.fpn.fpn_cell_repeats):
                    logging.info('building cell %d', rep)
                    new_feats = self.build_bifpn_layer(feats, feat_sizes, params)
                    feats = [
                        new_feats[level]
                        for level in range(
                            self._min_level, self._max_level + 1)
                    ]
            return new_feats
