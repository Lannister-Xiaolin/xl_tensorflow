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
"""Config template to train Retinanet."""

from xl_tensorflow.utils import params_dict
from . import base_config_rcnn

# pylint: disable=line-too-long
EFFICIENTDET_CFG = params_dict.ParamsDict(base_config_rcnn.BASE_CFG)
EFFICIENTDET_CFG.override({
    'type': 'efficientdet',
    'architecture': {
        'parser': 'efficientdet_parser',
    },
    'efficientdet_parser': {
        'output_size': [640, 640],
        'num_channels': 3,
        'match_threshold': 0.5,
        'unmatched_threshold': 0.5,
        'aug_rand_hflip': True,
        'aug_scale_min': 1.0,
        'aug_scale_max': 1.0,
        'use_autoaugment': False,
        'autoaugment_policy_name': 'v0',
        'skip_crowd_during_training': True,
        'max_num_instances': 100,
    },
    'efficientdet_head': {
        'anchors_per_location': 9,
        'num_convs': 4,
        'num_filters': 256,
        'use_separable_conv': False,
    },
    'efficientdet_loss': {
        'focal_loss_alpha': 0.25,
        'focal_loss_gamma': 1.5,
        'huber_loss_delta': 0.1,
        'box_loss_weight': 50,
    },
    'enable_summary': True,
}, is_strict=False)

EFFICIENTDET_RESTRICTIONS = [
]

EFFICIENTDET_CFG_DICT = {

}

efficientdet_model_param_dict = {
    'efficientdet-d0':
        {
            "name": 'efficientdet-d0',
            'architecture': {
                'backbone': 'efficientnet-b0',
                'min_level': 3,
                'max_level': 7,
                'multilevel_features': 'bifpn',
                'use_bfloat16': False,
                'num_classes': 91,
            },
            'fpn': {
                'fpn_cell_repeats': 3,  # efd
                'fpn_feat_dims': 64},
            'efficientdet_parser': {
                'output_size': [512, 512],
            }
        }
}
# pylint: enable=line-too-long
