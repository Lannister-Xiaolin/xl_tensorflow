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
        'min_level': 3,
        'max_level': 7,
        'multilevel_features': 'bifpn',
        'use_bfloat16': False,
        'num_classes': 91,
    },
    'efficientdet_parser': {
        'output_size': [640, 640],
        'num_channels': 3,
        'match_threshold': 0.5,
        'unmatched_threshold': 0.5,
        'aug_rand_hflip': True,
        'aug_scale_min': 0.1,
        'aug_scale_max': 2.0,
        'use_autoaugment': True,
        'autoaugment_policy_name': 'v0',
        "autoaugment_ratio": 1.0,
        'skip_crowd_during_training': True,
        'max_num_instances': 100,
    },
    'efficientdet_head': {
        'anchors_per_location': 9,
        'num_convs': 4,
        'num_filters': 256,
        'use_separable_conv': False,
    },
    'train': {"l2_weight_decay": 4e-5},
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

            },
            'fpn': {
                'fpn_cell_repeats': 3,  # efd
                'fpn_feat_dims': 64},
            'efficientdet_parser': {
                'output_size': [512, 512],
            },
            'efficientdet_head': {
                'num_convs': 3,
                'num_filters': 64,
                'use_separable_conv': False,
            }
        },
    'efficientdetlite-d0':
        {
            "name": 'efficientdetlite-d0',
            'architecture': {
                'backbone': 'efficientnetlite-b0',

            },
            'fpn': {
                'fpn_cell_repeats': 3,  # efd
                'fpn_feat_dims': 64},
            'efficientdet_parser': {
                'output_size': [512, 512],
            },
            'efficientdet_head': {
                'num_convs': 3,
                'num_filters': 64,
                'use_separable_conv': False,
            }
        },
    'efficientdet-d1':
        {
            "name": 'efficientdet-d1',
            'architecture': {
                'backbone': 'efficientnet-b1',
            },
            'fpn': {
                'fpn_cell_repeats': 4,  # efd
                'fpn_feat_dims': 88},
            'efficientdet_parser': {
                'output_size': [640, 640],
            },
            'efficientdet_head': {
                'num_convs': 3,
                'num_filters': 88,
                'use_separable_conv': False,
            }
        },
    'efficientdetlite-d1':
        {
            "name": 'efficientdetlite-d1',
            'architecture': {
                'backbone': 'efficientnetlite-b1',

            },
            'fpn': {
                'fpn_cell_repeats': 3,  # efd
                'fpn_feat_dims': 88},
            'efficientdet_parser': {
                'output_size': [640, 640],
            },
            'efficientdet_head': {
                'num_convs': 3,
                'num_filters': 88,
                'use_separable_conv': False,
            }
        },
    'efficientdet-d2':
        {
            "name": 'efficientdet-d2',
            'architecture': {
                'backbone': 'efficientnet-b2',
            },
            'fpn': {
                'fpn_cell_repeats': 5,  # efd
                'fpn_feat_dims': 112},
            'efficientdet_parser': {
                'output_size': [768, 768],
            },
            'efficientdet_head': {
                'num_convs': 3,
                'num_filters': 112,
                'use_separable_conv': False,
            }
        },
    'efficientdetlite-d2':
        {
            "name": 'efficientdetlite-d2',
            'architecture': {
                'backbone': 'efficientnetlite-b2',
            },
            'fpn': {
                'fpn_cell_repeats': 5,  # efd
                'fpn_feat_dims': 112},
            'efficientdet_parser': {
                'output_size': [768, 768],
            },
            'efficientdet_head': {
                'num_convs': 3,
                'num_filters': 112,
                'use_separable_conv': False,
            }
        },
    'efficientdet-d3':
        {
            "name": 'efficientdet-d3',
            'architecture': {
                'backbone': 'efficientnet-b3',
            },
            'fpn': {
                'fpn_cell_repeats': 6,  # efd
                'fpn_feat_dims': 160},
            'efficientdet_parser': {
                'output_size': [896, 896],
            },
            'efficientdet_head': {
                'num_convs': 4,
                'num_filters': 160,
                'use_separable_conv': False,
            }
        },
    'efficientdetlite-d3':
        {
            "name": 'efficientdetlite-d3',
            'architecture': {
                'backbone': 'efficientnetlite-b3',
            },
            'fpn': {
                'fpn_cell_repeats': 6,  # efd
                'fpn_feat_dims': 160},
            'efficientdet_parser': {
                'output_size': [896, 896],
            },
            'efficientdet_head': {
                'num_convs': 4,
                'num_filters': 160,
                'use_separable_conv': False,
            }
        },
    'efficientdet-d4':
        {
            "name": 'efficientdet-d4',
            'architecture': {
                'backbone': 'efficientnet-b4',
            },
            'fpn': {
                'fpn_cell_repeats': 7,  # efd
                'fpn_feat_dims': 224},
            'efficientdet_parser': {
                'output_size': [1024, 1024],
            },
            'efficientdet_head': {
                'num_convs': 4,
                'num_filters': 224,
                'use_separable_conv': False,
            }
        },
    'efficientdetlite-d4':
        {
            "name": 'efficientdetlite-d4',
            'architecture': {
                'backbone': 'efficientnetlite-b4',
            },
            'fpn': {
                'fpn_cell_repeats': 7,  # efd
                'fpn_feat_dims': 224},
            'efficientdet_parser': {
                'output_size': [1024, 1024],
            },
            'efficientdet_head': {
                'num_convs': 4,
                'num_filters': 224,
                'use_separable_conv': False,
            }
        },
    'efficientdet-d5':
        {
            "name": 'efficientdet-d5',
            'architecture': {
                'backbone': 'efficientnet-b5',
            },
            'fpn': {
                'fpn_cell_repeats': 7,  # efd
                'fpn_feat_dims': 288},
            'efficientdet_parser': {
                'output_size': [1280, 1280],
            },
            'efficientdet_head': {
                'num_convs': 4,
                'num_filters': 288,
                'use_separable_conv': False,
            }
        },
    'efficientdet-d6':
        {
            "name": 'efficientdet-d6',

            'architecture': {
                'backbone': 'efficientnet-b6',
            },
            'fpn': {
                'fpn_cell_repeats': 8,  # efd
                'fpn_feat_dims': 384,
                "fpn_name": "bifpn_sum"  # efd
            },
            'efficientdet_parser': {
                'output_size': [1280, 1280],
            },
            'efficientdet_head': {
                'num_convs': 5,
                'num_filters': 384,
                'use_separable_conv': False,
            }
        },
    'efficientdet-d7':
        {
            "name": 'efficientdet-d7',
            "anchor_scale": 5.0,
            'architecture': {
                'backbone': 'efficientnet-b7',
            },
            'fpn': {
                'fpn_cell_repeats': 8,  # efd
                'fpn_feat_dims': 384,
                "fpn_name": "bifpn_sum"  # efd
            },
            'efficientdet_parser': {
                'output_size': [1536, 1536],
            },
            'efficientdet_head': {
                'num_convs': 5,
                'num_filters': 384,
                'use_separable_conv': False,
            }
        }
}
# pylint: enable=line-too-long
