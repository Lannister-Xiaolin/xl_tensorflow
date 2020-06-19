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

"""Base efficientdet template."""

RESNET_FROZEN_VAR_PREFIX = r'(resnet\d+)\/(conv2d(|_([1-9]|10))|batch_normalization(|_([1-9]|10)))\/'
REGULARIZATION_VAR_REGEX = r'.*(kernel|weight):0$'

EFFICIENTDET_CFG = params_dict.ParamsDict({
    'model_dir': '',
    'use_tpu': False,
    'strategy_type': 'tpu',
    'isolate_session_state': False,
    'type': 'efficientdet',
    'train': {
        'iterations_per_loop': 100,
        'batch_size': 64,
        'total_steps': 22500,
        'num_cores_per_replica': None,
        'input_partition_dims': None,
        'optimizer': {
            'type': 'momentum',
            'momentum': 0.9,
            'nesterov': True,  # `False` is better for TPU v3-128.
        },
        'learning_rate': {
            'type': 'step',
            'warmup_learning_rate': 0.0067,
            'warmup_steps': 500,
            'init_learning_rate': 0.08,
            'learning_rate_levels': [0.008, 0.0008],
            'learning_rate_steps': [15000, 20000],
        },
        'checkpoint': {
            'path': '',
            'prefix': '',
        },
        # One can use 'RESNET_FROZEN_VAR_PREFIX' to speed up ResNet training
        # when loading from the checkpoint.
        'frozen_variable_prefix': '',
        'train_file_pattern': '',
        'train_dataset_type': 'tfrecord',
        'transpose_input': False,
        'regularization_variable_regex': REGULARIZATION_VAR_REGEX,
        'l2_weight_decay': 4e-5,
        'gradient_clip_norm': 0.0,
        'input_sharding': False,
    },
    'efficientdet_loss': {
        'focal_loss_alpha': 0.25,
        'focal_loss_gamma': 1.5,
        'huber_loss_delta': 0.1,
        'box_loss_weight': 50,
    },
    'eval': {
        'input_sharding': True,
        'batch_size': 8,
        'eval_samples': 5000,
        'min_eval_interval': 180,
        'eval_timeout': None,
        'num_steps_per_eval': 1000,
        'type': 'box',
        'use_json_file': True,
        'val_json_file': '',
        'eval_file_pattern': '',
        'eval_dataset_type': 'tfrecord',
        # When visualizing images, set evaluation batch size to 40 to avoid
        # potential OOM.
        'num_images_to_visualize': 0,
    },
    'predict': {
        'batch_size': 8,
    },
    'architecture': {
        'parser': 'efficientdet_parser',
        'min_level': 3,
        'max_level': 7,
        'multilevel_features': 'bifpn',
        'use_bfloat16': False,
        'num_classes': 91,
    },
    'anchor': {
        'num_scales': 3,
        'aspect_ratios': [1.0, 2.0, 0.5],
        'anchor_size': 4.0,
    },
    'norm_activation': {
        'activation': 'swish',
        'batch_norm_momentum': 0.99,
        'batch_norm_epsilon': 1e-3,
        'batch_norm_trainable': True,
        'use_sync_bn': False,
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
        'use_separable_conv': True,
    },
    'fpn': {
        'fpn_feat_dims': 256,
        'use_separable_conv': True,
        'use_batch_norm': True,
        'fpn_cell_repeats': 3,  # efd
        "apply_bn_for_resampling": True,  # efd
        "conv_after_downsample": False,  # efd
        "conv_bn_act_pattern": False,  # efd
        "use_native_resize_op": True,  # efd
        "pooling_type": None,  # efd
        "fpn_config": None,  # efd
        "fpn_weight_method": None,  # efd
        "fpn_name": None  # efd
    },
    'postprocess': {
        'use_batched_nms': False,
        'max_total_size': 100,
        'nms_iou_threshold': 0.5,
        'score_threshold': 0.05,
        'pre_nms_num_boxes': 5000,
    },
    'enable_summary': True,
    'data_format': 'channels_last',  # efd
    "is_training_bn": True,  # efd,全局
    "act_type": "swish"  # efd，全局
})
# pylint: enable=line-too-long

EFFICIENTDET_CFG.override({

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
                'use_separable_conv': True,
            }
        },
    'efficientdetlite-d0':
        {
            "name": 'efficientdetlite-d0',
            "act_type": "relu",  # efd，全局
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
                'use_separable_conv': True,
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
                'use_separable_conv': True,
            }
        },
    'efficientdetlite-d1':
        {
            "name": 'efficientdetlite-d1',
            "act_type": "relu",  # efd，全局
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
                'use_separable_conv': True,
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
                'use_separable_conv': True,
            }
        },
    'efficientdetlite-d2':
        {
            "name": 'efficientdetlite-d2',
            "act_type": "relu",  # efd，全局
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
                'use_separable_conv': True,
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
                'use_separable_conv': True,
            }
        },
    'efficientdetlite-d3':
        {
            "name": 'efficientdetlite-d3',
            "act_type": "relu",  # efd，全局
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
                'use_separable_conv': True,
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
                'use_separable_conv': True,
            }
        },
    'efficientdetlite-d4':
        {
            "name": 'efficientdetlite-d4',
            "act_type": "relu",  # efd，全局
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
                'use_separable_conv': True,
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
                'use_separable_conv': True,
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
                'use_separable_conv': True,
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
                'use_separable_conv': True,
            }
        }
}
# pylint: enable=line-too-long
