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
"""Factory to provide model configs."""

from . import maskrcnn_config
from . import retinanet_config
from . import efficientdet_config
from xl_tensorflow.utils import params_dict


def config_generator(model):
    """Model function generator."""
    if model == 'retinanet':
        default_config = retinanet_config.RETINANET_CFG
        restrictions = retinanet_config.RETINANET_RESTRICTIONS
    elif model == 'mask_rcnn':
        default_config = maskrcnn_config.MASKRCNN_CFG
        restrictions = maskrcnn_config.MASKRCNN_RESTRICTIONS
    elif "efficientdet" in model:
        default_config = efficientdet_config.EFFICIENTDET_CFG
        default_config.override(efficientdet_config.efficientdet_model_param_dict[model], is_strict=False)
        restrictions = efficientdet_config.EFFICIENTDET_RESTRICTIONS
        pass
    else:
        raise ValueError('Model %s is not supported.' % model)

    return params_dict.ParamsDict(default_config, restrictions)