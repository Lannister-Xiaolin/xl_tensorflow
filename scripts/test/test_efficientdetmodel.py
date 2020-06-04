#!usr/bin/env python3
# -*- coding: UTF-8 -*-
from xl_tensorflow.models.vision import EfficientNetB1
from xl_tool.xl_io import read_json, save_to_json
import xl_tensorflow.models.vision.detection.configs.factory as config_factory
from xl_tensorflow.models.vision.detection.architecture.fpn import BiFpn
from xl_tensorflow.models.vision import EfficientNetB0
from xl_tensorflow.models.vision.detection.body.efficientdet_model import EfficientDetModel
import numpy as np
from tensorflow.keras import layers, Model

params = config_factory.config_generator("efficientdet-d1")

model_fn = EfficientDetModel(params)
model = model_fn.build_model(params)
model.load_weights(
    r"E:\Programming\Python\TOOL\weights\efficientnet\efficientnet-b1_weights_tf_dim_ordering_tf_kernels.h5",
    by_name=True, skip_mismatch=True)
data = np.random.random((2, 640, 640, 3))
print(model(data, training=True)['cls_outputs'])
print(model(data, training=False)['cls_outputs'])

# print(model_f)
