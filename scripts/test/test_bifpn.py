#!usr/bin/env python3
# -*- coding: UTF-8 -*-
from xl_tensorflow.models.vision import EfficientNetB1
from xl_tool.xl_io import read_json, save_to_json
import xl_tensorflow.models.vision.detection.configs.factory as config_factory
from xl_tensorflow.models.vision.detection.architecture.fpn import BiFpn
from xl_tensorflow.models.vision import EfficientNetB0
import numpy as np
from tensorflow.keras import layers,Model
inputs = layers.Input(shape=(512,512,3))
features = EfficientNetB0(input_tensor=inputs,fpn_features=True)
# print(features)
params = config_factory.config_generator("efficientdet-d1")
bifpn = BiFpn()
output_feats = bifpn(features,params)
model = Model(inputs, output_feats)
# print(model.summary())
# print(model(np.random.random((1,512,512,3))))
print(output_feats)
# print(params.as_dict())
save_to_json(params.as_dict(),
             r"F:\BaiduNetdiskDownload\test.json")
