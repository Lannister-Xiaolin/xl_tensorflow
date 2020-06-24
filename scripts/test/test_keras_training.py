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
from xl_tensorflow.models.vision.detection.dataloader.input_reader import InputFn
params = config_factory.config_generator("efficientdet-d1")
params.architecture.num_classes = 21
model_fn = EfficientDetModel(params)
model,inference_model,lite_model = model_fn.build_model_keras(params)
loss_fn = model_fn.build_loss_fn_keras()
model.compile(loss=loss_fn)