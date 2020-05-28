#!usr/bin/env python3
# -*- coding: UTF-8 -*-
from xl_tensorflow.models.vision import EfficientNetB1
from xl_tool.xl_io import read_json, save_to_json
import xl_tensorflow.models.vision.detection.configs.factory as config_factory
from xl_tensorflow.models.vision.detection.architecture.fpn import BiFpn

params = config_factory.config_generator("efficientdet-d0")
print(params.as_dict())
save_to_json(params.as_dict(),
             r"F:\BaiduNetdiskDownload\test.json")
