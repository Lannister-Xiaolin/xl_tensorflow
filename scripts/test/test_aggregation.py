#!usr/bin/env python3
# -*- coding: UTF-8 -*-
from xl_tensorflow.models.vision.detection.configs.yolo_config import get_yolo_config
from tensorflow.keras import layers,Model
from xl_tensorflow.models.vision.detection.body.aggregation import pan_network,fpn_network
p7 = layers.Input(shape=(19,19,512))
p6 = layers.Input(shape=(38,38,256))
p5 = layers.Input(shape=(76,76,128))
configs = get_yolo_config("yolov3")
outputs = fpn_network([p5,p6,p7],configs)
model = Model([p5,p6,p7], outputs)
model.save(r"E:\Temp\test\fuck.h5")
print(model.summary())
