#!usr/bin/env python3
# -*- coding: UTF-8 -*-
from xl_tensorflow.models.vision.detection.inference.yolo_inference import tflite_export_yolo

root = r"E:\Temp\test\yolo"
model = tflite_export_yolo("yolov3_efficientnetliteb1",20,
                           r"E:\Temp\test\yolo\yolov3_int.tflite",quant="float16",
                           weights=root + r"/050_val_22.626_train_18.307_yolov3_efficientnetliteb1_weights.h5")

import numpy as np
from PIL import Image
import tensorflow as tf
image = np.array(Image.open(root+"/"+"003590.jpg").resize((416,416)))*1.0
result = model.predict(np.expand_dims(image,0))
print([i.shape for i in result])
# boxes = tf.image.non_max_suppression(result[0][0],result[1][0],100000,score_threshold=0.2)
# print(result[2][0][list(np.array(boxes))],result[1][0][list(np.array(boxes))],"\n", result[0][0][list(np.array(boxes))])
# print(np.sum(result[2][0]>0.2))
# print(np.sum(result[1][0]>0.2))

