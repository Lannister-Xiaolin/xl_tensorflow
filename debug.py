#!usr/bin/env python3
# -*- coding: UTF-8 -*-
from xl_tensorflow.models.yolov3.model import tiny_yolo_body,yolo_body
from xl_tensorflow.utils.deploy import tf_saved_model_to_lite,serving_model_export
from tensorflow.keras import Input,Model
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D
num_anchors = 9
num_classes = 16
yolo_model = yolo_body(Input(shape=(224, 224, 3)), num_anchors // 3, num_classes)
x = GlobalAveragePooling2D()(yolo_model.outputs[0])
x = Dense(16, activation="softmax")(x)
test_model = Model(inputs= yolo_model.inputs,outputs=x)
print(test_model.summary())
serving_model_export(test_model,path=r"E:\Temp\test",version=5,auto_incre_version=False)
tf_saved_model_to_lite(r"E:\Temp\test\5",r"E:\Temp\test\float_model.tflite",input_shape=[1,224,224,3],)
# tf_saved_model_to_lite(r"E:\Temp\test\5",r"E:\Temp\test\int_quant_model.tflite",input_shape=[1,416,416,3],quantize_method="int")
