#!usr/bin/env python3
# -*- coding: UTF-8 -*-
from xl_tensorflow.utils.deploy import serving_model_export,tf_saved_model_to_lite
import tensorflow as tf
from tensorflow.keras.layers import Input,Conv2D,UpSampling2D
x1 = Input(shape=(224,224,3))
x = Conv2D(3,3,strides=(2,2),padding="same")(x1)
x = UpSampling2D(interpolation="bilinear")(x)
model = tf.keras.Model(x1,x)

serving_model_export(model,r"E:\Temp\test\temp_saved")
tf_saved_model_to_lite(r"E:\Temp\test\temp_saved\3",r"E:\Temp\test\temp_saved\upsampe.tflite")


