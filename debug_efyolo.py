
# from xl_tensorflow.models.yolov3.model import tiny_yolo_body,yolo_body
from xl_tensorflow.utils.deploy import tf_saved_model_to_lite,serving_model_export
from tensorflow.keras import Input,Model
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D
# pron
from xl_tensorflow.models.yolov3.training import *

image_input = Input(shape=(320, 320, 3))
model_body = yolo_efficientnetb3_body(image_input, 3, 16)
# input()
x = GlobalAveragePooling2D()(model_body.outputs[0])
x = Dense(16, activation="softmax")(x)
# print()
test_model = Model(inputs= model_body.inputs,outputs=x)
print(test_model.summary())
serving_model_export(test_model,path=r"E:\Temp\test",version=5,auto_incre_version=False)
tf_saved_model_to_lite(r"E:\Temp\test\5",r"E:\Temp\test\float_model.tflite",input_shape=[1,320,320,3],)
# tf_saved_model_to_lite(r"E:\Temp\test\5",r"E:\Temp\test\int_quant_model.tflite",input_shape=[1,416,416,3],quantize_method="int")
