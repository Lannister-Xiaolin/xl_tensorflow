from xl_tensorflow.models.vision.classification.efficientnet import EfficientNetB0
from xl_tensorflow.utils.deploy import serving_model_export,tf_saved_model_to_lite
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2

model = EfficientNetB0(weights=None, input_shape=[512,512,3],classes=36,using_se_global_pooling=False)
# model.layers[0]._name ="image_tensor"
# model.layers[-1]._name = "fffff"
# print(model.summary())
# tf.saved_model.save(model, r"E:\Temp\test\1")
serving_model_export(model,r"E:\Temp\test\1")
tf_saved_model_to_lite(r"E:\Temp\test\1\2",r"E:\Temp\test\1.tflite")
# from xl_tensorflow.models.vision.classification.utils.tfrecord import images2tfrecord
# images2tfrecord(r"E:\Temp\test\image", r"E:\Temp\test\test.tfrecord", r"E:\Temp\test.json")