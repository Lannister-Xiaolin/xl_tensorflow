from xl_tensorflow.models.vision.classification.efficientnet import EfficientNetB0
from xl_tensorflow.utils.deploy import serving_model_export,tf_saved_model_to_lite
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3

model = InceptionV3(weights=None, input_shape=[299,299,3],classes=1000)
# model.layers[0]._name ="image_tensor"
# model.layers[-1]._name = "fffff"
# print(model.summary())
# tf.saved_model.save(model, r"E:\Temp\test\1")
serving_model_export(model,r"E:\Temp\test\incepcion")
tf_saved_model_to_lite(r"E:\Temp\test\incepcion\2",
                       r"E:\Temp\test\incepcion\InceptionV3.tflite".lower())
with open(r"E:\Temp\test\incepcion\labels1000.txt", "w") as f:
    f.write("\n".join([str(i) for i in range(1000)]))
# from xl_tensorflow.models.vision.classification.utils.tfrecord import images2tfrecord
# images2tfrecord(r"E:\Temp\test\image", r"E:\Temp\test\test.tfrecord", r"E:\Temp\test.json")