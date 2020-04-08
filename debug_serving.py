from xl_tensorflow.utils.deploy import serving_model_export
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2

model = MobileNetV2()
model.layers[0]._name ="image_tensor"
model.layers[-1]._name = "fffff"
model = tf.keras.Model(model.inputs,model.outputs)
print(model.summary())
tf.saved_model.save(model, r"E:\Temp\test\1", )