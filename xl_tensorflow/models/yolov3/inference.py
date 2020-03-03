import os
import numpy as np
from PIL import Image
from .utils import letterbox_image,draw_image
from .model import yolo_body, tiny_yolo_body, yolo_eval
from tensorflow.keras import Input
from tensorflow.keras.models import Model
import tensorflow as tf


def _get_anchors(anchors_path):
    anchors_path = os.path.expanduser(anchors_path)
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


anchors = _get_anchors("./model_data/yolo_anchors.txt")
num_anchors = 9


def inference_model(num_classes, num_anchors, image_shape):
    yolo_model = yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
    boxes_, scores_, classes_ = yolo_eval(yolo_model.outputs, anchors, 20, (416, 416))
    model = Model(inputs=yolo_model.inputs, outputs=(boxes_, scores_, classes_))
    return model


def predict_image(num_classes, image_file, input_size=(416, 416), ):
    # todo 计算公式中需要输入图片的尺寸不能直接用于部署，需要进行修改
    num_classes = num_classes
    image = Image.open(image_file)
    boxed_image = letterbox_image(image, input_size)
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    model = inference_model(num_classes, num_anchors,image.size)
    out_boxes, out_scores, out_classes = model.predict(image_data)
    if out_boxes.shape[1] == 0:
        print("未发现目标框！！")
        return
    else:
        draw_image()
