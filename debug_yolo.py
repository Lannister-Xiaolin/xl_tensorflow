#!usr/bin/env python3
# -*- coding: UTF-8 -*-
# from xl_tensorflow.models.yolov3.model import tiny_yolo_body,yolo_body
# from xl_tensorflow.utils.deploy import tf_saved_model_to_lite,serving_model_export
# from tensorflow.keras import Input,Model
# from tensorflow.keras.layers import Dense,GlobalAveragePooling2D
# num_anchors = 9
# num_classes = 16
# yolo_model = yolo_body(Input(shape=(224, 224, 3)), num_anchors // 3, num_classes)
# x = GlobalAveragePooling2D()(yolo_model.outputs[0])
# x = Dense(16, activation="softmax")(x)
# test_model = Model(inputs= yolo_model.inputs,outputs=x)
# print(test_model.summary())
# serving_model_export(test_model,path=r"E:\Temp\test",version=5,auto_incre_version=False)
# tf_saved_model_to_lite(r"E:\Temp\test\5",r"E:\Temp\test\float_model.tflite",input_shape=[1,224,224,3],)
# tf_saved_model_to_lite(r"E:\Temp\test\5",r"E:\Temp\test\int_quant_model.tflite",input_shape=[1,416,416,3],quantize_method="int")
import tensorflow as tf

# tf.compat.v1.disable_eager_execution()
from xl_tensorflow.models.yolov3.model import *
from xl_tensorflow.models.yolov3.utils import *
from tensorflow.keras.layers import Input, Lambda
from xl_tensorflow.models.yolov3.inference import tf_saved_model_to_lite


def test_yolo_tflosss():
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        # defalt_anchors = np.array([[10., 13.],
        #                            [16., 30.],
        #                            [33., 23.],
        #                            [30., 61.],
        #                            [62., 45.],
        #                            [59., 119.],
        #                            [116., 90.],
        #                            [156., 198.],
        #                            [373., 326.]], dtype="float")
        image_input = Input(shape=(416, 416, 3))
        # model_body = yolo_body(image_input, 3, 15, False)
        # y1 = tf.reshape(model_body.outputs[0], (-1, 13, 13, 3, 20))
        # y2 = tf.reshape(model_body.outputs[1], (-1, 26, 26, 3, 20))
        # y3 = tf.reshape(model_body.outputs[2], (-1, 52, 52, 3, 20))
        # model = Model(model_body.inputs, [y1, y2, y3])
        model = yolo_body(image_input, 3, 15, True)
        model.compile(loss=[YoloLoss(i, (416, 416), 15, giou_loss=True) for i in range(3)])
    num_classes = 15
    anchors = get_anchors(r"E:\Programming\Python\5_CV\学习案例\xl_tf2_yolov3\model_data\yolo_anchors.txt")
    input_shape = (416, 416)  # multiple of 32, hw
    with open(r"E:\Programming\Python\5_CV\学习案例\xl_tf2_yolov3\model_data\train.txt", encoding="utf-8") as f:
        train_lines = f.readlines()
    train_gen = data_generator_wrapper(train_lines, 4, input_shape, anchors, num_classes)
    for i in range(10):
        image, labels = next(train_gen)
        # image, labels = inputs[0], inputs[1:]
        model.fit(image, labels)


def tflite():
    tf_saved_model_to_lite(r"E:\Temp\test\yolo\2", r"E:\Temp\test\yolo.tflite",
                           input_shape=[None, 416, 416, 3], allow_custom_ops=True)
def compare():
    from tensorflow.keras.models import load_model
    from xl_tensorflow.layers.actication import Mish
    model_darknet = load_model(r"E:\Programming\Python\TOOL\weights\yolo\yolov4.h5",{"Mish":Mish})
    model_my = yolo_body(Input(shape=(608, 608, 3)), 3, 80, architecture="yolov4", reshape_y=False)
    print(len(model_my.layers), len(model_darknet.layers))
    print(model_my.outputs, model_darknet.outputs)
    model_my.save(r"E:\Programming\Python\TOOL\packege_xl_tf\scripts\yolov4_my.h5")
    model_darknet.save(r"E:\Programming\Python\TOOL\packege_xl_tf\scripts\yolov4_darknet.h5")
    for i in range(len(model_darknet.layers)):
        try:
            if model_my.layers[i].trainable_weights[0].shape == model_darknet.layers[i].trainable_weights[0].shape:
                continue
            else:
                print(i,"my: ",model_my.layers[i]._name,model_my.layers[i].trainable_weights[0].shape, "\tdarknet:", model_darknet.layers[i]._name,model_darknet.layers[i].trainable_weights[0].shape,si)
        except IndexError:
            continue
    model_my.load_weights(r"E:\Programming\Python\TOOL\weights\yolo\yolov4_weights.h5")
    model_my.save_weights(r"E:\Programming\Python\TOOL\weights\yolo\yolov4_xl_weights.h5")
    print( yolo_body(Input(shape=(608, 608, 3)), 3, 20, architecture="yolov4", reshape_y=False).load_weights(r"E:\Programming\Python\TOOL\weights\yolo\yolov4_weights.h5",skip_mismatch=True,by_name=True))
    print( yolo_body(Input(shape=(416, 416, 3)), 3, 20, architecture="yolov4", reshape_y=False).load_weights(r"E:\Programming\Python\TOOL\weights\yolo\yolov4_xl_weights.h5",skip_mismatch=True,by_name=True))

def compare_yolov3():
    from tensorflow.keras.models import load_model
    from xl_tensorflow.layers.actication import Mish
    model_darknet = load_model(r"E:\Programming\Python\TOOL\packege_xl_tf\scripts\yolov3.h5",{"Mish":Mish})
    model_my = yolo_body(Input(shape=(608, 608, 3)), 3, 80, architecture="yolov3", reshape_y=False)
    print(len(model_my.layers), len(model_darknet.layers))
    print(model_my.outputs, model_darknet.outputs)
    model_my.save(r"E:\Programming\Python\TOOL\packege_xl_tf\scripts\yolov4_my.h5")
    model_darknet.save(r"E:\Programming\Python\TOOL\packege_xl_tf\scripts\yolov4_darknet.h5")
    for i in range(len(model_darknet.layers)):
        try:
            if model_my.layers[i].trainable_weights[0].shape == model_darknet.layers[i].trainable_weights[0].shape:
                continue
            else:
                print(i,"my: ",model_my.layers[i]._name,model_my.layers[i].trainable_weights[0].shape, "\tdarknet:", model_darknet.layers[i]._name,model_darknet.layers[i].trainable_weights[0].shape)
        except IndexError:
            continue
    model_my.load_weights(r"E:\Programming\Python\TOOL\packege_xl_tf\scripts\yolov3_weights.h5")
    model_my.save_weights(r"E:\Programming\Python\TOOL\packege_xl_tf\scripts\yolov3_xl_weights.h5")
if __name__ == '__main__':
    # test_yolo_tflosss()
    # tflite()
    # from xl_tensorflow.models.vision.detection.configs.yolo import get_yolo_config
    # print(get_yolo_config())
    # from xl_tensorflow.models.vision.classification.darknet import DarkNet53,CspDarkNet53
    # model = CspDarkNet53(input_shape=(608,608,3),weights=None)
    # model.save(r"E:\Temp\test\fuck.h5")
    from tensorflow.keras import Input, Model, layers
    from xl_tensorflow.models.vision.detection.body.yolo import yolo_body
    from xl_tensorflow.models.yolov3.training import yolo_body as yolo_body_3

    # image_input = Input(shape=(608, 608, 3))
    # model_body = yolo_body_3(image_input, 3, 35, True)

    # model = yolo_body(Input(shape=(608, 608, 3)), 3, 80, architecture="yolov4", reshape_y=False)
    # model.load_weights(r"E:\Programming\Python\TOOL\packege_xl_tf\scripts\yolov4_weights.h5")
    # print(model.summary())
    # # print(model.get_layer("mish_37"))
    # model.save(r"E:\Programming\Python\TOOL\packege_xl_tf\scripts\yolov4_custom2.h5")
    # compare_yolov3()
    compare()
    # from xl_tensorflow.models.vision.classification.darknet import CspDarkNet53
    # print(len(CspDarkNet53(include_top=False,weights=None).layers))