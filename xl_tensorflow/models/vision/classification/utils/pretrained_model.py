#!usr/bin/env python3
# -*- coding: UTF-8 -*-

import os
import numpy as np
import sys
from tensorflow.keras.applications import ResNet50, ResNet50V2, InceptionV3, DenseNet169, \
    InceptionResNetV2, MobileNet, ResNet101, NASNetMobile, VGG16, MobileNetV2
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Conv2D, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from xl_tensorflow.models.vision.classification.efficientnet import EfficientNetB0, EfficientNetB1, \
    EfficientNetB2, EfficientNetB3, EfficientNetB4, \
    EfficientNetB5, EfficientNetB6, EfficientNetB7, EfficientNetLiteB1, EfficientNetLiteB2, EfficientNetLiteB3, \
    EfficientNetLiteB4, EfficientNetLiteB0
from xl_tensorflow.models.vision.classification.mobilenet_v3 import MobileNetV3Small, MobileNetV3Large

eff_input_dict = {'efficientnetb0': 224, 'efficientnetb1': 240,
                  'efficientnetb2': 260,
                  'efficientnetb3': 300,
                  'efficientnetb4': 380,
                  'efficientnetb5': 456,
                  'efficientnetb6': 528,
                  'efficientnetb7': 600,
                  'efficientnetliteb1': 240,
                  'efficientnetliteb2': 260,
                  'efficientnetliteb3': 280,
                  'efficientnetliteb4': 300}


def my_call_backs(model_name, log_path=None, model_path=None, monitor="val_loss", patience=5,
                  reducelr=3, factor=0.2, update_freq="epoch"):
    """回调函数列表，包括tensorboard, 学习率衰减, 提前终止，模型检测点"""
    if "win" in sys.platform:
        log_dir = os.path.join(os.getcwd(), r"\logs\{}".format(model_name)) if not log_path else os.path.join(log_path,
                                                                                                              model_name)
    else:
        log_dir = r"./logs/{}".format(model_name) if not model_path else os.path.join(model_path, model_name)
    model_path = "./model"
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    tensorboard = TensorBoard(log_dir=log_dir, write_graph=False, histogram_freq=False, update_freq=update_freq)
    reducelr = ReduceLROnPlateau(monitor=monitor, factor=factor, patience=reducelr)
    early_stop = EarlyStopping(monitor=monitor, min_delta=1e-7, patience=patience, verbose=0, mode='auto',
                               baseline=None)
    model_check_point = ModelCheckpoint("./model/{}_weight.h5".format(model_name), verbose=1, save_best_only=True,
                                        save_weights_only=True)
    return [tensorboard, reducelr, early_stop, model_check_point]


class ImageFineModel:
    model_dict = {
        "vgg16": VGG16,
        "resnet50": ResNet50,
        "resnet50v2": ResNet50V2,
        "resnet101": ResNet101,
        "incepitonresnetv2": InceptionResNetV2,
        "densenet169": DenseNet169,
        "incepitionv3": InceptionV3,
        "mobilenet": MobileNet,
        "mobilenetv2": MobileNetV2,
        'nasnetmobile': NASNetMobile,
        'efficientnetb0': EfficientNetB0,
        'efficientnetb1': EfficientNetB1,
        'efficientnetb2': EfficientNetB2,
        'efficientnetb3': EfficientNetB3,
        'efficientnetliteb0': EfficientNetLiteB0,
        'efficientnetliteb1': EfficientNetLiteB1,
        'efficientnetliteb2': EfficientNetLiteB2,
        'efficientnetliteb3': EfficientNetLiteB3,
        'efficientnetliteb4': EfficientNetLiteB4,
        'efficientnetb4': EfficientNetB4,
        'efficientnetb5': EfficientNetB5,
        'efficientnetb6': EfficientNetB6,
        'efficientnetb7': EfficientNetB7,
        "mobilenetv3large": MobileNetV3Large,
        "mobilenetv3small": MobileNetV3Small,

    }
    weight = {
        "resnet50": "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",
        "incepitonresnetv2": "inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5",
        "densenet169": "densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5",
        "incepitionv3": "inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5",
        "mobilenetv2": "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5",
        "efficientnetb0": None

    }

    @classmethod
    def create_fine_model(cls, pre_model_name, cat_num, suffix="", prefix="my_", weights="imagenet",
                          non_flatten_trainable=False, loss="categorical_crossentropy", using_se_global_pooling=False,
                          metrics=("accuracy",), learning_rate=0.0005, input_shape=None, dropout=None,
                          activation=None):
        if pre_model_name in ["mobilenetv3large", "mobilenetv3small"]:
            model = cls.model_dict[pre_model_name](include_top=True, weights=None, classes=cat_num,
                                                   input_shape=input_shape) if input_shape else cls.model_dict[
                pre_model_name](include_top=True, weights=None, classes=cat_num)
            model.layers[0]._name = "images_tensor"
            model.layers[-1]._name = "output_tensor"
            model._name = prefix + pre_model_name + suffix
            model.compile(Adam(0.001), loss=loss, metrics=list(metrics))
        else:
            if pre_model_name in ["efficientnetb{}".format(i) for i in range(8)]:
                pre_model = cls.model_dict[pre_model_name](include_top=False, weights=weights, activation=activation,
                                                           using_se_global_pooling=using_se_global_pooling,
                                                           input_shape=(eff_input_dict[pre_model_name],
                                                                        eff_input_dict[pre_model_name], 3))
            else:
                pre_model = cls.model_dict[pre_model_name](include_top=False, weights=weights)
            pre_model.trainable = non_flatten_trainable
            pre_model.layers[0]._name = "images_tensor"
            x = GlobalAveragePooling2D()(pre_model.output)
            if dropout:
                x = Dropout(dropout, name="top_dropout")(x)
            x = Dense(cat_num, activation="softmax", name="output_tensor")(x)
            model = Model(inputs=pre_model.input, outputs=[x],
                          name=prefix + pre_model_name + suffix)
            model.compile(Adam(learning_rate), loss=loss, metrics=list(metrics))
        return model


def test_efficientnet():
    model = ImageFineModel().create_fine_model("efficientnetb0", 2)
    print(model.summary())
    model.fit(np.random.rand(1, 224, 224, 3), np.array([[0, 1]]))


def test_mobilenetv3large():
    model = ImageFineModel().create_fine_model("mobilenetv3large", 2)
    print(model.summary())
    model.fit(np.random.rand(1, 224, 224, 3), np.array([[0, 1]]))
