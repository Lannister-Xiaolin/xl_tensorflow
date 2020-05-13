#!usr/bin/env python3
# -*- coding: UTF-8 -*-
from tensorflow.keras import Input, Model
from ..body.yolo import yolo_body, yolo_eval
from ..configs.anchors import YOLOV4_ANCHORS, YOLOV3_ANCHORS
from ..loss.yolo_loss import YoloLoss
from ..dataloader.yolo_loader import get_classes, create_datagen
import tensorflow as tf
from xl_tensorflow.utils.common import nondistribute, xl_call_backs
from tensorflow.keras.optimizers import Adam


# TODO base64输入，预处理，resize和padding处理， lite版本
def single_inference_model_serving(model_name, weights,
                                   num_classes,
                                   origin_image_shape=(416, 416),
                                   input_shape=(416, 416),
                                   anchors="v3",
                                   score_threshold=.1,
                                   iou_threshold=.5,
                                   max_detections=20,
                                   dynamic_shape=False, return_xy=True):
    """
    用于部署在serving端的模型，固定输入尺寸和图片尺寸，会对iou值和置信度进行过滤0.1
    Args:
        model_name: string must be of of following:
                    "yolov3 yolov4 yolov3-spp yolov4-efficientnetb0"
        origin_image_shape: 高*宽
        weights:
        num_classes:
        dynamic_shape：是否允许将图片尺寸作为动态输入，
        return_xy:是否范围xy格式，默认yx格式
    Returns:
        tf.keras.Model object, 预测图片的绝对值坐标x1,y1,x2,y2
    """
    anchors = YOLOV4_ANCHORS if anchors == "v4" else YOLOV3_ANCHORS
    yolo_model = yolo_body(Input(shape=(*input_shape, 3)),
                           len(anchors) // 3, num_classes, model_name)

    if weights:
        yolo_model.load_weights(weights)
    if dynamic_shape:
        shape_input = Input(shape=(2,))
        boxes_, scores_, classes_ = yolo_eval(yolo_model.outputs,
                                              anchors, num_classes, shape_input, max_detections,
                                              score_threshold,
                                              iou_threshold, return_xy=return_xy)
        model = Model(inputs=yolo_model.inputs + [shape_input], outputs=(boxes_, scores_, classes_))
    else:
        boxes_, scores_, classes_ = yolo_eval(yolo_model.outputs,
                                              anchors, num_classes, origin_image_shape, max_detections,
                                              score_threshold,
                                              iou_threshold, return_xy=return_xy)
        model = Model(inputs=yolo_model.inputs, outputs=(boxes_, scores_, classes_))
    return model


def mul_gpu_training_custom_data(train_annotation_path, val_annotation_path,
                                 classes_path, batch_size=8, iou_loss="",
                                 input_shape=(416, 416), architecture="yolov3",
                                 suffix="voc", pre_weights=None, anchors="v3",
                                 use_multiprocessing=True, workers=4, skip_mismatch=False,
                                 tfrecord=False, generater2tfdata=True,
                                 lrs=(1e-4, 1e-5),
                                 freeze_layers=(185, 0),
                                 epochs=(20, 30), initial_epoch=0,
                                 paciences=(10, 5),
                                 reduce_lrs=(3, 3)):
    """
    Todo 加速训练
    Args:
        number_classes:
        input_shape:
        body:
        freeze_layers: 185-yolov3  250-yolov4
    Returns:

    """
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    mirrored_strategy = tf.distribute.MirroredStrategy() if len(gpus) > 1 else nondistribute()
    anchors = YOLOV3_ANCHORS if anchors == "v3" else YOLOV4_ANCHORS
    with mirrored_strategy.scope():
        image_input = Input(shape=(*input_shape, 3))
        model = yolo_body(image_input, 3, num_classes, architecture=architecture, reshape_y=True)
        if pre_weights:
            model.load_weights(pre_weights, by_name=skip_mismatch, skip_mismatch=skip_mismatch)

    # 创建训练数据
    train_dataset, val_dataset, num_train, num_val = create_datagen(train_annotation_path, val_annotation_path,
                                                                    batch_size, input_shape,
                                                                    anchors, num_classes)
    for i in range(len(lrs)):
        with mirrored_strategy.scope():
            if freeze_layers[i] > 0:
                for j in range(freeze_layers[i]):
                    model.layers[j].trainable = False
            else:
                for j in range(len(model.layers)):
                    model.layers[j].trainable = True
            model.compile(Adam(lrs[i]),
                          loss=[YoloLoss(i, input_shape, num_classes, iou_loss=iou_loss) for i in
                                range(3)])
        callback = xl_call_backs(f"./logs/{architecture}_{suffix}",
                                 f"./model/{architecture}_{suffix}",
                                 save_best_only=False, patience=paciences[i], reduce_lr=reduce_lrs[i])

        model.fit(train_dataset, validation_data=val_dataset,
                  epochs=epochs[i],
                  steps_per_epoch=max(1, num_train // batch_size),
                  validation_steps=max(1, num_val // batch_size),
                  initial_epoch=initial_epoch if i == 0 else epochs[i - 1],
                  callbacks=callback, use_multiprocessing=use_multiprocessing, workers=workers)
    return model
