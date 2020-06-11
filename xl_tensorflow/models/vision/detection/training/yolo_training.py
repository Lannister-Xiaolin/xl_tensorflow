#!usr/bin/env python3
# -*- coding: UTF-8 -*-
from tensorflow.keras import Input, Model
from ..body.yolo import yolo_body, yolo_eval
from xl_tensorflow.models.vision.detection.dataloader.utils.anchors_yolo import YOLOV4_ANCHORS, YOLOV3_ANCHORS
from ..loss.yolo_loss import YoloLoss
from ..dataloader.yolo_loader import get_classes, create_datagen
import tensorflow as tf
from xl_tensorflow.utils.common import nondistribute, xl_call_backs
from tensorflow.keras.optimizers import Adam
from xl_tensorflow.models.vision.detection.dataloader import YoloInputFn
from xl_tool.xl_io import read_json


# def yolo_inferece():


def mul_gpu_training_custom_data(train_annotation_path, val_annotation_path,
                                 classes_path="", batch_size=8, iou_loss="giou",
                                 input_shape=(416, 416), num_classes=None,
                                 architecture="yolov3",
                                 suffix="voc", pre_weights=None, anchors="v3",
                                 use_multiprocessing=False, workers=1,
                                 tfrecord=True,
                                 lrs=(1e-4, 5e-4, 5e-5),
                                 freeze_layers=(185, 0, 0),
                                 epochs=(20, 40, 60), initial_epoch=0,
                                 paciences=(10, 10, 8),
                                 reduce_lrs=(5, 5, 5), trunc_inf=True,
                                 ignore_thresh=0.4, print_loss=True,
                                 iou_scale=1.0,
                                 autoaugment_policy_name="v0", autoaugment_ratio=0.8, buffer=1000,
                                 aug_scale_max=1.5, aug_scale_min=0.5):
    """

    Args:
        train_annotation_path: file pattern for tf record or train.txt
        val_annotation_path:
        classes_path:
        batch_size:
        iou_loss:
        input_shape:
        num_classes:
        architecture:
        suffix:
        pre_weights:
        anchors:
        use_multiprocessing:
        workers:
        tfrecord:
        lrs:
        freeze_layers:
            185 v3
            200 liteb1/b2
            332 b1
            250 v4
        epochs:
        initial_epoch:
        paciences:
        reduce_lrs:
        trunc_inf:
        ignore_thresh:
        print_loss:
        iou_scale:
        autoaugment_policy_name: v0 v1 v2 v3 v4
        autoaugment_ratio: ratio of shift and rotate, base of 50 for box cut  250 for trainslate

    Returns:

    """
    if not tfrecord:
        class_names = get_classes(classes_path)
        num_classes = len(class_names)
    else:
        num_classes = int(num_classes)
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
            try:
                model.load_weights(pre_weights)
            except:
                print("逐层加载预训练权重")
                model2 = yolo_body(image_input, 3, 80, architecture=architecture, reshape_y=True)
                try:
                    model2.load_weights(pre_weights)
                except ValueError:
                    model2.load_weights(pre_weights, by_name=True)
                for i in range(len(model2.layers)):
                    try:
                        model.layers[i].set_weights(model2.layers[i].get_weights())
                    except Exception as e:
                        print(e)

    # 创建训练数据
    if not tfrecord:
        train_dataset, val_dataset, num_train, num_val = create_datagen(train_annotation_path, val_annotation_path,
                                                                        batch_size, input_shape,
                                                                        anchors, num_classes)
    else:
        train_dataset = YoloInputFn(input_shape, train_annotation_path,
                                    num_classes, aug_scale_max=aug_scale_max, aug_scale_min=aug_scale_min,
                                    use_autoaugment=True, aug_rand_hflip=True,
                                    autoaugment_policy_name=autoaugment_policy_name, anchor=anchors,
                                    autoaugment_ratio=autoaugment_ratio, buffer=buffer)(batch_size=batch_size)
        val_dataset = YoloInputFn(input_shape, val_annotation_path,
                                  num_classes, anchor=anchors, aug_scale_max=1.0, aug_scale_min=1.0,
                                  use_autoaugment=False, aug_rand_hflip=False)(
            batch_size=batch_size)
    for i in range(len(lrs)):
        if epochs[i] <= initial_epoch: continue
        with mirrored_strategy.scope():
            if freeze_layers[i] > 0:
                for j in range(freeze_layers[i]):
                    model.layers[j].trainable = False
            else:
                print("___________make all layers trainable________________")
                for j in range(len(model.layers)):
                    model.layers[j].trainable = True
            model.compile(Adam(lrs[i]),
                          loss=[YoloLoss(i, input_shape, num_classes, iou_loss=iou_loss, trunc_inf=trunc_inf,
                                         name=f"state_{i}", ignore_thresh=ignore_thresh,
                                         print_loss=print_loss, iou_scale=iou_scale) for i in
                                range(3)])

        callback = xl_call_backs(architecture, log_path=f"./logs/{architecture}_{suffix}",
                                 model_path=f"./model/{architecture}_{suffix}",
                                 save_best_only=False, patience=paciences[i], reduce_lr=reduce_lrs[i])
        if not tfrecord:
            model.fit(train_dataset, validation_data=val_dataset,
                      epochs=epochs[i],
                      steps_per_epoch=max(1, num_train // batch_size),
                      validation_steps=max(1, num_val // batch_size),
                      initial_epoch=initial_epoch,
                      callbacks=callback, use_multiprocessing=use_multiprocessing, workers=workers)
        else:
            model.fit(train_dataset, validation_data=val_dataset,
                      epochs=epochs[i],
                      initial_epoch=initial_epoch,
                      callbacks=callback)
        initial_epoch = epochs[i]
    return model
