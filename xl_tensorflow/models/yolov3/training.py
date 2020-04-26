#!usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Retrain the YOLO model for your own dataset.
"""

import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from .model import yolo_body, tiny_yolo_body, yolo_loss, \
    yolo_efficientnetb3_body, yolo_efficientnetliteb4_body, yolo_efficientnetliteb1_body, yolo_efficientnetb0_body
from .utils import get_random_data, preprocess_true_boxes, get_anchors, get_classes, data_generator_wrapper


def _main(train_annotation_path, val_annotation_path, classes_path, anchors_path, weights_path):
    log_dir = './logs/000/'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (416, 416)  # multiple of 32, hw

    is_tiny_version = len(anchors) == 6  # default setting
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes,
                                  freeze_body=2, weights_path=weights_path)
    else:
        model = create_model(input_shape, anchors, num_classes,
                             freeze_body=2,
                             weights_path=weights_path)  # make sure you know what you freeze

    log = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    with open(train_annotation_path, encoding="utf-8") as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding="utf-8") as f:
        val_lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(train_lines)
    np.random.seed(None)
    num_val = int(len(train_lines))
    num_train = len(val_lines)
    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = 32
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(train_lines, batch_size, input_shape, anchors, num_classes),
                            steps_per_epoch=max(1, num_train // batch_size),
                            validation_data=data_generator_wrapper(val_lines, batch_size, input_shape, anchors,
                                                                   num_classes),
                            validation_steps=max(1, num_val // batch_size),
                            epochs=50,
                            initial_epoch=0,
                            callbacks=[log, checkpoint])
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4),
                      loss={'yolo_loss': lambda y_true, y_pred: y_pred})  # recompile to apply the change
        print('Unfreeze all of the layers.')

        batch_size = 32  # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(train_lines, batch_size, input_shape, anchors, num_classes),
                            steps_per_epoch=max(1, num_train // batch_size),
                            validation_data=data_generator_wrapper(val_lines, batch_size, input_shape, anchors,
                                                                   num_classes),
                            validation_steps=max(1, num_val // batch_size),
                            epochs=100,
                            initial_epoch=50,
                            callbacks=[log, checkpoint, reduce_lr, early_stopping])
        model.save_weights('./model/yolov3_trained_weights_final.h5')


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
                 weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [
        Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l], num_anchors // 3, num_classes + 5)) for
        l in range(3)]

    model_body = yolo_body(image_input, num_anchors // 3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers) - 3)[freeze_body - 1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
                      weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h // {0: 32, 1: 16}[l], w // {0: 32, 1: 16}[l],
                           num_anchors // 2, num_classes + 5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors // 2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers) - 2)[freeze_body - 1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


def create_datagenerator(train_annotation_path, val_annotation_path, batch_size, input_shape, anchors, num_classes,
                         seed=100):
    with open(train_annotation_path, encoding="utf-8") as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding="utf-8") as f:
        val_lines = f.readlines()
    num_train = int(len(train_lines))
    np.random.seed(seed)
    np.random.shuffle(train_lines)
    num_val = len(val_lines)
    train = data_generator_wrapper(train_lines, batch_size, input_shape, anchors, num_classes)
    val = data_generator_wrapper(val_lines, batch_size, input_shape, anchors,
                                 num_classes)
    return train, val, num_train, num_val


def create_callback(log_dir, checkpoint_dir, reduce_lr_patience=5, early_stopping_patience=10):
    log_dir = log_dir
    log = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(checkpoint_dir,
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=reduce_lr_patience, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience, verbose=1)
    return [checkpoint, reduce_lr, early_stopping]


body_dict = {
    "darknet": yolo_body,
    "efficientnetb0": yolo_efficientnetb0_body,
    "efficientnetb3": yolo_efficientnetb3_body,
    "efficientnetliteb1": yolo_efficientnetliteb1_body,
    "efficientnetliteb4": yolo_efficientnetliteb4_body
}


def mul_gpu_training(train_annotation_path, val_annotation_path, classes_path, batch_size=8,
                     input_shape=(416, 416), body="darknet", suffix="voc", pre_weights=None,giou_loss=False,mul_gpu=False):
    """
    Todo 加速训练
    Args:
        number_classes:
        input_shape:
        body:

    Returns:

    """
    import tensorflow as tf
    from xl_tensorflow.models.yolov3.model import YoloLoss
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    if mul_gpu:
        mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():
        image_input = Input(shape=(*input_shape, 3))
        model = body_dict[body](image_input, 3, num_classes, reshape_y=True)
        if pre_weights:
            model.load_weights(pre_weights, by_name=True, skip_mismatch=True)
        model.compile(loss=[YoloLoss(i, input_shape, num_classes, giou_loss=True) for i in range(3)])
    callback = create_callback(f"./log/yolo_{body}_{suffix}", f"./mdoel/yolo_{body}_{suffix}_weights.h5",
                               early_stopping_patience=20)
    # 创建训练数据
    train_dataset, val_dataset, num_train, num_val = create_datagenerator(train_annotation_path, val_annotation_path,
                                                                          batch_size, input_shape,
                                                                          YoloLoss.defalt_anchors, num_classes)
    with mirrored_strategy.scope():
        for i in range(185): model.layers[i].trainable = False
        model.compile(loss=[YoloLoss(i, input_shape, num_classes, giou_loss=giou_loss) for i in range(3)])
    model.fit(train_dataset, validation_data=val_dataset,
              epochs=50,
              steps_per_epoch=max(1, num_train // batch_size),
              validation_steps=max(1, num_val // batch_size),
              initial_epoch=0,
              callbacks=callback)
    callback = create_callback(f"./log/yolo_{body}_{suffix}", f"./mdoel/yolo_{body}_{suffix}_weights.h5")
    with mirrored_strategy.scope():
        for i in range(185): model.layers[i].trainable = True
        model.compile(loss=[YoloLoss(i, input_shape, num_classes, giou_loss=giou_loss) for i in range(3)])
    model.fit(train_dataset, validation_data=val_dataset,
              epochs=100,
              steps_per_epoch=max(1, num_train // batch_size),
              validation_steps=max(1, num_val // batch_size),
              initial_epoch=30,
              callbacks=callback)
    return model


if __name__ == '__main__':
    image_input = Input(shape=(380, 380, 3))
    # model_body = yolo_efficientnet_body(image_input, 3, 16)
