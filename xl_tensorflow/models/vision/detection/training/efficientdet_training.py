#!usr/bin/env python3
# -*- coding: UTF-8 -*-
import logging

import tensorflow as tf
from xl_tensorflow.models.vision import EfficientNetB1
from xl_tool.xl_io import read_json, save_to_json
import xl_tensorflow.models.vision.detection.configs.factory as config_factory
from xl_tensorflow.models.vision.detection.architecture.fpn import BiFpn
from xl_tensorflow.models.vision import EfficientNetB0
from xl_tensorflow.models.vision.detection.body.efficientdet_model import EfficientDetModel
import numpy as np
from tensorflow.keras import layers, Model
from xl_tensorflow.models.vision.detection.dataloader import input_reader
from xl_tensorflow.models.vision.detection.inference.efficientdet_inference import det_post_process_combined, \
    batch_image_preprocess
from xl_tensorflow.models.vision.detection.training.xl_detection_executor import DetectionDistributedExecutor
from absl import flags, app

flags.DEFINE_integer('save_checkpoint_freq', None,
                     'Number of steps to save checkpoint.')
FLAGS = flags.FLAGS


def mul_gpu_training_custom_data(model_name, training_file_pattern, eval_file_pattern, number_classes,
                                 pre_weights=None, mode="train"):
    # todo 不兼容2.2.0， tf.OneDeviceStrategy has no attribute run
    params = config_factory.config_generator(model_name)
    params.architecture.num_classes = number_classes
    params.train.batch_size = 4
    # 设置分布式训练策略
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) == 0:
        strategy = tf.distribute.OneDeviceStrategy("device:CPU:0")
    elif len(gpus) == 1:
        strategy = tf.distribute.OneDeviceStrategy("device:GPU:0")
    else:
        strategy = tf.distribute.MirroredStrategy()
    # 建立模型与数据加载
    model_builder = EfficientDetModel(params)
    # if training_file_pattern:
    # Use global batch size for single host.
    train_input_fn = input_reader.InputFn(
        file_pattern=training_file_pattern,
        params=params,
        mode=input_reader.ModeKeys.PREDICT_WITH_GT,
        batch_size=params.train.batch_size)
    if eval_file_pattern:
        eval_input_fn = input_reader.InputFn(
            file_pattern=eval_file_pattern,
            params=params,
            mode=input_reader.ModeKeys.PREDICT_WITH_GT,
            batch_size=params.eval.batch_size,
            num_examples=params.eval.eval_samples)
    if mode == 'train':
        def _model_fn(params):
            return model_builder.build_model(params, mode=input_reader.ModeKeys.TRAIN)

        logging.info(
            'Train num_replicas_in_sync %d num_workers %d is_multi_host %s' % (
                strategy.num_replicas_in_sync, 1, False))

        dist_executor = DetectionDistributedExecutor(
            strategy=strategy,
            params=params,
            model_fn=_model_fn,
            loss_fn=model_builder.build_loss_fn,
            is_multi_host=False,
            predict_post_process_fn=model_builder.post_processing,
            trainable_variables_filter=model_builder
                .make_filter_trainable_variables_fn())
        params.override(
            {
                'train': {"iterations_per_loop": 40,
                          "total_steps": 80}
            },
            is_strict=False)
        return dist_executor.train(
            train_input_fn=train_input_fn,
            model_dir=params.model_dir,
            iterations_per_loop=params.train.iterations_per_loop,
            total_steps=params.train.total_steps,
            init_checkpoint=model_builder.make_restore_checkpoint_fn(),
            custom_callbacks=None,
            save_config=True)


# mul_gpu_training_custom_data("efficientdet-d0",
#                              r"E:\Temp\test\tfrecord\*.tfrecord",
#                              None,21)
def main(_):
    mul_gpu_training_custom_data("efficientdet-d0",
                                 r"E:\Temp\test\tfrecord\*.tfrecord",
                                 None, 21)


if __name__ == '__main__':
    app.run(main)


## todo 自己重写循环或者重写keras model方法


def mul_gpu_training_custom_loop(model_name, training_file_pattern, eval_file_pattern, number_classes,
                                 pre_weights=None, mode="train"):
    """
    完全自己重写循环
    Args:
        model_name:
        training_file_pattern:
        eval_file_pattern:
        number_classes:
        pre_weights:
        mode:

    Returns:

    """
    params = config_factory.config_generator(model_name)
    params.architecture.num_classes = number_classes
    params.train.batch_size = 4
    # 设置分布式训练策略
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) == 0:
        strategy = tf.distribute.OneDeviceStrategy("device:CPU:0")
    elif len(gpus) == 1:
        strategy = tf.distribute.OneDeviceStrategy("device:GPU:0")
    else:
        strategy = tf.distribute.MirroredStrategy()
    # 建立模型与数据加载
    model_builder = EfficientDetModel(params)
    # if training_file_pattern:
    # Use global batch size for single host.
    train_input_fn = input_reader.InputFn(
        file_pattern=training_file_pattern,
        params=params,
        mode=input_reader.ModeKeys.PREDICT_WITH_GT,
        batch_size=params.train.batch_size)

# train_dataset = input_reader(batch_size=4)
# model_fn = EfficientDetModel(params)
# input_reader = input_reader.InputFn(r"E:\Temp\test\tfrecord\*.tfrecord", params, "train", 1)
#
# model = model_fn.build_model(params)
# loss_fn = model_fn.build_loss_fn()
# epochs = 10
#
# optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4)
# for epoch in range(epochs):
#     print('Start of epoch %d' % (epoch,))
#
#     # Iterate over the batches of the dataset.
#     for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
#
#         # Open a GradientTape to record the operations run
#         # during the forward pass, which enables autodifferentiation.
#         with tf.GradientTape() as tape:
#
#             # Run the forward pass of the layer.
#             # The operations that the layer applies
#             # to its inputs are going to be recorded
#             # on the GradientTape.
#             logits = model(x_batch_train, training=True)  # Logits for this minibatch
#
#             # Compute the loss value for this minibatch.
#             loss_value = loss_fn(y_batch_train, logits)
#
#         # Use the gradient tape to automatically retrieve
#         # the gradients of the trainable variables with respect to the loss.
#         grads = tape.gradient(loss_value['total_loss'], model.trainable_weights)
#
#         # Run one step of gradient descent by updating
#         # the value of the variables to minimize the loss.
#         optimizer.apply_gradients(zip(grads, model.trainable_weights))
#
#         # Log every 200 batches.
#         if step % 2 == 0:
#             print('Training loss (for one batch) at step %s: %s' % (step, float(loss_value['total_loss'])))
#             print('Seen so far: %s samples' % ((step + 1) * 64))
