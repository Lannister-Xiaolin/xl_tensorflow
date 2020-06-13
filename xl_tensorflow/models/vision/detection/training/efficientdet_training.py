#!usr/bin/env python3
# -*- coding: UTF-8 -*-
import logging
import os

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
from xl_tensorflow.models.vision.detection.training.xl_detection_executor import DetectionDistributedExecutor
from absl import flags, app, logging

flags.DEFINE_integer('save_checkpoint_freq', None,
                     'Number of steps to save checkpoint.')
FLAGS = flags.FLAGS


def mul_gpu_training_custom_loop(model_name, training_file_pattern, eval_file_pattern, number_classes, optimizer="adam",
                                 mode="train", train_batch_size=4, eval_batch_size=None, iterations_per_loop=None,
                                 total_steps=None,
                                 model_dir=None,
                                 learning_rate=0.01, save_freq=None, pre_weights=None,
                                 l2_weight_decay=None, eval_samples=None,
                                 autoaugment_policy_name="v0",
                                 autoaugment_ratio=0.8,
                                 use_autoaugment=False,
                                 unmatched_threshold=0.5,
                                 aug_scale_min=0.1,
                                 aug_scale_max=2.0):
    """

    Args:
        model_name:
        training_file_pattern:
        eval_file_pattern:
        number_classes:
        optimizer:
        mode:
        train_batch_size:
        eval_batch_size:
        iterations_per_loop:
        total_steps:
        model_dir:
        learning_rate:
        save_freq:
        pre_weights:
        l2_weight_decay:
        eval_samples:  评估数据集数量，-1表示全部，None 为5000，可选任意大于0的数字

    Returns:

    """
    # todo 提前终止，以及其他损失函数
    # todo keras格式权重保存， 预训练权重加载，以及冻结网络层训练等

    params = config_factory.config_generator(model_name)
    params.architecture.num_classes = number_classes
    params.train.batch_size = train_batch_size
    params.train.l2_weight_decay = l2_weight_decay if l2_weight_decay else params.train.l2_weight_decay
    params.train.optimizer.type = optimizer
    params.train.iterations_per_loop = params.train.iterations_per_loop if not iterations_per_loop else iterations_per_loop
    params.train.total_steps = params.train.total_steps if not total_steps else total_steps
    params.efficientdet_parser.use_autoaugment = use_autoaugment
    params.efficientdet_parser.autoaugment_policy_name = autoaugment_policy_name
    params.efficientdet_parser.autoaugment_ratio = autoaugment_ratio
    params.efficientdet_parser.unmatched_threshold = unmatched_threshold
    params.efficientdet_parser.aug_scale_min = aug_scale_min
    params.efficientdet_parser.aug_scale_max = aug_scale_max

    params.train.override({'learning_rate': {
        'type': 'step',
        'warmup_learning_rate': learning_rate * 0.1,
        'warmup_steps': max(int(params.train.total_steps * 0.02), 200),
        'init_learning_rate': learning_rate,
        'learning_rate_levels': [learning_rate * 0.1, learning_rate * 0.01],
        'learning_rate_steps': [int(params.train.total_steps * 0.67), int(params.train.total_steps * 0.83)],
    }}, is_strict=False)

    # 模型保存路径与checkpoint保存路径
    model_dir = "./model" if not model_dir else model_dir
    os.makedirs(model_dir, exist_ok=True)
    # 设置分布式训练策略
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) == 0:
        strategy = tf.distribute.OneDeviceStrategy("device:CPU:0")
        logging.info("No gpu devices, using cpu")
    elif len(gpus) == 1:
        strategy = tf.distribute.OneDeviceStrategy("device:GPU:0")
        logging.info("Find one  gpu devices, using OneDeviceStrategy")
    else:
        strategy = tf.distribute.MirroredStrategy()
        logging.info("Find {}  gpu devices, using MirroredStrategy".format(len(gpus)))
    # 建立模型与数据加载
    model_builder = EfficientDetModel(params)
    train_input_fn = input_reader.InputFn(
        file_pattern=training_file_pattern,
        params=params,
        mode=input_reader.ModeKeys.TRAIN,
        batch_size=params.train.batch_size)
    if eval_file_pattern:
        eval_input_fn = input_reader.InputFn(
            file_pattern=eval_file_pattern,
            params=params,
            mode=input_reader.ModeKeys.PREDICT_WITH_GT,
            batch_size=eval_batch_size if eval_batch_size else train_batch_size,
            num_examples=params.eval.eval_samples if eval_samples is None else eval_samples)
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
            trainable_variables_filter=model_builder.make_filter_trainable_variables_fn())
        return dist_executor.train(
            train_input_fn=train_input_fn,
            eval_input_fn=eval_input_fn if eval_file_pattern else None,
            eval_metric_fn=model_builder.eval_metrics if eval_file_pattern else None,
            model_dir=model_dir,
            iterations_per_loop=params.train.iterations_per_loop,
            total_steps=params.train.total_steps,
            init_checkpoint=model_builder.make_restore_checkpoint_fn(),
            custom_callbacks=None,
            save_config=True, save_freq=save_freq, pre_weights=pre_weights)


# mul_gpu_training_custom_data("efficientdet-d0",
#                              r"E:\Temp\test\tfrecord\*.tfrecord",
#                              None,21)
def main(_):
    mul_gpu_training_custom_loop("efficientdet-d0",
                                 r"E:\Temp\test\tfrecord\*.tfrecord",
                                 r"E:\Temp\test\tfrecord\*.tfrecord", 21, train_batch_size=4, iterations_per_loop=10,
                                 total_steps=300)


if __name__ == '__main__':
    app.run(main)
