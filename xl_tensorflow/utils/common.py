#!usr/bin/env python3
# -*- coding: UTF-8 -*-
from functools import reduce
from contextlib import contextmanager
import time
import tensorflow as tf
import sys
import os

MEAN_RGB = (0.485 * 255, 0.456 * 255, 0.406 * 255)
STDDEV_RGB = (0.229 * 255, 0.224 * 255, 0.225 * 255)

TRAIN = 'train'
EVAL = 'eval'
PREDICT = 'predict'
PREDICT_WITH_GT = 'predict_with_gt'


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.
    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


class nondistribute:
    """空策略，替代单GPU和CPU"""

    @contextmanager
    def scope(self):
        start = time.time()
        try:
            yield
        finally:
            end = time.time()
            print('{}: {}'.format("执行时间", end - start))


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_list_feature(value):
    """int64 list to feature(value don't need to and '[]")"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_list_feature(value):
    """float list to feature(value don't need to and '[]")"""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_list_feature(value):
    """byte list to feature(value don't need to and '[]")"""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}


def xl_call_backs(model_name, log_path=None, model_path=None, monitor="val_loss", patience=5,
                  reduce_lr=3, factor=0.2, update_freq="epoch", save_best_only=True):
    """回调函数列表，包括tensorboard, 学习率衰减, 提前终止，模型检测点"""
    if "win" in sys.platform:
        log_dir = os.path.join(os.getcwd(), r"\logs\{}".format(model_name)) if not log_path else os.path.join(log_path,
                                                                                                              model_name)
    else:
        log_dir = r"./logs/{}".format(model_name) if not log_path else log_path
    model_path = "./model" if not model_path else model_path
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    print(os.path.abspath(model_path), os.path.abspath(log_dir))
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=False, histogram_freq=False,
                                                 update_freq=update_freq)
    reducelr = tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=factor, patience=reduce_lr)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor=monitor, min_delta=1e-7, patience=patience, verbose=0,
                                                  mode='auto',
                                                  baseline=None)
    model_check_point = tf.keras.callbacks.ModelCheckpoint(os.path.join(model_path,
                                                                        "{epoch:03d}_val_{loss:.3f}_train_{loss:.3f}"
                                                                        + f"_{model_name}_weights.h5" if not save_best_only else os.path.join(
                                                                            model_path, f"{model_name}_weights.h5")),
                                                           verbose=1,
                                                           save_best_only=save_best_only,
                                                           save_weights_only=True)
    return [tensorboard, reducelr, early_stop, model_check_point]
