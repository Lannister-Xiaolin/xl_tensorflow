#!usr/bin/env python3
# -*- coding: UTF-8 -*-
import os
import pathlib
import numpy as np
import tensorflow as tf


def quantize_model(converter, method="float16", int_quantize_sample=(100, 224, 224, 3)):
    """量化模型
    Args:
        converter: tf.lite.TFLiteConverter对象
        method：str, valid value：float16,int,weight
        int_quantize_sample: int量化时使用的代表性数据集
            https://tensorflow.google.cn/lite/performance/post_training_integer_quant?hl=zh_cn
    """
    if method == "float16":
        print("float16量化")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    if method == "int":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        images = np.random.random(int_quantize_sample).astype("float32")
        mnist_ds = tf.data.Dataset.from_tensor_slices((images)).batch(1)

        def representative_data_gen():
            for input_value in mnist_ds.take(100):
                yield [input_value]

        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    if method == "weight":
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    return converter


def tf_saved_model_to_lite(model_path, save_lite_file, input_shape=None, quantize_method=None):
    """
    tensorflow saved model转成lite格式
    Args:
        model_path:  saved_model path（include version directory）
        save_lite_file: lite file name(full path)
        input_shape； specified input shape, if none means  [None, 224, 224, 3]
        quantize_method: str, valid value：float16,int,weight
    """
    if input_shape:
        model = tf.saved_model.load(model_path)
        concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        concrete_func.inputs[0].set_shape(input_shape)
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    else:
        try:
            converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
        except ValueError:
            model = tf.saved_model.load(model_path)
            concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
            concrete_func.inputs[0].set_shape(input_shape if input_shape else [None, 224, 224, 3])
            converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    if quantize_method:
        converter = quantize_model(converter, quantize_method,
                                   (100, *input_shape[1:]) if input_shape else (100, 224, 224, 3))
    return pathlib.Path(save_lite_file).write_bytes(converter.convert())


def serving_model_export(model, path, version=1, auto_incre_version=True):
    """导出模型到tenserflow.serving 即savemodel格式
    Arg:
        model：keras或者tensorflow模型
        path:模型存储路径，推荐最后一个文件夹以模型名称命名
        version:模型版本号，注意：path/{version}才是最终模型存储路径
        auto_incre_version: 是否自动叠加版本
    """
    if auto_incre_version is True:
        version = max([int(i) for i in os.listdir(path) if os.path.isdir(path+"/"+i)]) + 1 if os.listdir(path) else version
    version_path = os.path.join(path, str(version))
    os.makedirs(version_path, exist_ok=True)
    try:
        tf.saved_model.save(model, version_path, )
    except Exception as e:
        print("模型导出异常：{}".format(e))
        raise AssertionError
# tf_saved_model_to_lite(r"E:\Temp\test\eff\1",r"E:\Temp\test\eff\test.tflite",input_shape=[None, 380, 380, 3])