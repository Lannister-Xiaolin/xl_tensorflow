#!usr/bin/env python3
# -*- coding: UTF-8 -*-
import os
import pathlib
import numpy as np
import tensorflow as tf
from xl_tool.xl_concurrence import MyThread


def quantize_model(converter, method="float16", int_quantize_sample=(100, 224, 224, 3)):
    """量化模型
    Args:
        converter: tf.lite.TFLiteConverter对象
        method: str, valid value：float16,int,weight
        int_quantize_sample: int量化时使用的代表性数据集
            https://tensorflow.google.cn/lite/performance/post_training_integer_quant?hl=zh_cn
    """
    assert method in ("float16", "int", "weight")
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


def tf_saved_model_to_lite(model_path, save_lite_file, input_shape=None, quantize_method=None, allow_custom_ops=False):
    """
    tensorflow saved model转成lite格式
    Args:
        model_path:  saved_model path（include version directory）
        save_lite_file: lite file name(full path)
        input_shape； specified input shape, if none means  [None, 224, 224, 3]
        quantize_method: str, valid value：float16,int,weight
        allow_custom_ops:是否允许自定义算子
    """

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
    if allow_custom_ops:
        converter.allow_custom_ops = True
        converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS, tf.lite.OpsSet.TFLITE_BUILTINS]
    return pathlib.Path(save_lite_file).write_bytes(converter.convert())


def serving_model_export(model, path, version=1, auto_incre_version=True):
    """导出模型到tenserflow.serving 即savemodel格式
    注意使用signaturedefault默认是按照layer的层名进行定义的，多输出按照 layername, layername_1,
    Arg:
        model：keras或者tensorflow模型，注意输入层名字为：images_tensor， 输出层为：outputs
        path:模型存储路径，推荐最后一个文件夹以模型名称命名
        version:模型版本号，注意：path/{version}才是最终模型存储路径
        auto_incre_version: 是否自动叠加版本
    """
    if auto_incre_version is True:
        old_version = [int(i) for i in os.listdir(path) if i[0] in "0123456789" and os.path.isdir(path + "/" + i)]
        if old_version:
            version = max(old_version) + 1
        else:
            version = version
    version_path = os.path.join(path, str(version))
    os.makedirs(version_path, exist_ok=True)
    try:
        tf.saved_model.save(model, version_path, )
    except Exception as e:
        print("模型导出异常：{}".format(e))
        raise AssertionError


def b64_image_model_wrapper(model, target_size, method=tf.image.ResizeMethod.BILINEAR, mean=0.0, std=255.0,
                            input_name="b64_image", outs_signature="b64_output_tensor"):
    """使用b64的方式对图片进行处理, 注意图片必须未websafebase64方式，tensorflow默认以该方式解码"""

    def preprocess_and_decode(img_str, new_shape=target_size):
        img = tf.io.decode_base64(img_str)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, new_shape, method=method)
        return img

    def batch_decode_on_cpu(image_files):
        with tf.device("/cpu:0"):
            ouput_tensor = tf.map_fn(lambda im: preprocess_and_decode(im[0]), image_files, dtype="float32")
        return ouput_tensor

    input64 = tf.keras.layers.Input(shape=(1,), dtype="string", name=input_name)
    ouput_tensor = tf.keras.layers.Lambda(lambda x: batch_decode_on_cpu(x))(input64)
    x = (ouput_tensor - mean) / std
    x = model(x)
    new_model = tf.keras.Model(input64, x)
    new_model.output_names[0] = outs_signature
    return new_model


def load_lite_model(lite_model_path="./lite/net_lite/efficientnetb1/13/efficientnetb1_int_quant.tflite"):
    import tflite_runtime.interpreter as tflite
    interpreter = tflite.Interpreter(model_path=lite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


def tflite_evaluate(lite_model_path, keras_data_generater, mul_thread=None):
    """tflite评估函数"""
    predicts = []
    one_hots = []
    from tqdm import tqdm
    def lite_interpreter(keras_data_generater, start, end):
        pbar = tqdm(list(range(start, end)))
        interpreter, input_details, output_details = load_lite_model(lite_model_path)
        for i in pbar:
            input_data, label = keras_data_generater[i]
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predicts.append(output_data)
            one_hots.append(label)
            pbar.set_description("推理进度： ")
        return predicts, one_hots

    if mul_thread:
        from math import ceil
        threads = list(range(mul_thread))
        step = ceil(len(keras_data_generater) / mul_thread)
        for i in range(mul_thread):
            start = step * i
            end = step * (i + 1) if i < (mul_thread - 1) else len(keras_data_generater)
            threads[i] = MyThread(target=lite_interpreter,
                                  args=(keras_data_generater, start, end))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        for thread in threads:
            sub_predicts, sub_one_hots = thread.get_result()
            predicts.extend(sub_predicts)
            one_hots.extend(sub_one_hots)
    else:
        predicts, real = lite_interpreter(keras_data_generater, 0, len(keras_data_generater))
    real = np.concatenate(one_hots, axis=0).argmax(axis=-1)
    predicts = np.concatenate(predicts, axis=0)
    top1 = np.sum(predicts.argmax(axis=-1) == real) / len(predicts)
    return top1, predicts, real


def export_model_config(model_names):
    base = """model_config_list {{\n{}\n}}"""
    config_template = "  config {{\n    name: '{}',\n    " \
                      "base_path: '/models/{}/',\n\tmodel_platform: 'tensorflow'\n  }}"
    return base.format(",\n".join(map(lambda x: config_template.format(x, x), model_names)))
