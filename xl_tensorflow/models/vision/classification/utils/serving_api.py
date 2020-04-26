#!usr/bin/env python3
# -*- coding: UTF-8 -*-


import json
import numpy as np
import requests
from PIL import Image
import grpc
import base64
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from google.protobuf.json_format import MessageToJson

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


def serving_request_image_classifier(image_files, cat2id, id2cat=None, model_name="efficientnetb2", target_size=None,
                                     top=3,
                                     serving_host="http://10.125.31.57:8501/v1/models/{}:predict",
                                     mean=0, std=255.0, b64_mode=False):
    """
    tensorflow.serving食物请求sdk
    Args:
        image_files: 图片文件路径列表，单个文件也必须为列表
        model_name: 模型名称，当前有效模型为默认名称efficientnetb2
        serving_host: 模型请求接口，目前只有默认值有效，多模型部署时，会更改接口
    Returns:
        正常时，返回top3结果，列表，格式如下：
            [{'broccoli': 0.769328654, 'pizza': 0.23008424, 'hamburger': 0.000517699867},
            {'pizza': 0.877195954, 'broccoli': 0.103550591, 'hamburger': 0.0165584981}]
        出现错误时：
            返回错误提示字典
    """
    if not b64_mode:
        if not target_size:
            target_size = [eff_input_dict[model_name]] * 2 if model_name in eff_input_dict.keys() else (224, 224)
        data = np.stack(
            [np.array(Image.open(image_file).resize(target_size)) - mean / std for image_file in image_files])
        data = data.tolist()
        data = json.dumps({
            "signature_name": "serving_default",
            "instances": data
        })
    else:
        data = [[base64.urlsafe_b64encode(open(i, "rb").read()).decode()] for i in image_files]
        data = json.dumps({
            "signature_name": "serving_default",
            "instances": data
        })
    result = requests.post(serving_host.format(model_name),
                           headers={"content-type": "application/json"},
                           data=data)
    id2cat = {v: k for k, v in cat2id.items()} if not id2cat else id2cat
    try:
        predict = np.array(json.loads(result.text)["predictions"])
        labels_top3 = predict.argsort()[:, -top:].tolist()
        prob_top3 = (np.sort(predict)[:, -top:]).tolist()
        result = [{id2cat[labels_top3[i][j]]: prob_top3[i][j] for j in range(top - 1, -1, -1)} for i in
                  range(len(labels_top3))]
    except KeyError:
        return json.loads(result)
    return result


def serving_grpc_image_classifier(image_files, cat2id, id2cat=None, model_name="efficientnetb0",
                                  target_size=None, top=3,
                                  serving_host="10.125.31.57:8500", mean=0, std=255.0, b64_mode=False):
    """grpc接口，速度为restful接口的1/3至1/5
    Notes: 注意输入层名字为：images_tensor， 输出层为：outputs
    """
    if not b64_mode:
        if not target_size:
            target_size = (
                eff_input_dict[model_name], eff_input_dict[model_name]) if model_name in eff_input_dict.keys() else (
                224, 224)
        data = np.stack(
            [np.array(Image.open(image_file).resize(target_size)) - mean / std for image_file in image_files])
    else:
        data = [[base64.urlsafe_b64encode(open(i, "rb").read()).decode()] for i in image_files]
    length = len(data)
    channel = grpc.insecure_channel(serving_host)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    id2cat = {v: k for k, v in cat2id.items()} if not id2cat else id2cat
    try:
        # model_pb2.py指定了version version_label signature_name name
        request.model_spec.name = model_name
        request.model_spec.signature_name = 'serving_default'
        request.inputs['images_tensor' if not b64_mode else "image_b64"].CopyFrom(
            tf.make_tensor_proto(data.astype("float32"), shape=[length, target_size[0], target_size[1], 3])
            if not base64 else
            tf.make_tensor_proto(data, shape=[length, 1]))
        result = stub.Predict(request, 10.0)
        outputs = (json.loads(MessageToJson(result))["outputs"]).values()
        predict = np.array(list(outputs)[0]['floatVal']).reshape((length, len(cat2id)))
        labels_top3 = predict.argsort()[:, -top:].tolist()
        prob_top3 = (np.sort(predict)[:, -top:]).tolist()
        result = [{id2cat[labels_top3[i][j]]: prob_top3[i][j] for j in range(top - 1, -1, -1)} for i in
                  range(len(labels_top3))]
    except grpc.RpcError as rpc_e:
        return {'error': rpc_e}
    return result


if __name__ == '__main__':
    cat2id = {
        "bacon": 0,
        "barbecued_pork": 1,
        "broccoli": 2,
        "chicken_breast": 3,
        "chicken_brittle_bone": 4,
        "chicken_feet": 5,
        "chicken_wing": 6,
        "chives": 7,
        "corn": 8,
        "drumstick": 9,
        "egg_tart": 10,
        "eggplant": 11,
        "empty": 12,
        "flammulina_velutipes": 13,
        "green_pepper": 14,
        "hamburger_steak": 15,
        "lamb_chops": 16,
        "lamb_kebab": 17,
        "oyster": 18,
        "peanut": 19,
        "pizza": 20,
        "potato": 21,
        "prawn": 22,
        "pumpkin_block": 23,
        "ribs": 24,
        "salmon": 25,
        "saury": 26,
        "sausage": 27,
        "scallion_cake": 28,
        "scallop": 29,
        "shiitake_mushroom": 30,
        "steak": 31,
        "sweet_potato": 32,
        "tilapia": 33,
        "toast": 34,
        "whole_chicken": 35
    }
    print(serving_request_image_classifier([r"F:\Download\A60200000042_20200323151101.jpg"], cat2id,
                                           model_name="efficientnetb6", top=4))
    print(serving_grpc_image_classifier([r"F:\Download\A60200000042_20200323151101.jpg"], cat2id,
                                        model_name="efficientnetb6", top=4))
