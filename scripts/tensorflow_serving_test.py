#!usr/bin/env python3
# -*- coding: UTF-8 -*-
# todo 待完成

import json
import numpy as np
import requests
from PIL import Image
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import grpc
import tensorflow as tf
import base64
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from google.protobuf.json_format import MessageToJson, Parse
import requests
serving_host="10.125.31.57:8503"
data = [[base64.urlsafe_b64encode(open(i,"rb").read()).decode()] for i in ["./test_image/2.jpg"]]
channel = grpc.insecure_channel(serving_host)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
request = predict_pb2.PredictRequest()
# id2cat = dict(map(lambda i:(i[1],i[0]), cat2id.items()))
request.model_spec.name = "efficientnetb1_b64"
request.model_spec.signature_name = 'serving_default'
request.inputs['b64_image'].CopyFrom(
    tf.make_tensor_proto(data,shape=[2,1]))
result = stub.Predict(request, 10.0)
outputs = (json.loads(MessageToJson(result))["outputs"]).values()