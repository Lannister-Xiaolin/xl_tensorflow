#!usr/bin/env python3
# -*- coding: UTF-8 -*-
import xl_tensorflow.models.vision.detection.configs.factory as config_factory
import xl_tensorflow.models.vision.detection.dataloader.input_reader as input_reader
from xl_tensorflow.models.vision.detection.body import factory as model_factory
# from xl_tensorflow.models.vision import
params = config_factory.config_generator("retinanet")
params.architecture.use_bfloat16 = False
train_input_fn = input_reader.InputFn(
        file_pattern=r"E:\Temp\test\efficiendet_merge_test\*.tfrecord",
        params=params,
        mode=input_reader.ModeKeys.TRAIN,
        batch_size=params.train.batch_size)
model_builder = model_factory.model_generator(params)
model = model_builder.build_model(params.as_dict())
dataset = train_input_fn(batch_size=1)
data = (next(dataset.as_numpy_iterator()))
# loss = model_builder.build_loss_fn()
model.compile(loss="sparse_categorical_crossentropy")
print(data[0].shape)
print(model.predict(data[0]))
print(model.outputs)
print(model.inputs)
# print(model.summary())

# print(data)