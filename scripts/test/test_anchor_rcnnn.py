#!usr/bin/env python3
# -*- coding: UTF-8 -*-
import xl_tensorflow.models.vision.detection.configs.factory as config_factory
import xl_tensorflow.models.vision.detection.dataloader.input_reader as input_reader
params = config_factory.config_generator("retinanet")
reader = input_reader.InputFn(
        file_pattern=r"E:\Temp\test\efficiendet_merge_test\*.tfrecord",
        params=params,
        mode=input_reader.ModeKeys.TRAIN,
        batch_size=params.train.batch_size)

dataset = reader(batch_size=1)
data = (next(dataset.as_numpy_iterator()))
print(data)