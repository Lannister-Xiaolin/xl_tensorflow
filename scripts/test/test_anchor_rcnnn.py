#!usr/bin/env python3
# -*- coding: UTF-8 -*-
import xl_tensorflow.models.vision.detection.configs.factory as config_factory
import xl_tensorflow.models.vision.detection.dataloader.input_reader as input_reader
import numpy as np
params = config_factory.config_generator("efficientdet-d0")
params.efficientdet_parser.unmatched_threshold = 0.4
reader = input_reader.InputFn(
        file_pattern=r"E:\Temp\test\efficiendet_merge_test\*.tfrecord",
        params=params,
        mode=input_reader.ModeKeys.TRAIN,
        batch_size=params.train.batch_size)

dataset = reader(batch_size=1).as_numpy_iterator()
import time

for i in range(100):
        data = (next(dataset))
        # print("---",[np.array(data[1]["cls_targets"][i]).max() for i in range(3,8)])
        # print("---",[np.array(data[1]["cls_targets"][i]).min() for i in range(3,8)])
        time.sleep(1)
        print()