from xl_tensorflow.models.vision.detection.dataloader.yolo_parser import Parser,tf_example_decoder
from xl_tensorflow.models.vision.detection.dataloader.input_reader import YoloInputFn
import tensorflow as tf
import sys
import numpy as np
import logging

logging.info("help me")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
tf.debugging.enable_check_numerics()
import xl_tensorflow.models.vision.detection.configs.factory as config_factory
from xl_tensorflow.models.vision.detection.architecture.fpn import BiFpn
from xl_tensorflow.models.vision import EfficientNetB0
from xl_tensorflow.models.vision.detection.body.efficientdet_model import EfficientDetModel
import glob
from tensorflow.keras import layers, Model
from xl_tensorflow.models.vision.detection.dataloader.input_reader import InputFn

#
# params = config_factory.config_generator("efficientdet-d0")
# params.architecture.num_classes = 86
# # params.efficientdet_parser.unmatched_threshold = 0.4
# input_reader = InputFn(r"./val/*.tfrecord", params, "predict_with_gt", 8)
# train_dataset = input_reader(batch_size=64)
def map_fn_test():
    dataset = tf.data.Dataset.list_files(
        r"E:\Temp\test\tfrecord\*.tfrecord", shuffle=False)

    def _prefetch_dataset(filename):
        dataset = tf.data.TFRecordDataset(filename).prefetch(1)
        return dataset

    dataset = dataset.apply(
        tf.data.experimental.parallel_interleave(
            _prefetch_dataset, cycle_length=32, sloppy=True))
    paser = Parser((416, 416), 20, use_bfloat16=False,use_autoaugment=True)
    fukc = dataset.as_numpy_iterator()
    for i in range(10):
        a = next(fukc)
        result = paser(a)
        import time
        time.sleep(1)
        print(result[1][0].shape)

def fuck_test():
    dataset = tf.data.Dataset.list_files(r"E:\Temp\test\tfrecord\*tfrecord")
    dataset = dataset.interleave(
        map_func=tf.data.TFRecordDataset, cycle_length=8,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    params = config_factory.config_generator("efficientdet-d0")
    params.architecture.num_classes = 86
    # params.efficientdet_parser.unmatched_threshold = 0.4
    inputfn = InputFn(r"E:\Temp\test\tfrecord\*tfrecord", params, "train", 8)
    # train_dataset = input_reader(batch_size=64)


    gen = (inputfn().as_numpy_iterator())
    # decoder = tf_example_decoder.TfExampleDecoder()
    # inputfn = YoloInputFn((416, 416), r"E:\Temp\test\tfrecord\*.tfrecord",mode="",
    #                       aug_scale_max=1.5,num_classes=85,use_autoaugment=False,
    #                       buffer=1)

    try:
        i = 0
        while True:
            i += 1
            try:
                temp = (next(gen))
                # if i< 20829:
                #     continue
                a = inputfn._parser_fn(temp)
                # temp = decoder.decode(temp)
                # print(temp["source_id"])
            except Exception as e:
                a = inputfn._parser_fn(temp)
                a = inputfn._parser_fn(temp)
                a = inputfn._parser_fn(temp)

            print(i)
    except Exception as e:
        raise e



def input_reader_test():
    inputfn = YoloInputFn((416, 416), r"E:\Temp\test\tfrecord\*.tfrecord",mode="", aug_scale_max=1.5,num_classes=85,use_autoaugment=False,buffer=1)
    dataset = inputfn(batch_size=1)
    gen = dataset.as_numpy_iterator()
    import time
    try:
        i = 0
        while True:
            time.sleep(0.1)
            # print("ddd")
            temp = (next(gen))
            time.sleep(0.1)
            print(temp[0].shape, i)
            i+=1
    except Exception as e:
        raise e

fuck_test()
# input_reader_test()
# print(result)
# map_fn_test()./val
