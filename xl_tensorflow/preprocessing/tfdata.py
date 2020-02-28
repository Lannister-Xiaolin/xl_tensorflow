#!/usr/bin/python
import logging
import sys
import os
import tensorflow as tf
import numpy as np
import imghdr
import threading
from math import ceil
from xl_tool.xl_io import file_scanning, save_to_json
import xl_tool.xl_concurrence
import threading


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


def image2tfexample(filename, label="", class_id=0):
    """convert image to tensorflow example"""
    image_bytes = tf.io.read_file(filename)
    image_bytes = tf.image.encode_jpeg(tf.image.decode_image(image_bytes, channels=3))
    if imghdr.what(filename) == 'png':
        filename = os.path.basename(filename).replace("png", "jpg")
    image_array = tf.image.decode_image(image_bytes)
    height, width = image_array.shape[0:2]
    example = tf.train.Example(features=tf.train.Features(feature={
        'width': _int64_feature(width),
        'height': _int64_feature(height),
        'image': _bytes_feature(image_bytes),
        'label': _bytes_feature(tf.compat.as_bytes(label)),
        'filename': _bytes_feature(tf.compat.as_bytes(filename)),
        "class_id": _int64_feature(class_id),
    }))
    return example



def write_tfrecord(record_file, files, label2class_id=None):
    writer = tf.io.TFRecordWriter(record_file)
    for file in files:
        label = os.path.split(os.path.split(file)[0])[1] if label2class_id else ""
        class_id = label2class_id[label] if label2class_id else 0
        exmple = image2tfexample(file, label, class_id)
        writer.write(exmple.SerializeToString())
    writer.close()


def images_to_tfrecord(root_path, record_file, c2l_file, mul_thread=None):
    """
    convert image to .tfrecord file
    Args:
        root_path: image root path, please, confirm images are placed in different directories
        record_file: record_file name
        c2l_file:  classes to label id json file
        mul_thread: whether to use multhread, int to use mul thread
    Returns:
    """

    labels = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    logging.info(f"发现类别数量：{len(labels)}")
    label2class_id = {labels[i]: i for i in range(len(labels))} if labels else dict()
    files = file_scanning(root_path, file_format="jpg|jpeg|png", sub_scan=True, full_path=True)
    logging.info(f"扫描到有效文件数量：{len(files)}")
    if not mul_thread or mul_thread < 2:
        write_tfrecord(record_file, files, label2class_id)
    else:
        assert type(mul_thread) == int
        threads = []
        number = ceil(len(files) / mul_thread)
        for i in range(mul_thread):
            sub_thread_files = files[i * number:(i + 1) * number]
            sub_record_file = record_file + str(i)
            thread = threading.Thread(target=write_tfrecord, args=(sub_record_file, sub_thread_files, label2class_id))
            threads.append(thread)
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    save_to_json(label2class_id, c2l_file, indent=4)


def tf_data_from_tfrecord(tf_record_files, num_classes=6, batch_size=8,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE,
                          target_size=(224, 224),resize_method="bilinear"):
    """convert image dataset to tfrecord"""

    def parse_map_function(eg):
        example = tf.io.parse_example(eg[tf.newaxis], {
            'image': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            'class_id': tf.io.FixedLenFeature(shape=(), dtype=tf.int64)
        })
        # TODO 补充图片预处理和数据增强程序
        image = tf.image.resize(tf.io.decode_jpeg(example['image'][0], channels=3),
                                target_size,method=resize_method)
        class_id = tf.one_hot(example['class_id'][0], depth=num_classes)
        return image, class_id

    raw_dataset = tf.data.TFRecordDataset(tf_record_files)
    parsed_dataset = raw_dataset.map(parse_map_function, num_parallel_calls=num_parallel_calls).batch(batch_size)
    return parsed_dataset


if __name__ == '__main__':
    images_to_tfrecord(r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\2_网络图片\2_未标注",
                       r"F:\Download\x.tfrecords",
                       r"F:\Download\a.json",
                       mul_thread=0)
