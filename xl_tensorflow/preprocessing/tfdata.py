#!usr/bin/env python3
# -*- coding: UTF-8 -*-
import time

import tensorflow as tf

"""
评测方法:
    遍历（生成可训练的数据）完一个数据集（超过3万图片）需要花费的时间
"""


def parse_image(filename, target_size=None):
    parts = tf.strings.split(filename, '/')
    label = parts[-2]

    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    if target_size:
        image = tf.image.resize(image, [128, 128])
    return image, label


class ImageDataset(tf.data.Dataset):
    def _generator(num_samples):
        # Opening the file
        time.sleep(0.03)

        for sample_idx in range(num_samples):
            # Reading data (line, record) from the file
            time.sleep(0.015)

            yield (sample_idx,)

    def __new__(cls, num_samples=3):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=tf.dtypes.int64,
            output_shapes=(1,),
            args=(num_samples,)
        )
