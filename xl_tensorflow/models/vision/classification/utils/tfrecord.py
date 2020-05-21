#!/usr/bin/python
import logging
import os
import xml

import tensorflow as tf
from math import ceil
from xl_tool.xl_io import file_scanning, save_to_json
import threading
import tensorflow_addons as tfa
import random
from xl_tensorflow.utils.common import _bytes_feature, _bytes_list_feature, _float_feature, _float_list_feature, \
    _int64_feature, _int64_list_feature
from .preprocessing import preprocess_for_train, preprocess_for_eval
from .autoaug import AutoAugment, RandAugment


def image2tfexample(filename, label="", class_id=0):
    """convert image to tensorflow example"""
    image_bytes = tf.io.read_file(filename)
    image_bytes = tf.image.encode_jpeg(tf.image.decode_image(image_bytes, channels=3))
    # if imghdr.what(filename) == 'png':
    #     filename = os.path.basename(filename).replace("png", "jpg")
    image_array = tf.image.decode_image(image_bytes)
    height, width = image_array.shape[0:2]
    example = tf.train.Example(features=tf.train.Features(feature={
        'width': _int64_feature(width),
        'height': _int64_feature(height),
        'image': _bytes_feature(image_bytes),
        'label': _bytes_feature(tf.compat.as_bytes(label)),
        'filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
        "class_id": _int64_feature(class_id),
    }))
    return example


def images2single_tfrecord(record_file, files, label2index=None, shuffle=True):
    writer = tf.io.TFRecordWriter(record_file)
    from tqdm import tqdm
    if shuffle:
        random.shuffle(files)
    pbar = tqdm(files)
    for file, label in pbar:
        try:
            # label = os.path.split(os.path.split(file)[0])[1] if label2class_id else ""
            class_id = label2index[label] if label2index else 0
        except KeyError:
            logging.warning("发现严重错误！")
            label = None
            class_id = None
            for label in label2index.keys():
                if label in file:
                    label = label
                    class_id = label2index[label] if label2index else 0
                    break
        exmple = image2tfexample(file, label, class_id)
        writer.write(exmple.SerializeToString())
        pbar.set_description("tfrecord转换进度：")
    writer.close()


def images2tfrecord(root_path, record_file, label2index_file, classes=None, mul_thread=None, shuffle=True,
                    file_format="jpg|jpeg|JPG|JPEG"):
    """
    convert image to .tfrecord file
    Args:
        root_path: image root path, please, confirm images are placed in different directories
        record_file: record_file name
        c2l_file:  classes to label id json file
        classes: List of String是否指定类别
        mul_thread: whether to use multhread, int to use mul thread
    Returns:
    """

    labels = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))] \
        if not classes else classes
    labels = sorted([d for d in labels])
    logging.info(f"发现类别数量：{len(labels)}")
    logging.info("\n".join(labels))
    label2index = {labels[i]: i for i in range(len(labels))} if labels else dict()
    files = []
    for d in labels:
        fs = file_scanning(root_path + "/" + d, file_format=file_format, sub_scan=True, full_path=True)
        logging.info(d + ": {}".format(len(fs)))
        lbs = [d, ] * len(fs)
        fs = list(zip(fs, lbs))
        files.extend(fs)
        # files = file_scanning(root_path, file_format="jpg|jpeg|png", sub_scan=True, full_path=True)
    logging.info(f"扫描到有效文件数量：{len(files)}")
    if shuffle:
        random.shuffle(files)
    if not mul_thread or mul_thread < 2:
        images2single_tfrecord(record_file, files, label2index)
    else:
        assert type(mul_thread) == int
        threads = []
        number = ceil(len(files) / mul_thread)
        for i in range(mul_thread):
            sub_thread_files = files[i * number:(i + 1) * number]
            sub_record_file = record_file + str(i)
            thread = threading.Thread(target=images2single_tfrecord,
                                      args=(sub_record_file, sub_thread_files, label2index))
            threads.append(thread)
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    save_to_json(label2index, label2index_file, indent=4)


def tf_image_augmentation(image, size, target_size=(224, 224), adjust_gamma=None, random_brightness=None,
                          resize_method="bilinear", random_contrast=None, rotate=None, zoom_range=None,
                          random_crop=None, random_flip_left_right=None, random_flip_up_down=None,
                          keep_aspect=True, noise=None, random_aspect=False):
    """
    Args:
        image: image tensor
        size: (height,wight)
        target_size: (size1,size2) 目标尺寸
        adjust_gamma: float or [lower, upper], 随机gamma校正, 通常大于1，越大色调越暗淡,推荐值：(0.8,1.2)，不推荐与brightness一起使用
        random_brightness: float，随机亮度值调整，0，1，0为原图，1为白色，推荐值：0-0.15
        resize_method: 调整
        random_contrast:随机对比度，None或者tuple/list, (lower,higher),大于1增加对比度，推荐(0.8,1.5)
        rotate: float, 随机进行旋转,单位为弧度，2*3.14表示一圈
        zoom_range:float or [lower, upper], 随机进行缩放 ，float的缩放范围为[1-zoom_range, 1+zoom_range]
        random_crop: float or tuple. 随机裁剪比例，ie:(0.9,0.9)
        random_flip_left_right: True or None左右翻转
        random_flip_up_down:True or None 上下翻转
        keep_aspect: bool, 是否保持长宽比,
        noise:None or float, 是否添加随机噪声，推荐：0.03
        random_aspect: 是否随机确定保留长宽比
    Returns:

    """
    # ToDO 新增其他数据增强方式
    if adjust_gamma:
        gamma = random.uniform(*adjust_gamma) if type(adjust_gamma) == tuple else adjust_gamma
        image = tf.image.adjust_gamma(image, gamma)
    if noise and type(noise) in (float, tf.float32):
        noise = tf.keras.backend.random_normal(shape=(*size, 3), mean=0.0, stddev=noise, dtype=tf.float32)
        image = noise + image
    image = tf.image.random_brightness(image, max_delta=random_brightness) if random_brightness else image
    image = tf.image.random_contrast(image, *random_contrast) if random_contrast else image
    if rotate:
        angle = random.uniform(-rotate, rotate)
        image = tfa.image.rotate(image, angle) if rotate else image
    if zoom_range:
        # 为确保缩放有效，需要先缩放，然后恢复到原尺寸（使用crop或者resize）

        zoom = random.uniform(1.0 - zoom_range, 1.0 + zoom_range) if type(
            zoom_range) == float else random.uniform(
            *zoom_range)
        zoom_size = (
            int(tf.cast(size[0], tf.float32) * zoom),
            int(tf.cast(size[1], tf.float32) * zoom))
        image = tf.image.resize(image, zoom_size, method=resize_method)
        if zoom > 1.0:
            image = tf.image.crop_to_bounding_box(image, (zoom_size[0] - size[0]) // 2, (zoom_size[1] - size[1]) // 2,
                                                  size[0], size[1])
        else:
            image = tf.image.pad_to_bounding_box(image, -(zoom_size[0] - size[0]) // 2, -(zoom_size[1] - size[1]) // 2,
                                                 size[0], size[1])
    if random_crop:
        crop_size = (
            random_crop * tf.cast(size[0], tf.float32),
            random_crop * tf.cast(size[1], tf.float32),
            3) if type(
            random_crop) in (float, tf.float32) else (
            random_crop[0] * tf.cast(size[0], tf.float32),
            random_crop[1] * tf.cast(size[1], tf.float32), 3)

        image = tf.image.random_crop(image, size=crop_size)
    image = tf.image.random_flip_left_right(image) if random_flip_left_right else image
    image = tf.image.random_flip_up_down(image) if random_flip_up_down else image
    if target_size:
        if random_aspect:
            aspect = random.choice([True, False])
            image = tf.image.resize(image, target_size,
                                    method=resize_method) if not aspect else tf.image.resize_with_pad(image,
                                                                                                      *target_size)
        else:
            image = tf.image.resize(image, target_size,
                                    method=resize_method) if not keep_aspect else tf.image.resize_with_pad(image,
                                                                                                           *target_size)

    return tf.clip_by_value(image, 0, 1)


def image_from_tfrecord(tf_record_files, num_classes=1000, batch_size=4, buffer_size=1000,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE,
                        target_size=(224, 224),
                        normalized_mean=0.0, normalized_std=255.0,
                        is_training=True, augmenter=AutoAugment(translate_const=25),
                        eval_no_crop_padding=True, min_object_covered: float = 0.7,
                        aspect_ratio_range: tuple = (0.4, 2.5),
                        area_range: tuple = (0.7, 1.0),
                        max_attempts: int = 10
                        ):
    """load data from tfrecord"""

    def parse_map_function(eg):
        example = tf.io.parse_example(eg[tf.newaxis], {
            'image': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            'class_id': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
            'width': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
            'height': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
        })
        image_bytes = example['image'][0]
        if normalized_mean not in [None, False]:
            if is_training:
                image = preprocess_for_train(image_bytes, image_size=target_size[0], augmenter=augmenter,
                                             mean_subtract=True,
                                             standardize=True, min_object_covered=min_object_covered,
                                             aspect_ratio_range=list(aspect_ratio_range),
                                             area_range=list(area_range),
                                             max_attempts=max_attempts, )
            else:
                image = preprocess_for_eval(image_bytes, target_size[0],
                                            mean_subtract=True,
                                            standardize=True, no_crop_padding=eval_no_crop_padding)
        else:
            if is_training:
                image = preprocess_for_train(image_bytes, image_size=target_size[0], augmenter=augmenter,
                                             mean_subtract=False,
                                             standardize=False, min_object_covered=min_object_covered,
                                             aspect_ratio_range=list(aspect_ratio_range),
                                             area_range=list(area_range),
                                             max_attempts=max_attempts, )
            else:
                image = preprocess_for_eval(image_bytes, target_size[0],
                                            mean_subtract=False, standardize=False,
                                            no_crop_padding=eval_no_crop_padding)

            image = (image - normalized_mean) / normalized_std
        class_id = tf.one_hot(example['class_id'][0], depth=num_classes)
        return image, class_id

    raw_dataset = tf.data.TFRecordDataset(tf_record_files)
    length = 0
    if not buffer_size:
        for _ in raw_dataset:
            length += 1
    buffer_size = buffer_size if buffer_size else length
    parsed_dataset = raw_dataset.map(parse_map_function, num_parallel_calls=num_parallel_calls).shuffle(buffer_size,
                                                                                                        reshuffle_each_iteration=True).batch(
        batch_size)
    return parsed_dataset


def image_from_tfrecord_hand(tf_record_files, num_classes=6, batch_size=8, buffer_size=20000,
                             num_parallel_calls=tf.data.experimental.AUTOTUNE,
                             target_size=(224, 224), resize_method="bilinear", normalized_mean=0.0,
                             normalized_std=255.0,
                             adjust_gamma=None, random_brightness=None,
                             random_contrast=None, rotate=None, zoom_range=None,
                             random_crop=None, random_flip_left_right=None, random_flip_up_down=None,
                             keep_aspect=True, noise=None, random_aspect=False):
    """load data from tfrecord"""

    # Todo 评估shuffle、cache等性能
    def parse_map_function(eg):
        example = tf.io.parse_example(eg[tf.newaxis], {
            'image': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            'class_id': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
            'width': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
            'height': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
        })
        image = (tf.cast(tf.io.decode_jpeg(example['image'][0], channels=3),
                         tf.float32) - normalized_mean) / normalized_std
        image = tf_image_augmentation(image,
                                      (tf.cast(example['height'][0], tf.int32), tf.cast(example['width'][0], tf.int32)),
                                      target_size=target_size,
                                      resize_method=resize_method,
                                      adjust_gamma=adjust_gamma, random_brightness=random_brightness,
                                      random_contrast=random_contrast, rotate=rotate, zoom_range=zoom_range,
                                      random_crop=random_crop, random_flip_left_right=random_flip_left_right,
                                      random_flip_up_down=random_flip_up_down,
                                      keep_aspect=keep_aspect, noise=noise, random_aspect=random_aspect)

        class_id = tf.one_hot(example['class_id'][0], depth=num_classes)
        return image, class_id

    raw_dataset = tf.data.TFRecordDataset(tf_record_files)
    length = 0
    if not buffer_size:
        for _ in raw_dataset:
            length += 1
    buffer_size = buffer_size if buffer_size else length
    parsed_dataset = raw_dataset.map(parse_map_function, num_parallel_calls=num_parallel_calls).shuffle(buffer_size,
                                                                                                        reshuffle_each_iteration=True).batch(
        batch_size)
    return parsed_dataset


if __name__ == '__main__':
    pass
