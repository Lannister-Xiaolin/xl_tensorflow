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


def _int64_list_feature(value):
    """int64 list to feature(value don't need to and '[]")"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_list_feature(value):
    """float list to feature(value don't need to and '[]")"""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_list_feature(value):
    """byte list to feature(value don't need to and '[]")"""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


coordinate_name = ["xmin", "ymin", "xmax", "ymax"]


def get_bndbox(xml_file):
    """提取voc标注文件里面的bounding box坐标
    Args:
        xml_file: xml文件
    Returns：
        bndboxes:
            [{"name":annonation_class,"coordinates"：["xmin", "ymin", "xmax", "ymax"]}]
    """
    dom = xml.dom.minidom.parse(xml_file)
    doc = dom.documentElement
    xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = []  # List of normalized right x coordinates in bounding box
    ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = []  # List of normalized bottom y coordinates in bounding box
    classes_text = []
    for object_node in doc.getElementsByTagName('object'):
        classes_text.append(object_node.getElementsByTagName('name')[0].firstChild.data)
        xmins.append(int(object_node.getElementsByTagName("xmin")[0].firstChild.data))
        xmaxs.append(int(object_node.getElementsByTagName("xmax")[0].firstChild.data))
        ymins.append(int(object_node.getElementsByTagName("ymin")[0].firstChild.data))
        ymaxs.append(int(object_node.getElementsByTagName("ymax")[0].firstChild.data))
    return xmins, ymins, xmaxs, ymaxs, classes_text


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


def voc2tfexample(image_file, xml_file, label2id):
    """
    convert voc labeled object data to tf.Example
    Args:
        image_file: image file
        xml_file: xml file
        label2id: label to id

    Returns:
        tf.Example object
    """
    image_bytes = tf.io.read_file(image_file)
    image_bytes = tf.image.encode_jpeg(tf.image.decode_image(image_bytes, channels=3))
    image_array = tf.image.decode_image(image_bytes)
    height, width = image_array.shape[0:2]
    filename = tf.compat.as_bytes(
        os.path.basename(image_file))  # Filename of the image. Empty if image is not from file
    encoded_image_data = image_bytes  # Encoded image bytes
    image_format = tf.compat.as_bytes(os.path.basename(image_file).split(".")[-1])

    xmins, ymins, xmaxs, ymaxs, classes_text = get_bndbox(xml_file)
    classes = [label2id[name] for name in classes_text]  # List of integer class id of bounding box (1 per box)
    classes_text = [tf.compat.as_bytes(name) for name in classes_text]
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/filename': _bytes_feature(filename),
        'image/source_id': _bytes_feature(filename),
        'image/encoded': _bytes_feature(encoded_image_data),
        'image/format': _bytes_feature(image_format),
        'image/object/bbox/xmin': _float_list_feature(xmins),
        'image/object/bbox/xmax': _float_list_feature(xmaxs),
        'image/object/bbox/ymin': _float_list_feature(ymins),
        'image/object/bbox/ymax': _float_list_feature(ymaxs),
        'image/object/class/text': _bytes_list_feature(classes_text),
        'image/object/class/label': _int64_list_feature(classes),
    }))
    return example


def write_image_tfrecord(record_file, files, label2id=None):
    writer = tf.io.TFRecordWriter(record_file)
    from tqdm import tqdm
    from random import shuffle
    shuffle(files)
    shuffle(files)
    pbar = tqdm(files)
    for file, label in pbar:
        try:
            # label = os.path.split(os.path.split(file)[0])[1] if label2class_id else ""
            class_id = label2id[label] if label2id else 0
        except KeyError:
            logging.warning("发现严重错误！")
            label = None
            class_id = None
            for label in label2id.keys():
                if label in file:
                    label = label
                    class_id = label2id[label] if label2id else 0
                    break
        exmple = image2tfexample(file, label, class_id)
        writer.write(exmple.SerializeToString())
        pbar.set_description("tfrecord转换进度：")
    writer.close()


def write_voc_tfrecord(record_file, files, label2id):
    """
    Args:
        record_file:record file name
        files: list, ie:[(image_file,xml_file), ......, ]
        label2id:label to id dict
    """
    writer = tf.io.TFRecordWriter(record_file)
    from tqdm import tqdm
    from random import shuffle
    shuffle(files)
    shuffle(files)
    pbar = tqdm(files)
    for image_file, xml_file in pbar:
        exmple = voc2tfexample(image_file, xml_file, label2id)
        writer.write(exmple.SerializeToString())
        pbar.set_description("tfrecord转换进度：")
    writer.close()


def vocs2tfrecord(image_path, xml_path, record_file, label2id, label2id_file=None, mul_thread=None):
    """
    convert image to .tfrecord file
    Args:
        image_path: image root path, please, confirm images are placed in different directories
        xml_path
        record_file: record_file name
        label2id_file:  classes to label id json file
        label2id:
        mul_thread: whether to use multhread, int to use mul thread
    Returns:
    """
    image_files = file_scanning(image_path, file_format="jpg|jpeg|png", sub_scan=True, full_path=True)
    xml_files = file_scanning(xml_path, file_format="xml", sub_scan=True, full_path=True)
    valid_files = set([os.path.basename(file).split(".")[0] for file in image_files]) & set(
        [os.path.basename(file).split(".")[0] for file in xml_files])
    image_files = (sorted([file for file in image_files if os.path.basename(file).split(".")[0] in valid_files],
                          key=lambda x: os.path.basename(x).split(".")[0]))
    xml_files = (sorted([file for file in xml_files if os.path.basename(file).split(".")[0] in valid_files],
                        key=lambda x: os.path.basename(x).split(".")[0]))
    files = list(zip(image_files, xml_files))
    logging.info(f"发现图片：{len(image_files)}张\txml文件：{xml_files}")

    if not mul_thread or mul_thread < 2:
        write_voc_tfrecord(record_file, files, label2id)
    else:
        assert type(mul_thread) == int
        threads = []
        number = ceil(len(files) / mul_thread)
        for i in range(mul_thread):
            sub_thread_files = files[i * number:(i + 1) * number]
            sub_record_file = record_file + str(i)
            thread = threading.Thread(target=write_voc_tfrecord,
                                      args=(sub_record_file, sub_thread_files, label2id))
            threads.append(thread)
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    if label2id_file:
        save_to_json(label2id, label2id_file, indent=4)


def images2tfrecord(root_path, record_file, label2id_file, classes=None, mul_thread=None):
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
    label2id = {labels[i]: i for i in range(len(labels))} if labels else dict()
    files = []
    for d in labels:
        fs = file_scanning(root_path + "/" + d, file_format="jpg|jpeg|png", sub_scan=True, full_path=True)
        logging.info(d + ": {}".format(len(fs)))
        lbs = [d, ] * len(fs)
        fs = list(zip(fs, lbs))
        files.extend(fs)
        # files = file_scanning(root_path, file_format="jpg|jpeg|png", sub_scan=True, full_path=True)
    logging.info(f"扫描到有效文件数量：{len(files)}")
    if not mul_thread or mul_thread < 2:
        write_image_tfrecord(record_file, files, label2id)
    else:
        assert type(mul_thread) == int
        threads = []
        number = ceil(len(files) / mul_thread)
        for i in range(mul_thread):
            sub_thread_files = files[i * number:(i + 1) * number]
            sub_record_file = record_file + str(i)
            thread = threading.Thread(target=write_image_tfrecord,
                                      args=(sub_record_file, sub_thread_files, label2id))
            threads.append(thread)
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    save_to_json(label2id, label2id_file, indent=4)


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


def image_from_tfrecord(tf_record_files, num_classes=6, batch_size=8, buffer_size=20000,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE,
                        target_size=(224, 224), resize_method="bilinear", normalized_mean=0.0, normalized_std=255.0,
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


def yolo_data_from_tfrecord():
    # Todo 待完成，暂不确定加载形式
    pass


def efficientdet_data_from_tfrecord():
    # Todo 待完成，暂不确定加载形式
    # [https://github.com/tensorflow/models/blob/master/official/vision/detection/dataloader/tf_example_decoder.py]
    pass


if __name__ == '__main__':
    # a = voc2tfexample(
    #     r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\1_真实场景\0_已标框\bacon\0a634ce3-4f09-5bd4-99a0-238331e88fad.jpg",
    #     r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\1_真实场景\0_已标框\bacon\0a634ce3-4f09-5bd4-99a0-238331e88fad.xml",
    #     {"bacon": 0})
    vocs2tfrecord(r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\1_真实场景\0_已标框",
                  r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\1_真实场景\0_已标框",
                  r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\1_真实场景\fuck.tfrecord",
                  {"bacon": 0, "broccoli": 1, "corn": 2, "corn kernels": 3, "hamburger": 4, "pizza": 5,
                   "pork belly piece": 6}, mul_thread=3)
    # print(a)
