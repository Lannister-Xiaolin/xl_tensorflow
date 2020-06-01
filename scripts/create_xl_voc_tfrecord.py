# Copyright 2020 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Convert PASCAL dataset to TFRecord.

Example usage:
   python ./xl-tensorflow/scripts/create_xl_voc_tfrecord.py  --data_dir=./train --output_path=./tfrecord/train
     --label_map_json_path=food.json  --num_shards=8
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import json
import os

from absl import app
from absl import flags
from absl import logging

from lxml import etree
import PIL.Image
import tensorflow.compat.v1 as tf
import multiprocessing
import threading
from xl_tensorflow.datasets.tfrecord import tfrecord_util

flags.DEFINE_string('data_dir', '', 'Root directory to  VOC format dataset include image and xml files.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord and json.')
flags.DEFINE_string('label_map_json_path', None,
                    'Path to label map json file with a dictionary.')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                                                          'difficult instances')
flags.DEFINE_integer('num_shards', 8, 'Number of shards for output file.')
flags.DEFINE_integer('num_images', None, 'Max number of imags to process.')
flags.DEFINE_integer('image_path', None, 'image path if image file and xml file saved in different dir')
flags.DEFINE_string('prefix', "voc", 'prefix for output name of tfrecord')
flags.DEFINE_integer('num_threads', None, 'Number of threads to run must be equal to num_shards if set')
FLAGS = flags.FLAGS

GLOBAL_IMG_ID = 0  # global image id.
GLOBAL_ANN_ID = 0  # global annotation id.


def get_image_id(filename):
    """Convert a string to a integer."""
    # Warning: this function is highly specific to pascal filename!!
    # Given filename like '2008_000002', we cannot use id 2008000002 because our
    # code internally will convert the int value to float32 and back to int, which
    # would cause value mismatch int(float32(2008000002)) != int(2008000002).
    # COCO needs int values, here we just use a incremental global_id, but
    # users should customize their own ways to generate filename.
    del filename
    global GLOBAL_IMG_ID
    GLOBAL_IMG_ID += 1
    return GLOBAL_IMG_ID


def get_ann_id():
    """Return unique annotation id across images."""
    global GLOBAL_ANN_ID
    GLOBAL_ANN_ID += 1
    return GLOBAL_ANN_ID


def dict_to_tf_example(data,
                       image_file,
                       label_map_dict,
                       auto_label_map=False,
                       auto_label_index=-1,
                       ignore_difficult_instances=False,
                       ann_json_dict=None):
    """Convert XML derived dict to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    Args:
      data: dict holding PASCAL XML fields for a single image (obtained by running
        tfrecord_util.recursive_parse_xml_to_dict)
      dataset_directory: Path to root directory holding PASCAL dataset
      label_map_dict: A map from string label names to integers ids.
      ignore_difficult_instances: Whether to skip difficult instances in the
        dataset  (default: False).
      image_subdirectory: String specifying subdirectory within the PASCAL dataset
        directory holding the actual image data.
      ann_json_dict: annotation json dictionary.

    Returns:
      example: The converted tf.Example.

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    # img_path = os.path.join(data['folder'], image_subdirectory, data['filename'])
    full_path = image_file
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width = int(data['size']['width'])
    height = int(data['size']['height'])
    assert image.size[0] == width, f"实际图片尺寸{image.size[0]},{image.size[1]}与标注{width},{height}不一致"
    assert image.size[1] == height, f"实际图片尺寸{image.size[0]},{image.size[1]}与标注{width},{height}不一致"
    image_id = get_image_id(data['filename'])
    if ann_json_dict:
        image = {
            'file_name': data['filename'],
            'height': height,
            'width': width,
            'id': image_id,
        }
        ann_json_dict['images'].append(image)

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    if 'object' in data:
        for obj in data['object']:
            difficult = bool(int(obj['difficult']))
            if ignore_difficult_instances and difficult:
                continue

            difficult_obj.append(int(difficult))

            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append(obj['name'].encode('utf8'))
            if not auto_label_map:
                try:
                    classes.append(label_map_dict[obj['name']])
                except KeyError:
                    logging.warning(f"unknown classses: {obj['name']}")
                    continue
            else:
                try:
                    classes.append(label_map_dict[obj['name']])
                except KeyError:
                    auto_label_index = auto_label_index + 1
                    classes.append(auto_label_index)
                    label_map_dict[obj['name']] = auto_label_index

            truncated.append(int(obj['truncated']))
            poses.append(obj['pose'].encode('utf8'))

            if ann_json_dict:
                abs_xmin = int(obj['bndbox']['xmin'])
                abs_ymin = int(obj['bndbox']['ymin'])
                abs_xmax = int(obj['bndbox']['xmax'])
                abs_ymax = int(obj['bndbox']['ymax'])
                abs_width = abs_xmax - abs_xmin
                abs_height = abs_ymax - abs_ymin
                ann = {
                    'area': abs_width * abs_height,
                    'iscrowd': 0,
                    'image_id': image_id,
                    'bbox': [abs_xmin, abs_ymin, abs_width, abs_height],
                    'category_id': label_map_dict[obj['name']],
                    'id': get_ann_id(),
                    'ignore': 0,
                    'segmentation': [],
                }
                ann_json_dict['annotations'].append(ann)

    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/height':
                    tfrecord_util.int64_feature(height),
                'image/width':
                    tfrecord_util.int64_feature(width),
                'image/filename':
                    tfrecord_util.bytes_feature(data['filename'].encode('utf8')),
                'image/source_id':
                    tfrecord_util.bytes_feature(str(image_id).encode('utf8')),
                'image/key/sha256':
                    tfrecord_util.bytes_feature(key.encode('utf8')),
                'image/encoded':
                    tfrecord_util.bytes_feature(encoded_jpg),
                'image/format':
                    tfrecord_util.bytes_feature('jpeg'.encode('utf8')),
                'image/object/bbox/xmin':
                    tfrecord_util.float_list_feature(xmin),
                'image/object/bbox/xmax':
                    tfrecord_util.float_list_feature(xmax),
                'image/object/bbox/ymin':
                    tfrecord_util.float_list_feature(ymin),
                'image/object/bbox/ymax':
                    tfrecord_util.float_list_feature(ymax),
                'image/object/class/text':
                    tfrecord_util.bytes_list_feature(classes_text),
                'image/object/class/label':
                    tfrecord_util.int64_list_feature(classes),
                'image/object/difficult':
                    tfrecord_util.int64_list_feature(difficult_obj),
                'image/object/truncated':
                    tfrecord_util.int64_list_feature(truncated),
                'image/object/view':
                    tfrecord_util.bytes_list_feature(poses),
            }))
    return example, auto_label_index


def batch_processing(record_index, xml_files, image_files, label_map_dict, auto_label_map, ann_json_dict,
                     auto_label_index, total_number):
    writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.output_path, FLAGS.prefix + '-%05d-of-%05d.tfrecord' %
                                                      (record_index, FLAGS.num_shards)))
    for idx in range(len(xml_files)):
        if FLAGS.num_images and idx >= FLAGS.num_images:
            break
        if idx % 100 == 0:
            logging.info('On image %d of %d in %d total images, shard_index: %d', idx, len(xml_files), total_number,
                         record_index)
        path = xml_files[idx]
        with tf.gfile.GFile(path, 'r') as fid:
            xml_str = fid.read()
        try:
            xml = etree.fromstring(xml_str)
        except ValueError:
            xml = etree.fromstring(xml_str.encode("utf-8"))
        data = tfrecord_util.recursive_parse_xml_to_dict(xml)['annotation']
        try:
            tf_example, auto_label_index = dict_to_tf_example(
                data, image_files[idx],
                label_map_dict,
                auto_label_map=auto_label_map,
                auto_label_index=auto_label_index,
                ignore_difficult_instances=FLAGS.ignore_difficult_instances,
                ann_json_dict=ann_json_dict)
        except (ZeroDivisionError, AssertionError):
            logging.warning("ZeroDivisionError" + path)
            continue
        writer.write(tf_example.SerializeToString())

    writer.close()


def main(_):
    import time
    st = time.time()
    if not FLAGS.output_path:
        raise ValueError('output_path cannot be empty.')

    data_dir = FLAGS.data_dir
    image_path = FLAGS.image_path
    os.makedirs(FLAGS.output_path, exist_ok=True)
    logging.info('writing to output path: %s', FLAGS.output_path)
    print(FLAGS.output_path + '-%05d-of-%05d.tfrecord' %
          (1, FLAGS.num_shards))

    from xl_tool.xl_io import file_scanning
    from random import shuffle
    xml_files = file_scanning(data_dir, "xml", sub_scan=True)
    shuffle(xml_files)
    print("total xml file: ", len(xml_files))
    image_files = list(map(lambda i: i.replace("xml", "jpg") if not image_path \
        else os.path.join(image_path, os.path.basename(i).replace("xml", "jpg")), xml_files))
    xml_files, image_files = zip(
        *[(xml_files[i], image_files[i]) for i in range(len(image_files)) if os.path.exists(image_files[i])])
    xml_files, image_files = list(xml_files), list(image_files)
    print("valid xml file: ", len(xml_files))
    if FLAGS.label_map_json_path:
        with tf.io.gfile.GFile(FLAGS.label_map_json_path, 'rb') as f:

            label_map_dict = json.load(f)
        auto_label_map = False
        print(label_map_dict)
    else:
        label_map_dict = {}
        auto_label_map = True

    ann_json_dict = {
        'images': [],
        'type': 'instances',
        'annotations': [],
        'categories': []
    }

    for class_name, class_id in label_map_dict.items():
        cls = {'supercategory': 'none', 'id': class_id, 'name': class_name}
        ann_json_dict['categories'].append(cls)

    auto_label_index = -1
    # 只有线程数能被shard整除时和指定类别字典时才允许多线程并发，
    if FLAGS.num_threads and FLAGS.num_threads > 1 and FLAGS.label_map_json_path and (
            FLAGS.num_shards == FLAGS.num_threads):
        logging.info("多线程处理")
        from copy import deepcopy
        intervals = len(xml_files) // FLAGS.num_threads
        files_indexes_all = [(i * intervals, (i + 1) * intervals) if i < (FLAGS.num_threads - 1) else (
            i * intervals, max((i + 1) * intervals, len(xml_files))) for i in range(FLAGS.num_threads)]
        print(files_indexes_all)
        total_number = len(image_files)
        threads = [None] * FLAGS.num_threads
        for i in range(FLAGS.num_threads):
            threads[i] = threading.Thread(target=batch_processing,
                                          args=(i,
                                                deepcopy(xml_files[files_indexes_all[i][0]:files_indexes_all[i][1]]),
                                                deepcopy(image_files[files_indexes_all[i][0]:files_indexes_all[i][1]]),
                                                deepcopy(label_map_dict), auto_label_map, deepcopy(ann_json_dict),
                                                auto_label_index, total_number
                                                ))

        for thread in threads:
            thread.start()
        for thread in threads: thread.join()
    else:
        logging.info("单线程处理")
        writers = [
            tf.python_io.TFRecordWriter(os.path.join(FLAGS.output_path, FLAGS.prefix + '-%05d-of-%05d.tfrecord' %
                                                     (i, FLAGS.num_shards)))
            for i in range(FLAGS.num_shards)
        ]

        # Todo多进程数据共享的问题
        pool = multiprocessing.Pool(FLAGS.num_threads)
        for idx in range(len(xml_files)):
            if FLAGS.num_images and idx >= FLAGS.num_images:
                break
            if idx % 100 == 0:
                logging.info('On image %d of %d', idx, len(xml_files))
            path = xml_files[idx]
            with tf.gfile.GFile(path, 'r') as fid:
                xml_str = fid.read()
            try:
                xml = etree.fromstring(xml_str)
            except ValueError:
                xml = etree.fromstring(xml_str.encode("utf-8"))
            # 新增
            if not tf.io.is_jpeg(tf.io.read_file(image_files[idx])):
                print("图片编码错误！！", image_files[idx])
                continue
            data = tfrecord_util.recursive_parse_xml_to_dict(xml)['annotation']
            try:
                tf_example, auto_label_index = dict_to_tf_example(
                    data, image_files[idx],
                    label_map_dict,
                    auto_label_map=auto_label_map,
                    auto_label_index=auto_label_index,
                    ignore_difficult_instances=FLAGS.ignore_difficult_instances,
                    ann_json_dict=ann_json_dict)
            except ZeroDivisionError:
                logging.warning("ZeroDivisionError: " + path)
                continue
            except AssertionError as e:
                logging.warning("AssertionError: " + path + str(e))
                continue
            writers[idx % FLAGS.num_shards].write(tf_example.SerializeToString())

        for writer in writers:
            writer.close()

        json_file_path = os.path.join(FLAGS.output_path, 'label2index.json')
        if auto_label_map:
            with tf.io.gfile.GFile(json_file_path, 'w') as f:
                json.dump(label_map_dict, f)
    print("-----------Convert time cost(second):", time.time() - st)


if __name__ == '__main__':
    app.run(main)
