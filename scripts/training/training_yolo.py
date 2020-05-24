#!usr/bin/env python3
# -*- coding: UTF-8 -*-
from absl import app
from absl import flags
from absl import logging
# todo
flags.DEFINE_string('train_file', '', 'train_file pattern like /temp/*.tfrecord')
flags.DEFINE_string('val_file', '', 'val_file pattern like /temp/*.tfrecord')
flags.DEFINE_string('iou_loss', 'ciou', '')
flags.DEFINE_string('anchors', 'v3', '')
flags.DEFINE_string('architecture', 'yolov3', '')
flags.DEFINE_string('anchors', 'v3', '')
flags.DEFINE_string('anchors', 'v3', '')

flags.DEFINE_string('input_shape', 416, '')
flags.DEFINE_integer('num_classes', 80, '')
flags.DEFINE_integer('batch_size', 8, '')
flags.DEFINE_integer('initial_epoch',0, '')
flags.DEFINE_float('ignore_thresh', 0.4, '')
flags.DEFINE_float('iou_scale', 1.0, '')
flags.DEFINE_list("lrs", "1e-5, 1e-3, 1e-4", "")
flags.DEFINE_list("epochs", "2, 30, 50", "")
flags.DEFINE_list("paciences", "10, 10, 5", "")
flags.DEFINE_list("reduce_lrs", "3, 3, 3", "")
flags.DEFINE_list("reduce_lrs", "185, 185, 0", "")

# flags.DEFINE
# todo 待完善
from xl_tensorflow.models.vision.detection.training.yolo_training import *
from xl_tensorflow.models.vision.detection.body.yolo import *


def main():
    mul_gpu_training_custom_data("/tf/data/VOC_temp/train/*.tfrecord",
                                 val_annotation_path="/tf/data/VOC_temp/val/*.tfrecord",
                                 classes_path="",
                                 batch_size=8,
                                 iou_loss="ciou",
                                 num_classes=20,
                                 lrs=(1e-05, 0.0001, 0.0001),
                                 input_shape=(416, 416),
                                 architecture="yolov3", suffix="voc_ciou",
                                 tfrecord=True, epochs=(2, 12, 20),
                                 pre_weights="/tf/object_detection/yolov3_xl_weights.h5",
                                 workers=1)


if __name__ == '__main__':
    app.run(main)
