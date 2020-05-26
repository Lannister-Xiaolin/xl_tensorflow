#!usr/bin/env python3
# -*- coding: UTF-8 -*-
from absl import app
from absl import flags
from absl import logging

# todo
flags.DEFINE_string('train_file', '', 'train_file pattern like /temp/*.tfrecord')
flags.DEFINE_string('val_file', '', 'val_file pattern like /temp/*.tfrecord')
flags.DEFINE_integer('batch_size', 8, '')
flags.DEFINE_integer('num_classes', None, '')
flags.DEFINE_string('architecture', 'yolov3', 'yolov3 yolov4 ...')
flags.DEFINE_string('iou_loss', '', 'iou loss if not specified using mse')
flags.DEFINE_string('anchors', 'v3', 'anchor for training default using v3')
flags.DEFINE_integer('input_size', 416, 'input size')
flags.DEFINE_integer('initial_epoch', 0, '')
flags.DEFINE_float('ignore_thresh', 0.4, '')
flags.DEFINE_float('iou_scale', 1.0, '')
flags.DEFINE_string('pre_weights', '', 'weight to using')
flags.DEFINE_list("suffix", "voc", "suffix for saving dir")

flags.DEFINE_list("lrs", "1e-5, 1e-3, 1e-4", "")
flags.DEFINE_list("epochs", "2, 30, 50", "")
flags.DEFINE_list("paciences", "10, 10, 5", "")
flags.DEFINE_list("freeze_layers", "185, 185, 0", "")
flags.DEFINE_list("reduce_lrs", "3, 3, 3", "")

Flags = flags.FLAGS
# flags.DEFINE
# todo 待完善
from xl_tensorflow.models.vision.detection.training.yolo_training import *
from xl_tensorflow.models.vision.detection.body.yolo import *


def main(_):
    assert Flags.train_file and Flags.val_file, "训练文件不能为空"
    assert Flags.num_classes, "请指定类别数量"
    print(Flags.flags_into_string())
    mul_gpu_training_custom_data(Flags.train_file,
                                 val_annotation_path=Flags.val_file,
                                 classes_path="",
                                 batch_size=Flags.batch_size,
                                 iou_loss=Flags.iou_loss,
                                 num_classes=Flags.num_classes,
                                 lrs=[float(i) for i in Flags.lrs],
                                 input_shape=(Flags.input_size, Flags.input_size),
                                 architecture=Flags.architecture,
                                 suffix=Flags.suffix,
                                 anchors=Flags.anchors,
                                 freeze_layers=[int(i) for i in Flags.freeze_layers],
                                 initial_epoch=Flags.initial_epoch,
                                 tfrecord=True, epochs=[int(i) for i in Flags.epochs],
                                 pre_weights=Flags.pre_weights,
                                 ignore_thresh=Flags.ignore_thresh,
                                 iou_scale=Flags.iou_scale,
                                 paciences=[int(i) for i in Flags.paciences],
                                 reduce_lrs=[int(i) for i in Flags.reduce_lrs],
                                 workers=1)


if __name__ == '__main__':
    app.run(main)
