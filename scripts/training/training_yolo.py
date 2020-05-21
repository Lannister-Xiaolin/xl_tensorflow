#!usr/bin/env python3
# -*- coding: UTF-8 -*-
from absl import app
from absl import flags
from absl import logging
flags.DEFINE_string('data_dir', '', 'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
                                    'merged set.')
# todo 待完善
from xl_tensorflow.models.vision.detection.training.yolo_training import *
from xl_tensorflow.models.vision.detection.body.yolo import *
def main():
    mul_gpu_training_custom_data("/tf/data/VOC_temp/train/*.tfrecord",val_annotation_path="/tf/data/VOC_temp/val/*.tfrecord",
                             classes_path="",batch_size=8,iou_loss="ciou",num_classes=20,lrs=(1e-05, 0.0001, 0.0001),
                             input_shape=(416,416),architecture="yolov3",suffix="voc_ciou",tfrecord=True,epochs=(2, 12, 20),
                             pre_weights="/tf/object_detection/yolov3_xl_weights.h5",workers=1)
if __name__ == '__main__':
    app.run(main)