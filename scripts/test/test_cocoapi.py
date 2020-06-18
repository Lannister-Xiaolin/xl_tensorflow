#!usr/bin/env python3
# -*- coding: UTF-8 -*-
from xl_tensorflow.models.vision.detection.evaluation import coco_utils,coco_evaluator
import numpy as np
from pycocotools import cocoeval
gt_dataset = {"images": [{"id": 2, "height": 480, "width": 640}],
              "categories": [{"id": 1}, {"id": 2}],
              "annotations": [{
                  "image_id": 2,
                  "iscrowd": 0,
                  "category_id": 1,
                  "bbox": [
                      458.0,
                      242.0,
                      115.0,
                      118.0
                  ],
                  "area": 0.04417317733168602, "id": 1},
                  {
                      "image_id": 2,
                      "iscrowd": 0,
                      "category_id": 2,
                      "bbox": [
                          269.0,
                          221.0,
                          97.0,
                          104.0
                      ],
                      "area": 0.03283853456377983,
                      "id": 2
                  }
              ]
              }
coco_gt = coco_utils.COCOWrapper(
    eval_type='box',
    gt_dataset=gt_dataset)
coco_predictions = [
    {'image_id': 2, 'category_id': 1, 'bbox': np.array([458.00, 242.0, 115.0, 118.0], dtype="float32"), 'score': 1.0,
     'id': 1}]
coco_dt = coco_gt.loadRes(predictions=coco_predictions)
image_ids = [ann['image_id'] for ann in coco_predictions]
coco_eval = cocoeval.COCOeval(coco_gt, coco_dt, iouType='bbox')
coco_eval.params.imgIds = image_ids
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
coco_metrics = coco_eval.stats