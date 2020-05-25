#!usr/bin/env python3
# -*- coding: UTF-8 -*-
import numpy as np

YOLOV3_ANCHORS = np.array([[10., 13.],
                           [16., 30.],
                           [33., 23.],
                           [30., 61.],
                           [62., 45.],
                           [59., 119.],
                           [116., 90.],
                           [156., 198.],
                           [373., 326.]], dtype="float")

YOLOV4_ANCHORS = np.array([[12, 16],
                           [19, 36],
                           [40, 28],
                           [36, 75],
                           [76, 55],
                           [72, 146],
                           [142, 110],
                           [192, 243],
                           [459, 401]], dtype="float")
