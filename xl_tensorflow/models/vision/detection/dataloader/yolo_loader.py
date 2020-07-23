#!usr/bin/env python3
# -*- coding: UTF-8 -*-
from PIL import Image
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from math import ceil
import tensorflow as tf


def letterbox_image(image, size):
    '''
    不改变长宽比
    resize image with unchanged aspect ratio using padding
    '''
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (0, 0, 0))
    new_image.paste(image, (0, 0))
    return new_image


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    '''
    Preprocess true boxes to training input format
    所有box会定位到指定的grid
    Args：
        true_boxes: array, shape=(m, T, 5),绝对值
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
        input_shape: array-like, hw, multiples of 32
        anchors: array, shape=(N, 2), 2 refer to wh, N refer to number of achors
        num_classes: integer

    Returns

        y_true:
            list of array, shape like yolo_outputs, xywh are reletive value
            即相对值，相对整图比例， y_true 形状通常为 [array(1,26,26,3,85),]

            一个box只会对应一个尺度的一个grid, 尺度的选择根据与anchor box的iou来定
                首先计算box与9个anchor的iou，计算最高iou的anchorbox，选择该anchor box作为负责预测的anchor
                ,根据anchor索引和坐标定位到相应的grid
    '''
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors) // 3  # default setting
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    m = true_boxes.shape[0]
    grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
    y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes),
                       dtype='float32') for l in range(num_layers)]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0: continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes
        # 所有box都会与anchor进行对比，
        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box，此处已经确定对应最好的anchorbox了
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                # 一个box只会对应一个尺度一个grid, 尺度的选择根据与anchor box的iou来定
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                    # 单个grid最多允许三个box，多余的会被覆盖，根据anchor确定位置
                    # 即如果两个框对应同一个anchor,且位置相近的话，有一个会被覆盖
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b, t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    # real object confidence
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5 + c] = 1

    return tuple(y_true)


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def get_random_data(annotation_line, input_shape, random=True,
                    max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True, seperator="\t"):
    '''
    random preprocessing for real-time data augmentation
        1、所有图片都会resize到小于416以下，不足416则粘贴到背景为128的图片中，并归一化（/255）
        2、
    '''
    line = annotation_line.split(seperator)
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape
    box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

    if not random:
        # resize image
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2
        image_data = 0
        if proc_img:
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image) / 255.

        # correct boxes
        box_data = np.zeros((max_boxes, 5))
        if len(box) > 0:
            np.random.shuffle(box)
            if len(box) > max_boxes: box = box[:max_boxes]
            box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
            box[:, [1, 3]] = box[:, [1, 3]] * scale + dy
            box_data[:len(box)] = box

        return image_data, box_data

    # resize image
    new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)

    # place image
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # flip image or not
    flip = rand() < .5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = rgb_to_hsv(np.array(image) / 255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    image_data = hsv_to_rgb(x)  # numpy array, 0 to 1

    # correct boxes
    box_data = np.zeros((max_boxes, 5))
    if len(box) > 0:
        np.random.shuffle(box)
        box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
        box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
        if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
        if len(box) > max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box

    return image_data, box_data


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, seperate_y=True, seperator="\t"):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True, seperator=seperator)
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        if seperate_y:
            yield image_data, y_true
        else:
            yield (image_data, *y_true), np.zeros(batch_size)


def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes, seperate_y=True,
                           seperator=" "):
    annotation_lines = [i for i in annotation_lines if i.strip()]
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, seperate_y,
                          seperator=seperator)


def create_datagen(train_annotation_path, val_annotation_path, batch_size, input_shape, anchors, num_classes,
                   seed=100, seperator="\t", generater2tfdata=False, number_tfdata=4):
    with open(train_annotation_path, encoding="utf-8") as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding="utf-8") as f:
        val_lines = f.readlines()
    num_train = int(len(train_lines))
    np.random.seed(seed)
    np.random.shuffle(train_lines)
    num_val = len(val_lines)
    if not generater2tfdata:
        train = data_generator_wrapper(train_lines, batch_size, input_shape, anchors, num_classes, seperator=seperator)
        val = data_generator_wrapper(val_lines, batch_size, input_shape, anchors,
                                     num_classes, seperator=seperator)
        return train, val, num_train, num_val
    else:
        # Todo 暂未完成
        samples_per_tfdata_train = ceil(num_train / number_tfdata)
        samples_per_tfdata_val = ceil(num_train / number_tfdata)
        # trains = [data_generator_wrapper(train_lines[i * samples_per_tfdata_train:(i + 1) * samples_per_tfdata_train],
        #                                  batch_size, input_shape, anchors, num_classes, seperator=seperator) for i in
        #           range(number_tfdata)]
        # vals = [data_generator_wrapper(val_lines[i * samples_per_tfdata_val:(i + 1) * samples_per_tfdata_val],
        #                                batch_size, input_shape, anchors, num_classes, seperator=seperator) for i in
        #         range(number_tfdata)]
        # # output_shapes = ((tf.TensorShape((batch_size, 416, 416, 3)), tf.TensorShape((batch_size, 13, 13, 3, 20)),
        # #                   tf.TensorShape((batch_size, 26, 26, 3, 20)),
        # #                   tf.TensorShape((batch_size, 52, 52, 3, 20))), tf.TensorShape(batch_size))
        # output_types = (tf.float32, (tf.float32, tf.float32, tf.float32))
        # train_datasets = [tf.data.Dataset.from_generator(lambda: dataset, output_types=output_types) for dataset in
        #                   trains]
        # val_datasets = [tf.data.Dataset.from_generator(lambda: dataset, output_types=output_types) for dataset in
        #                   vals]
        # tf.data.Dataset.interleave()
