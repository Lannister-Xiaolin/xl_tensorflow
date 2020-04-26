#!usr/bin/env python3
# -*- coding: UTF-8 -*-
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import colorsys


def draw_boxes_cv(image, boxes, scores, labels, colors, classid2label):
    """
    Args:
        image: Numpy array,   RGB
        boxes: List of array or Array, format: X1,Y1,X2,Y2
        scores: List or Array
        labels:   List or Array
        colors:   List or Array
        class2label:   Dict, format: {id:label}
    Returns:
        new image array
    """
    image = image.copy()
    if not colors:
        colors = [np.random.randint(0, 256, 3).tolist() for _ in range(len(classid2label))]
    for b, l, s in zip(boxes, labels, scores):
        class_id = int(l)
        class_name = classid2label[class_id]
        xmin, ymin, xmax, ymax = list(map(int, b))
        score = '{:.4f}'.format(s)
        color = colors[class_id]
        label = '-'.join([class_name, score])
        ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)
        cv2.rectangle(image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
        cv2.putText(image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return image


def draw_boxes_pil(image, out_boxes, out_scores, out_classes, classid2label):
    if type(image) == np.ndarray:
        image = Image.fromarray(image)
    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                              size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300
    hsv_tuples = [(x / len(classid2label), 1., 1.) for x in range(len(classid2label))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = classid2label[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        left, top, right, bottom = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=colors[c])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw

    return np.array(image)
setattr(draw_boxes_pil,"__doc__", draw_boxes_pil.__doc__)