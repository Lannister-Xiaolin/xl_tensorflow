#!usr/bin/env python3
# -*- coding: UTF-8 -*-
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import colorsys


def draw_boxes_cv(image, out_boxes, out_scores, out_classes, colors, index2label):
    """
    Args:
        image: Numpy array,   RGB
        out_boxes: List of array or Array, format: X1,Y1,X2,Y2
        out_scores: List or Array
        out_classes:   List or Array
        colors:   List or Array
        index2label:   Dict, format: {id:label}
    Returns:
        new image array
    """
    image = image.copy()
    if not colors:
        colors = [np.random.randint(0, 256, 3).tolist() for _ in range(len(index2label))]
    for box, l, s in zip(out_boxes, out_classes, out_scores):
        class_id = int(l)
        class_name = index2label[class_id]
        xmin, ymin, xmax, ymax = list(map(int, box))
        # score = '{:.4f}'.format(s)
        color = colors[class_id]
        # label = '-'.join([class_name, score])
        label = '{} {:.2f}'.format(class_name, s)
        ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)
        cv2.rectangle(image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
        cv2.putText(image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return image


def draw_boxes_pil(image, out_boxes, out_scores, out_classes, index2label):
    if type(image) == np.ndarray:
        image = Image.fromarray(image)
    try:
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    except:
        try:
            font = ImageFont.truetype('arial.ttf', 24)
        except:
            font = ImageFont.load_default()

    thickness = (image.size[0] + image.size[1]) // 300
    hsv_tuples = [(x / len(index2label), 1., 1.) for x in range(len(index2label))]
    colors = list(map(lambda x: tuple([int(i * 255) for i in colorsys.hsv_to_rgb(*x)]), hsv_tuples))
    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = index2label[c]
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
        draw.text(tuple(text_origin), label, fill=(0, 0, 0), font=font)
        del draw

    return np.array(image)


setattr(draw_boxes_pil, "__doc__", draw_boxes_pil.__doc__)
