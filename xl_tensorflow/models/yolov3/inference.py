#!usr/bin/env python3
# -*- coding: UTF-8 -*-

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from xl_tensorflow.models.yolov3.utils import letterbox_image, draw_image
from xl_tensorflow.models.yolov3.model import yolo_body, tiny_yolo_body, \
    yolo_eval, yolo_eval_lite, \
    yolo_efficientnetliteb4_body, yolo_efficientnetliteb1_body
from tensorflow.keras import Input, Model
from xl_tool.xl_io import file_scanning


def draw_rectanngle(img, box, label, rectange_color=(0, 255, 0), label_color=(255, 0, 0)):
    img = img if type(img) not in (np.ndarray,) else Image.fromarray(img)
    top, left, bottom, right = box
    font = ImageFont.truetype(font='simsun.ttc',
                              size=np.floor(4e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300
    top = max(0, np.floor(top + 0.5).astype('int32'))
    left = max(0, np.floor(left + 0.5).astype('int32'))
    bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
    right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
    draw = ImageDraw.Draw(img)
    label_size = draw.textsize(label, font=font)
    if top - label_size[1] >= 0:
        text_origin = np.array([left, top - label_size[1]])
    else:
        text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
    for i in range(thickness):
        draw.rectangle(
            [left + i, top + i, right - i, bottom - i], outline=rectange_color)
    draw.rectangle(
        [tuple(text_origin), tuple(text_origin + label_size)], fill=rectange_color)
    draw.text(text_origin, label, fill=label_color, font=font)
    del draw
    return image


def _get_anchors(anchors_path):
    anchors_path = os.path.expanduser(anchors_path)
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def single_inference_model_lite(model_name, weights, num_classes, image_shape=(480, 640),
                                anchors=None,
                                input_shape=(416, 416),
                                score_threshold=.6,
                                iou_threshold=.5):
    """
    专门用于移动端处理，无5维tensor和掩码
    """
    # Todo 把iou和置信度,以及输入图片尺寸（高宽）， 写入模型
    body_dict = {"efficientnetliteb4": yolo_efficientnetliteb4_body,
                 "efficientnetliteb": yolo_efficientnetliteb1_body,
                 "darknet": yolo_body}
    if anchors == None:
        anchors = _get_anchors("./config/yolo_anchors.txt")
    yolo_model = body_dict[model_name](Input(shape=(*input_shape, 3)), len(anchors) // 3, num_classes)
    yolo_model.load_weights(weights)

    boxes_, scores_ = yolo_eval_lite(yolo_model.outputs, anchors, num_classes, image_shape, 20,
                                     score_threshold,
                                     iou_threshold)
    model = Model(inputs=yolo_model.inputs, outputs=(boxes_, scores_))
    return model


def single_inference_model_serving(model_name, weights,
                                   num_classes,
                                   image_shape=(416, 416),
                                   anchors=None,
                                   input_shape=(416, 416),
                                   score_threshold=.1,
                                   iou_threshold=.5):
    """
    用于部署在serving端的模型，固定输入尺寸和图片尺寸，会对iou值和置信度进行过滤0.1
    暂时不将尺寸和阙值写入模型，因此返回框的尺寸和位置需要根据图片进行重新调整（与resize方式有关）
    Args:
        image_shape: 宽高
    """
    # Todo 把iou和置信度,以及输入图片尺寸（高宽）， 写入模型
    body_dict = {"efficientnetliteb4": yolo_efficientnetliteb4_body,
                 "efficientnetliteb": yolo_efficientnetliteb1_body,
                 "darknet": yolo_body}
    if anchors == None:
        anchors = _get_anchors("./config/yolo_anchors.txt")
    yolo_model = body_dict[model_name](Input(shape=(*input_shape, 3)),
                                       len(anchors) // 3, num_classes)
    yolo_model.load_weights(weights)
    boxes_, scores_, classes_ = yolo_eval(yolo_model.outputs,
                                          anchors, num_classes, image_shape, 20,
                                          score_threshold,
                                          iou_threshold)
    model = Model(inputs=yolo_model.inputs, outputs=(boxes_, scores_, classes_))
    return model


def predict_image(model, image_file, input_shape=(416, 416), xy_order=False):
    # todo 计算公式中需要输入图片的尺寸不能直接用于部署，需要进行修改
    image = Image.open(image_file)
    boxed_image = letterbox_image(image, input_shape)
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    out_boxes, out_scores, out_classes = model.predict(image_data)
    try:
        if xy_order:
            out_boxes = [map(lambda x: [x[1, x[0], x[3], x[2]]], list(out_boxes))]
        else:
            out_boxes = list(out_boxes)
        out_scores, out_classes = list(out_scores), list(out_classes)
        return out_boxes, out_scores, out_classes
    except IndexError:
        return []


if __name__ == '__main__':

    num_classes = 15
    anchors = _get_anchors("./config/yolo_anchors.txt")
    files = file_scanning(r"F:\Download\qq\temp\test15\test15", file_format="jpg", sub_scan=True)
    import random
    from tqdm import tqdm

    random.shuffle(files)
    label_map = {0: 'bacon',
                 1: 'broccoli',
                 2: 'chicken_wing',
                 3: 'corn',
                 4: 'drumstick',
                 5: 'egg_tart',
                 6: 'flammulina_velutipes',
                 7: 'lamb_kebab',
                 8: 'pizza',
                 9: 'saury',
                 10: 'sausage',
                 11: 'steak',
                 12: 'sweet_potato',
                 13: 'tilapia',
                 14: 'toast'}

    unrecognized = 0
    # for file in tqdm(files[:100]):
    input_shape = (416, 416)
    model = single_inference_model_serving("darknet",
                                           r"F:\Download\yolo_darknet_weigths.h5",
                                           15, score_threshold=0.2,
                                           input_shape=input_shape, image_shape=(480, 640))
    count = [0, 0]
    # from xl_tensorflow.utils.deploy import serving_model_export, tf_saved_model_to_lite
    #
    # serving_model_export(single_inference_model_lite("darknet",
    #                                                  r"F:\Download\yolo_darknet_weigths.h5",
    #                                                  15, score_threshold=0.2,
    #                                                  input_shape=input_shape, image_shape=(480, 640)),
    #                      r"E:\Temp\test\yolo", auto_incre_version=False)
    # tf_saved_model_to_lite(r"E:\Temp\test\yolo\1", r"E:\Temp\test\yolo.tflite", input_shape=[None, *input_shape, 3])
    for i, file in enumerate(files[:100]):
        file = r"E:\Programming\android\AndroidTflite\app\src\main\assets\aaa.jpg"
        image = Image.open(file)
        boxed_image = letterbox_image(image, input_shape)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)

        # result = yolo_model.predict(image_data)
        # boxes_, scores_, classes_ = yolo_eval([tf.constant(result[0]), tf.constant(result[1]), tf.constant(result[2])],
        #                                       anchors,
        #                                       15, (480, 640))
        # boxes_, scores_, classes_ = np.array(boxes_), np.array(scores_), np.array(classes_)
        # if boxes_.shape[0] == 0:
        #     unrecognized += 1
        #     print(f"未检测到目标,总数{unrecognized}！！！\t", file)
        #
        #     continue
        # for i, box in enumerate(boxes_):
        #     box = np.array(box).astype(np.int)
        #     image = draw_rectanngle(image, box, label_map[classes_[i]])
        # image.show()
        # print(i+1)

        import time

        st = time.time()
        boxes_, scores_, classes_ = model.predict(image_data)
        print(boxes_,scores_,classes_)
        print(f"第 {i + 1} 个，预测时间{time.time() - st:.2f}")
        boxes_, scores_, classes_ = boxes_[0], scores_[0], classes_[0]
        if boxes_.shape[0] == 0:
            unrecognized += 1
            print(f"未检测到目标,总数{unrecognized}！！！\t", file)
            continue
        for i, box in enumerate(boxes_):
            box = np.array(box).astype(np.int)
            image = draw_rectanngle(image, box, label_map[classes_[i]])
        print(file)
        image.show()
        pass
