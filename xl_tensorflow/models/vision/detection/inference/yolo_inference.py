#!usr/bin/env python3
# -*- coding: UTF-8 -*-
import pathlib

from tensorflow.keras import Input, Model
from ..body.yolo import yolo_body, yolo_eval
from xl_tensorflow.models.vision.detection.dataloader.utils.anchors_yolo import YOLOV4_ANCHORS, YOLOV3_ANCHORS
from ..dataloader.yolo_loader import letterbox_image
from ..loss.yolo_loss import YoloLoss
from ..dataloader.yolo_loader import get_classes, create_datagen
import tensorflow as tf
import os
import shutil
from PIL import Image
import matplotlib.pyplot as plt
from xl_tool.xl_io import read_json
import numpy as np
from xl_tensorflow.metrics.rafaelpadilla.Evaluator import voc2ratxt, mao_raf_from_txtfile
from xl_tensorflow.models.vision.detection.utils.drawing import draw_boxes_pil


def single_inference_model(model_name, weights,
                           num_classes,
                           origin_image_shape=(416, 416),
                           input_shape=(416, 416),
                           anchors="v3",
                           score_threshold=.1,
                           iou_threshold=.5,
                           max_detections=20,
                           dynamic_shape=False, return_xy=True):
    """
    用于部署在serving端的模型，固定输入尺寸和图片尺寸，会对iou值和置信度进行过滤0.1
    Args:
        model_name: string must be of of following:
                    "yolov3 yolov4 yolov3-spp yolov4-efficientnetb0"
        origin_image_shape: 高*宽
        weights:
        num_classes:
        dynamic_shape：是否允许将图片尺寸作为动态输入，
        return_xy:是否范围xy格式，默认yx格式
    Returns:
        tf.keras.Model object, 预测图片的绝对值坐标x1,y1,x2,y2
    """
    anchors = YOLOV4_ANCHORS if anchors == "v4" else YOLOV3_ANCHORS
    yolo_model = yolo_body(Input(shape=(*input_shape, 3)),
                           len(anchors) // 3, num_classes, model_name, reshape_y=True)

    if weights:
        yolo_model.load_weights(weights)
    if dynamic_shape:
        shape_input = Input(shape=(2,))
        boxes_, scores_, classes_ = yolo_eval(yolo_model.outputs,
                                              anchors, num_classes, shape_input, max_detections,
                                              score_threshold,
                                              iou_threshold, return_xy=return_xy)
        model = Model(inputs=yolo_model.inputs + [shape_input], outputs=(boxes_, scores_, classes_))
    else:
        boxes_, scores_, classes_ = yolo_eval(yolo_model.outputs,
                                              anchors, num_classes, origin_image_shape, max_detections,
                                              score_threshold,
                                              iou_threshold, return_xy=return_xy)
        model = Model(inputs=yolo_model.inputs, outputs=(boxes_, scores_, classes_))

    return model


def tflite_export_yolo(model_name, num_classes, save_lite_file, weights="", input_shape=(416, 416), anchors="v3",
                       return_xy=True, score_threshold=.2,
                       iou_threshold=.5, quant="", int_quantize_sample=(100, 416, 416, 3)):
    """
    模型输入为固定尺寸，因此输出需要根据与固定尺寸的比例进行缩放和偏置（如过是右侧填充则不需要，居中两侧填充为）
    输出按照xyxy格式
    Args:
        model_name:
        num_classes:
        save_lite_file:
        weights:
        input_shape:
        anchors:
        return_xy:
        score_threshold:
        iou_threshold
        quant ：是否使用int8量化
        int_quantize_sample:量化评估数据
    Returns:

    """
    anchors = YOLOV4_ANCHORS if anchors == "v4" else YOLOV3_ANCHORS
    from tensorflow.keras import layers
    inputs = layers.Input(shape=(416, 416, 3))
    x = tf.multiply(inputs, 1 / 255.0)
    yolo_model = yolo_body(Input(shape=(*input_shape, 3)),
                           len(anchors) // 3, num_classes, model_name, reshape_y=False)
    if weights:
        yolo_model.load_weights(weights)
    boxes_, scores_ = yolo_eval(yolo_model(x),
                                anchors, num_classes, input_shape, 20,
                                score_threshold,
                                iou_threshold, return_xy=return_xy, lite_return=True)
    model = Model(inputs=inputs, outputs=[boxes_, scores_])
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # todo 量化暂时不支持LEAKY_RELU
    if quant == "int8":
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        images = np.random.random(int_quantize_sample).astype("float32")
        mnist_ds = tf.data.Dataset.from_tensor_slices((images)).batch(1)

        def representative_data_gen():
            for input_value in mnist_ds.take(100):
                yield [input_value]

        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    elif quant == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    else:
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    pathlib.Path(save_lite_file).write_bytes(converter.convert())
    return model


# todo 待新增，暂无需求
def seving_export_yolo():
    """
    接收base64 / resize后的数组（无需预处理模型）
    Returns:

    """
    pass


def yolo_inferece(image_files, output_dir, model_name, weights,
                  num_classes,
                  origin_image_shape=(416, 416),
                  input_shape=(416, 416),
                  anchors="v3",
                  score_threshold=.2,
                  iou_threshold=.5,
                  max_detections=20,
                  dynamic_shape=False, return_xy=True,
                  map_evaluate=False,
                  xml_files="", map_save="./map_evaluate", visual_one=False,
                  label2index_file="",
                  save_result=True,
                  ):
    print("加载模型中.....")
    model = single_inference_model(model_name=model_name, weights=weights,
                                   num_classes=num_classes,
                                   origin_image_shape=origin_image_shape,
                                   input_shape=input_shape,
                                   anchors=anchors,
                                   score_threshold=score_threshold,
                                   iou_threshold=iou_threshold,
                                   max_detections=max_detections,
                                   dynamic_shape=dynamic_shape, return_xy=return_xy)

    index2label = {v: k for k, v in read_json(label2index_file).items()}
    class_names = list(index2label.values())
    gt_path = ""
    dt_path = ""
    os.makedirs(output_dir, exist_ok=True)
    if map_evaluate:
        print("解析标注文件.....")
        gt_path = os.path.join(map_save, "gt_path")
        dt_path = os.path.join(map_save, "dt_path")
        try:
            shutil.rmtree(gt_path)
            shutil.rmtree(dt_path)
        except:
            pass
        os.makedirs(gt_path, exist_ok=True)
        os.makedirs(dt_path, exist_ok=True)
        print(len(xml_files) == len(image_files))
        for xml_file in xml_files:
            try:
                bndboxes = voc2ratxt(xml_file, box_format="xyxy")
            except:
                print(xml_file)
                continue
            with open(f"{gt_path}/{os.path.basename(xml_file).split('.')[0]}.txt", "w") as f:
                f.write("\n".join([" ".join([str(j) for j in i]) for i in bndboxes[1] if i[0] in class_names]))
    from tqdm import tqdm
    pbar = tqdm(image_files)
    for image_file in pbar:
        image = Image.open(image_file)
        basename = os.path.basename(image_file)
        image_id = os.path.basename(image_file).split('.')[0]
        boxed_image = letterbox_image(image, (416, 416))
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)
        boxes_, scores_, classes_ = model.predict([image_data, np.array([[*image.size][::-1]])])
        boxes_, scores_, classes_ = boxes_[0], scores_[0], classes_[0]
        if len(scores_) > 0:
            dt_boxes = []
            if map_evaluate:
                for i in range(len(scores_)):
                    dt_boxes.append(
                        f"{index2label[classes_[i]]} {scores_[i]:.2f} {int(boxes_[i][0])} {int(boxes_[i][1])} {int(boxes_[i][2])} {int(boxes_[i][3])}")
                with open(f"{dt_path}/{image_id}.txt", "w") as f:
                    f.write("\n".join(dt_boxes))
            if save_result:
                image = draw_boxes_pil(image, boxes_.tolist(), scores_.tolist(), classes_.tolist(), index2label)
                Image.fromarray(image).save(f"{output_dir}/{basename}")
            print(("\n".join(
                [(str((boxes_[i].tolist())) + "\t" + index2label[(np.array(classes_)[i])] + "\t" + str(
                    np.array(scores_)[i])) for i in
                 range(len(boxes_))])))
            if visual_one and save_result:
                from IPython import display
                display.display(display.Image(f"{output_dir}/{basename}"))
        else:
            print("no box detected: ", basename)
            pass
    if map_evaluate:
        map50, metrics_per_classes, map_str = mao_raf_from_txtfile(gt_path, dt_path)
        with open(f"{map_save}/map_result.txt", "w") as f:
            f.write(map_str)
