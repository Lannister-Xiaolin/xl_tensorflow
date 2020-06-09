#!usr/bin/env python3
# -*- coding: UTF-8 -*-
from xl_tensorflow.models.vision import EfficientNetB1
from xl_tool.xl_io import read_json, save_to_json
import xl_tensorflow.models.vision.detection.configs.factory as config_factory
from xl_tensorflow.models.vision.detection.architecture.fpn import BiFpn
from xl_tensorflow.models.vision import EfficientNetB0
from xl_tensorflow.models.vision.detection.body.efficientdet_model import EfficientDetModel
import numpy as np
from tensorflow.keras import layers, Model
from xl_tensorflow.models.vision.detection.dataloader.input_reader import InputFn
from xl_tensorflow.models.vision.detection.inference.efficientdet_inference import det_post_process_combined, \
    batch_image_preprocess


def model_test():
    params = config_factory.config_generator("efficientdet-d1")

    model_fn = EfficientDetModel(params)
    model = model_fn.build_model(params)
    model.load_weights(
        r"E:\Programming\Python\TOOL\weights\efficientnet\efficientnet-b1_weights_tf_dim_ordering_tf_kernels.h5",
        by_name=True, skip_mismatch=True)
    data = np.random.random((2, 640, 640, 3))
    print(model(data, training=True)['cls_outputs'])
    print(model(data, training=False)['cls_outputs'])


def training_test():
    params = config_factory.config_generator("efficientdet-d1")
    params.architecture.num_classes = 21
    input_reader = InputFn(r"E:\Temp\test\tfrecord\*.tfrecord", params, "train", 1)
    data = input_reader(batch_size=4)
    model_fn = EfficientDetModel(params)
    model = model_fn.build_model(params)
    next(data.as_numpy_iterator())
    def loss_fn(labels, outputs):
        return model_fn.build_loss_fn()(labels, outputs)['total_loss']

    # loss_fn = model_fn.build_loss_fn()
    # todo keras model调用loss_fn是对应每个输出，而不是所有输出（多输出注意如何使用），见training.py 1611行
    # todo  dict嵌套模式不适合使用 compile(loss='')的形式
    model.compile(loss=loss_fn)
    model.fit(data)
    temp = next(data.as_numpy_iterator())


def inference_test():
    params = config_factory.config_generator("efficientdet-d1")
    params.architecture.num_classes = 21
    model_fn = EfficientDetModel(params)
    model = model_fn.build_model(params)
    from PIL import Image
    import glob
    data = [np.array(Image.open(file)) for file in glob.glob(r"E:\Temp\test\image\a\*.jpg")]
    print([i.shape  for i in data])
    images, scales = batch_image_preprocess(data, (640, 640),batch_size=2)
    print(np.array(images[0]).mean(axis=-1),scales)
    outputs = model(images)
    #
    det_post_process_combined(params, scales=scales, min_score_thresh=0.2, max_boxes_to_draw=4, **outputs)
    print("测试通过，暂未发现异常")


if __name__ == '__main__':
    # inference_test()
    training_test()