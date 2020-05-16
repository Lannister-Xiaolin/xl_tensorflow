from xl_tensorflow.models.vision.detection.training.yolo_training import *
model = single_inference_model_serving("yolov4",
                                       "",
                                       20,dynamic_shape=True,score_threshold=0.1, return_xy=False)