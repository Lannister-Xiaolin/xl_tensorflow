from xl_tensorflow.models.vision.detection.inference.yolo_inference import *
from xl_tensorflow.models.vision.detection.inference.efficientdet_inference import *
import base64
from xl_tool.xl_io import file_scanning


def test_yolo():
    # model = yolo_inference_model("yolov3", "", 20, inference_mode="base64",
    #                              serving_export=True, serving_path=r"E:\Temp\test\serving\yolov3")
    #
    tflite_export_yolo("yolov3", 20,r"E:\Temp\test\serving\yolov3.tflite")
    # print(model.predict(np.array([[base64.urlsafe_b64encode(open(i, "rb").read())] for i in
    #                               file_scanning(r"E:\Temp\test", "jpg", sub_scan=False)])))


def test_efficientdet():
    model, lite_model = efficiendet_inference_model(inference_mode="dynamic",
                                                serving_export=False,
                                                serving_path=r"E:\Temp\test\serving\efficiendetd0", )
    print(lite_model.outputs)
    # model.save_weights("./fuck.h5")

    # images = np.random.random((10,512,512,3)).astype("float32")
    # mnist_ds = tf.data.Dataset.from_tensor_slices((images)).batch(1)
    #
    # def representative_data_gen():
    #     for input_value in mnist_ds.take(100):
    #         yield [input_value]


    converter = tf.lite.TFLiteConverter.from_keras_model(lite_model)
    # converter.representative_dataset = representative_data_gen
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # converter.inference_input_type = tf.uint8
    # converter.inference_output_type = tf.uint8
    # tflite_model = converter.convert()
    import pathlib

    #
    pathlib.Path(r"E:\Temp\test\serving\efficientdetd0.tflite").write_bytes(converter.convert())
    # data = np.random.randn(1, 640, 480, 3)

def test_shapeinput():
    model = yolo_inference_model("yolov3", "", 20, b64_mode=False, b64_shape_decode=False,
                                 serving_export=True, serving_path=r"E:\Temp\test\incepcion")
    # print(model.summary())
    print(model.inputs)
    print(model.outputs)
    print(model.predict([np.random.randn(2, 416, 416, 3), np.array([[416, 523], [562, 525]])]))


# test_shapeinput()
test_yolo()
# test_efficientdet()
