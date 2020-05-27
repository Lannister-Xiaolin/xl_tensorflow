from xl_tensorflow.models.vision.detection.inference.yolo_inference import *
import base64
from xl_tool.xl_io import file_scanning
def test_bs64():
    model = yolo_inference_model("yolov3","",20,b64_mode=True,b64_shape_decode=True)
    # print(model.summary())
    # print(model.inputs)
    # print(model.outputs)
    #
    print(model.predict(np.array([[base64.urlsafe_b64encode(open(i,"rb").read())] for i in file_scanning(r"E:\Temp\test","jpg",sub_scan=False)])))

def test_shapeinput():
    model = yolo_inference_model("yolov3","",20,b64_mode=False,b64_shape_decode=False)
    print(model.predict([np.random.randn(2,416,416,3), np.array([[416,523],[562,525]])]))
test_shapeinput()