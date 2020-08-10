from xl_tensorflow.models.vision.detection.inference.yolo_inference import *
import base64
from xl_tool.xl_io import file_scanning
def test_bs64():
    weights = r"E:\Programming\Python\TOOL\weights\yolo/xl_yolov3_weights.h5"
    model1 = yolo_inference_model("yolov3",weights,80,mean=None,std=None)
    model2 = single_inference_model("yolov3",weights,80,dynamic_shape=False)
    data = np.random.random((1,416,416,3))
    print(model1.inputs)
    print(model1.predict([data,np.array(data.shape).astype(np.float)]))
    print(model2.predict(data))

    # print(model.summary())
    # print(model.inputs)
    # print(model.outputs)
    #
    # print(model.predict(np.array([[base64.urlsafe_b64encode(open(i,"rb").read())] for i in file_scanning(r"E:\Temp\test","jpg",sub_scan=False)])))

def test_shapeinput():
    model = yolo_inference_model("yolov3","",20,b64_mode=False,b64_shape_decode=False)
    print(model.predict([np.random.randn(2,416,416,3), np.array([[416,523],[562,525]])]))
# test_shapeinput()
test_bs64()