from xl_tensorflow.metrics.rafaelpadilla.Evaluator import map_raf_from_lists

# dt = [["1", [["person", 0.14, 1, 156, 103, 336], ["person", 0.14, 36, 111, 198, 416]]]]
# gt = [["1", [["bottle", 6, 234, 45, 362], ["person", 1, 156, 103, 336],
#              ["person", 36, 111, 198, 416], ["person", 91, 42, 338, 500]]]]
# print(map_raf_from_lists(dt,gt))

def mul_detect():
    import os
    from xl_tool.xl_io import  file_scanning,read_txt
    gt_files = file_scanning("./xl_tensorflow/metrics/rafaelpadilla/test/groundtruths","txt")
    dt_files = file_scanning("./xl_tensorflow/metrics/rafaelpadilla/test/detections","txt")
    dt = []
    gt = []
    for file in gt_files:
        temp = []
        temp.append(os.path.basename(file).split(".")[0])
        texts = read_txt(file,return_list=True)
        boxes = [[(float(j) if len(j)<4 else j) for j in i.split() ] for i in texts if i.strip()]
        temp.append(boxes)
        gt.append(temp)
    for file in dt_files:
        temp = []
        temp.append(os.path.basename(file).split(".")[0])
        texts = read_txt(file,return_list=True)
        boxes = [[(float(j) if len(j)<5 else j) for j in i.split() ] for i in texts if i.strip()]
        temp.append(boxes)
        dt.append(temp)
    (map_raf_from_lists(dt, gt,0.3))
mul_detect()