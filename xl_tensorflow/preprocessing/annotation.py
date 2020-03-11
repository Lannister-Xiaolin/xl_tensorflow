import os
import xml.etree.ElementTree as ET
from xl_tool.xl_io import file_scanning


def voc2txt_annotation(xml_files, train_txt, classes, image_path=None, seperator=" ", encoding="utf-8"):
    """
    Convert voc data to train.txt file, format as follows:
    One row for one image;
        Row format: image_file_path box1 box2 ... boxN;
        Box format: x_min,y_min,x_max,y_max,class_id (no space).
    Here is an example:
        path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
        path/to/img2.jpg 120,300,250,600,2
    Args:
        xml_files: voc labeled image
        train_txt: txt file for saving txt annotation
        seperator: seperator for filepath and box, default whitespace
        classes: object classes to extract
        image_path: image path, if none ,the image name and path must be the same with xml file
    Returns:

    """
    train_fp = open(train_txt, "w", encoding=encoding)
    for xml_file in xml_files:
        in_file = open(xml_file)
        tree = ET.parse(in_file)
        root = tree.getroot()
        if not root.find('object'):
            continue
        train_fp.write(xml_file.replace("xml", "jpg") if not image_path else os.path.join(image_path, os.path.basename(
            root.find("path").text)))
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text),
                 int(xmlbox.find('ymax').text))
            train_fp.write(seperator + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        train_fp.write("\n")
    train_fp.close()


def main():
    classes = os.listdir(r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\1_真实场景\0_已标框")

    train_text = r"E:\Programming\Python\5_CV\学习案例\xl_tf2_yolov3\model_data\train.txt"
    path = r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\1_真实场景\0_已标框"
    xml_files = [i for i in file_scanning(path, sub_scan=True, full_path=True, file_format="xml") if
                 os.path.exists(i.replace("xml", "jpg"))]
    voc2txt_annotation(xml_files, train_text, classes,seperator="\t")


if __name__ == '__main__':
    main()

