import logging
import os
import shutil
import xml.etree.ElementTree as ET
from xl_tool.xl_io import file_scanning
from tqdm import tqdm


def voc2txt_annotation(xml_files, train_txt, classes, image_path=None, seperator="\t", encoding="utf-8"):
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
    print(f"总文件数量：{len(xml_files)}\n训练文件存储位置：{train_txt}\n抽取类别：{'  '.join(classes)}")
    pbar = tqdm(xml_files)
    for xml_file in pbar:
        in_file = open(xml_file)
        tree = ET.parse(in_file)
        root = tree.getroot()
        if not root.find('object'):
            continue
        train_fp.write(xml_file.replace("xml", "jpg") if not image_path else os.path.join(image_path, os.path.basename(
            root.find("filename").text)))
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
                 int(float(xmlbox.find('xmax').text)),
                 int(float(xmlbox.find('ymax').text)))
            train_fp.write(seperator + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        train_fp.write("\n")
    pbar.set_description("转换进度：")
    train_fp.close()


def voc2voc_dataset(data_path, target_path, validation_split=None, cat_val=True, subset="train"):
    """
    Convert voc labeled data to VOC Dataset, files were placed into directories below:
        Annotations:
        ImageSets:
            Main:
        JPEGImages:
    Args:
        data_path: labeled image and xml path
        target_path: target path to save data
    Returns:
    """
    xml_files = file_scanning(data_path, "xml", sub_scan=True)
    if not xml_files:
        return
    xml_files, image_files = zip(*[(xml_file, xml_file.replace("xml", "jpg")) for xml_file in xml_files if
                                   os.path.exists(xml_file.replace("xml", "jpg"))])
    logging.info(f"Find labeled data: {len(xml_files)}")
    ImageSets, JPEGImages, Annotations = f"{target_path}/ImageSets/Main", f"{target_path}/JPEGImages", \
                                         f"{target_path}/Annotations"
    os.makedirs(ImageSets, exist_ok=True)
    os.makedirs(JPEGImages, exist_ok=True)
    os.makedirs(Annotations, exist_ok=True)
    for file in xml_files: shutil.copy(file, f"{Annotations}")
    for file in image_files: shutil.copy(file, f"{JPEGImages}")
    name_map_func = lambda x: os.path.basename(x).split(".")[0]
    if not validation_split:
        with open(f"{ImageSets}/{subset}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(map(name_map_func, xml_files)))
    else:
        if cat_val:
            cats = os.listdir(data_path)
            files = [list(map(name_map_func, [k for k in file_scanning(f"{data_path}/{d}", "xml", sub_scan=True) if
                                              os.path.exists(k.replace("xml", "jpg"))])) for d in cats]
            val_indexes = [int(len(j) * validation_split) for j in files]
            train_files = []
            val_files = []
            for i, cat_files in enumerate(files):
                with open(f"{ImageSets}/train_{cats[i]}.txt", "w", encoding="utf-8") as f:
                    f.write("\n".join(cat_files[:val_indexes[i]]))
                with open(f"{ImageSets}/val_{cats[i]}.txt", "w", encoding="utf-8") as f:
                    f.write("\n".join(cat_files[val_indexes[i]:]))
                with open(f"{ImageSets}/trainval_{cats[i]}.txt", "w", encoding="utf-8") as f:
                    f.write("\n".join(cat_files))
                train_files.extend(cat_files[:val_indexes[i]])
                val_files.extend(cat_files[val_indexes[i]:])
            with open(f"{ImageSets}/trainval.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(train_files + val_files))
            with open(f"{ImageSets}/val.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(val_files))
            with open(f"{ImageSets}/train.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(train_files))
        else:
            files = list(map(name_map_func, xml_files))
            val_index = int(len(xml_files) * validation_split)
            with open(f"{ImageSets}/trainval.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(files))
            with open(f"{ImageSets}/val.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(files[val_index:]))
            with open(f"{ImageSets}/train.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(files[:val_index]))


def voc_merge(voc_07, voc_12, target_path):
    images_07 = file_scanning(f"{voc_07}/JPEGImages", "jpg|jpeg", sub_scan=True)
    xmls_07 = file_scanning(f"{voc_07}/Annotations", "xml", sub_scan=True)
    images_12 = file_scanning(f"{voc_12}/JPEGImages", "jpg|jpeg", sub_scan=True)
    xmls_12 = file_scanning(f"{voc_12}/Annotations", "xml", sub_scan=True)
    os.makedirs(f"{target_path}/JPEGImages", exist_ok=True)
    os.makedirs(f"{target_path}/Annotations", exist_ok=True)
    os.makedirs(f"{target_path}/ImageSets/Main", exist_ok=True)
    for file in images_07 + images_12:
        shutil.copy(file, f"{target_path}/JPEGImages")
    for file in xmls_07 + xmls_12:
        shutil.copy(file, f"{target_path}/Annotations")
    from xl_tool.xl_io import read_txt
    train = read_txt(f"{voc_07}/ImageSets/Main/train.txt", return_list=True) + read_txt(
        f"{voc_12}/ImageSets/Main/train.txt", return_list=True)
    with open(f"{target_path}/ImageSets/Main/train.txt", "w") as f:
        f.write("\n".join(train))


def main():
    classes = os.listdir(r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\7_增强图片\single_pyramid")
    train_text = r"E:\Programming\Python\5_CV\学习案例\xl_tf2_yolov3\model_data\train.txt"
    path = r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\7_增强图片\single_pyramid"
    xml_files = [i for i in file_scanning(path, sub_scan=True, full_path=True, file_format="xml") if
                 os.path.exists(i.replace("xml", "jpg"))]
    voc2txt_annotation(xml_files, train_text, classes, seperator="\t")


if __name__ == '__main__':
    voc2voc_dataset(r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\1_真实场景\0_已标框",
                    r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\1_真实场景\voc",
                    validation_split=0.8, cat_val=True)
