#!usr/bin/env python3
# -*- coding: UTF-8 -*-
import click
import os
from xl_tool.xl_io import read_txt, file_scanning
from xl_tensorflow.preprocessing.annotation import voc2txt_annotation


@click.command()
@click.option("--voc", help="xml path to read")
@click.option("--train_txt", help="txt to save ")
@click.option("--val_txt", help="txt to save ")
def main(train_txt, val_txt, voc):
    class_names = """aeroplane
    bicycle
    bird
    boat
    bottle
    bus
    car
    cat
    chair
    cow
    diningtable
    dog
    horse
    motorbike
    person
    pottedplant
    sheep
    sofa
    train
    tvmonitor""".split()

    valid_file = set(read_txt(os.path.join(voc, "trainval/ImageSets/Main/train.txt"), return_list=True))
    xml_files = [i for i in file_scanning(os.path.join(voc, "trainval/JPEGImages"), sub_scan=True, full_path=True,
                                          file_format="xml") if
                 os.path.basename(i).split(".")[0] in valid_file]
    voc2txt_annotation(xml_files, train_txt, class_names, seperator="\t",
                       image_path=os.path.join(voc, "trainval/JPEGImages"))

    valid_file = set(read_txt(os.path.join(voc, "trainval/ImageSets/Main/val.txt"), return_list=True))

    xml_files = [i for i in file_scanning(os.path.join(voc, "trainval/JPEGImages"), sub_scan=True, full_path=True,
                                          file_format="xml") if
                 os.path.basename(i).split(".")[0] in valid_file]

    voc2txt_annotation(xml_files, val_txt, class_names, seperator="\t",
                       image_path=os.path.join(voc, "trainval/JPEGImages"))


if __name__ == '__main__':
    main()
