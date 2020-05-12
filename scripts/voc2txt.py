#!usr/bin/env python3
# -*- coding: UTF-8 -*-
import click
from xl_tool.xl_io import file_scanning, read_txt
from xl_tensorflow.preprocessing.annotation import voc2txt_annotation


@click.command()
@click.option("--xml", help="xml path to read")
@click.option("--img", help="image path to read")
@click.option("--txt", help="txt to save ")
@click.option("--label", help="label file to read, seperate by \n")
def main(xml, txt, label, img):
    xml_files = file_scanning(xml, file_format="xml", sub_scan=True)
    train_txt = txt
    classes = read_txt(label, return_list=True, remove_linebreak=True)
    voc2txt_annotation(xml_files, train_txt, classes, image_path=img, seperator="\t", encoding="utf-8")
