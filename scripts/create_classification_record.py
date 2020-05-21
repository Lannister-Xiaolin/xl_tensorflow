#!usr/bin/env python3
# -*- coding: UTF-8 -*-
import click
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from xl_tensorflow.models.vision.classification.utils.tfrecord import  images2tfrecord
@click.command()
@click.option("--path", help="images path to read")
@click.option("--tfrecord",  help="tfrecord filename(include path) to save ")
@click.option("--l2d",  help="label to index json file")
@click.option("--thread",  help="label to index json file")
def main(path,tfrecord,l2d):
    images2tfrecord(path,tfrecord,l2d)
main()