#!usr/bin/env python3
# -*- coding: UTF-8 -*-
from xl_tensorflow.models.vision.detection.dataloader.input_reader import YoloInputFn
import logging
import click

logging.info("help me")
@click.command()
@click.option("--record", help="record file pattern")
def input_reader_test(record):
    inputfn = YoloInputFn((416,416), record,aug_scale_max=1.5,aug_scale_min=0.8,num_classes=20)
    dataset = inputfn(4)
    print(next(dataset.as_numpy_iterator()))
if __name__ == '__main__':
    input_reader_test()
