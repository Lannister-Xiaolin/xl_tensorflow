#!usr/bin/env python3
# -*- coding: UTF-8 -*-
import tensorflow as tf

dataset = tf.data.Dataset.list_files(r"E:\Temp\test\tfrecord\*.tfrecord")
