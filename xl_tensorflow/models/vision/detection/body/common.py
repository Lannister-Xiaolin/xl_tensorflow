#!usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
common module for custom definition of layers, models .etc
"""

from ..utils.yolo_utils import compose
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Conv2D
from functools import wraps
from tensorflow.keras.regularizers import l2

AGGREGATE_METHOD = ("add", "concat", "avg", "max", "min", "sub")
import tensorflow.keras.layers as layers
from ..utils.yolo_utils import compose


def node_aggregate(inputs, method="add", axis=-1):
    """
    aggregate different nodes
    Args:
        inputs:  lists of  tensor with same shape
        method:must be one of following:
               "add", "concat", "avg", "max", "min", "sub"
        axis: axis to concatenate when method is concat

    Returns:

    """
    assert method in AGGREGATE_METHOD, "node aggregate must be in {}".format(str(AGGREGATE_METHOD))
    if method == "concat":
        x = layers.Concatenate(axis)(inputs)
    elif method == "avg":
        x = layers.Average()(inputs)
    elif method == "max":
        x = layers.Maximum()(inputs)
    elif method == "min":
        x = layers.Minimum()(inputs)
    elif method == "sub":
        assert len(inputs) == 2, "sub aggregate method only support two input"
        x = layers.Subtract(inputs)
    else:
        x = layers.Add()(inputs)
    return x





