#!usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
feat aggregation module
to be implemented feat aggregation method:
    bifpn
    fpn
    panet
    sfam
    asff
"""

from ..utils.yolo_utils import compose
from .common import node_aggregate



def bifpn_network():
    pass


def pan_network(features, configs, ascending_shape=False):
    """
    panet
     Reference: [Path Aggregation Network](https://arxiv.org/abs/1803.01534)
    Args:
        features:
        configs:
        ascending_shape: bool, True if shape in features ordered shape in ascending(13,26,52), else False
    Returns:

    """
    backward_flows = []
    features = features if ascending_shape else features[::-1]
    for i in range(len(features)):
        features[i] = compose(*configs.agg_inputs_ops[i])(features[i])
    for i, feature in enumerate(features):
        if i == 0:
            backward_flows.append(feature)
        else:
            #  层间操作
            prev = compose(*configs.backward_ops[i - 1])(backward_flows[-1])
            node = node_aggregate([feature, prev], method="concat")
            #  横向操作
            node = compose(*configs.inlevel_backward_ops[i])(node)
            backward_flows.append(node)
    forward_flows = []
    for i, feature in enumerate(backward_flows[::-1]):
        if i == 0:
            forward_flows.append(feature)
        else:
            #  层间操作
            prev = compose(*configs.forward_ops[i - 1])(forward_flows[-1])
            node = node_aggregate([feature, prev], method="concat")
            #  横向操作
            node = compose(*configs.inlevel_forward_ops[i])(node)
            forward_flows.append(node)
    new_features = forward_flows[::-1] if ascending_shape else forward_flows
    return new_features


def fpn_network():
    pass
