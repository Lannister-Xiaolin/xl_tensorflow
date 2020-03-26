#!usr/bin/env python3
# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def recall_precision_f1(y_predict, y_true, class_num, class2id=None):
    """
    目标检测准确率评估
    Args:
        y_predict: 预测的box类别，格式[box_class1, ]
        y_true: [class1,class2,class2]
        class_num: class num
        class2id:
    Returns:
        pd.DataFrame()
    """
    assert len(y_predict) == len(y_true)
    id2class = dict((val, key) for key, val in class2id.items()) if class2id else None
    result = precision_recall_fscore_support(y_predict, y_true, labels=list(range(class_num)))
    result = {"precision": result[0], "recall": result[1], "f1_score": result[2], "support": result[3]}
    result = pd.DataFrame(result) if not id2class else\
        pd.DataFrame(result, index=[id2class[i] for i in range(class_num)])
    return result


print(recall_precision_f1(np.array([0, 1, 2, 0, 1, 2]),np.array([0, 2, 1, 0, 0, 1]),3,{"a":0,"b":1,"c":2}))