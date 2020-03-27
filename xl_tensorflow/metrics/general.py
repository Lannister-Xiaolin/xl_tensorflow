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
    result = pd.DataFrame(result) if not id2class else \
        pd.DataFrame(result, index=[id2class[i] for i in range(class_num)])
    return result


def topk_caculate(predict_p, true_labels, k=1, number_cat=None, epsilon=1e-19):
    """
    topk准确率计算
    Args:
        predict_p: 预测的概率分布shape=(N,number_cat)
        true_labels: 真实标签，shape=(N,)
        k: top k
        number_cat: number categories
        eposion: e

    Returns:
        topk acc [top1,top2,...,topk]
        cat acc [ [top1,top2,...,topk]， [top1,top2,...,topk]， [top1,top2,...,topk]]
    """
    total = len(predict_p)
    if not number_cat:
        number_cat = max(np.unique(true_labels), np.max(true_labels))
    real = np.array(true_labels)
    arg_sort = predict_p.argsort(axis=-1)
    total_acc = []  # 整体准确率
    cat_acc = []  # 分类准确率

    def acc_caculate(pred, real, prev_tp_sum, prev_cat_tp, real_cat_positive, number_cat, total):
        all_eval = (pred == real)
        cat_tp = [sum((real == i) & all_eval) for i in range(number_cat)]
        cat_tp = [prev_cat_tp[i] + cat_tp[i] for i in range(number_cat)]
        cat_precition = [(cat_tp[i] / (real_cat_positive[i] + epsilon)) for i in range(number_cat)]
        tp_sum = sum(all_eval) + prev_tp_sum
        topk = tp_sum / total
        return topk, cat_precition, cat_tp, tp_sum

    prev_tp_sum = 0
    prev_cat_tp = [0 for i in range(number_cat)]  # topk-1中的每个类别true positive总数
    real_cat_positive = [sum(real == i) for i in range(number_cat)]  # 每个类别的正样本总数
    for i in range(k):
        pred = arg_sort[:, -1 * (i + 1)]
        topk, cat_precition, prev_cat_tp, prev_tp_sum = acc_caculate(pred, real, prev_tp_sum, prev_cat_tp,
                                                                     real_cat_positive,
                                                                     number_cat, total)
        total_acc.append(topk)
        cat_acc.append(cat_precition)
    return total_acc, cat_acc


if __name__ == '__main__':
    test_pred = np.array(
        [[0.5, 0.3, 0.1], [0.3, 0.5, 0.1], [0.1, 0.3, 0.5], [0.5, 0.3, 0.1], [0.3, 0.5, 0.1], [0.1, 0.3, 0.5]])
    test_True = np.array([0, 0, 1, 2, 0, 2])
    print(topk_caculate(test_pred, test_True, 3, number_cat=3))
# print(recall_precision_f1(np.array([0, 1, 2, 0, 1, 2]),np.array([0, 2, 1, 0, 0, 1]),3,{"a":0,"b":1,"c":2}))
