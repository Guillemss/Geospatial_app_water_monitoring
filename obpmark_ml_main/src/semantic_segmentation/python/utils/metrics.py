import numpy as np


def accuracy(y_true, y_pred, print_it=False):
    y_pred = np.round(y_pred)
    TP = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    TN = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    acc = (TP + TN) / np.size(y_pred)
    if print_it:
        print(acc)
    return acc


def f1_score(y_true, y_pred, print_it=False):
    y_pred = np.round(y_pred)
    TP = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    FP = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    FN = np.sum(np.logical_and(y_pred == 0, y_true == 1))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)
    if print_it:
        print(f1)
    return f1


def IoU(y_true, y_pred, print_it=False):
    y_pred = np.round(y_pred)
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    if print_it:
        print(iou_score)
    return iou_score
