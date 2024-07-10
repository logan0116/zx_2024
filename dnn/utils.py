#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：zx_2 
@File    ：utils.py
@IDE     ：PyCharm 
@Author  ：Logan
@Date    ：2023/12/3 下午8:03 
"""
import numpy as np
import torch
from torch.utils.data import Dataset



def label_deal(y: torch.LongTensor, category):
    """
    分类标签处理
    :param y: 标签
    :param categories: 标签类别
    :return:
    """
    y = torch.where(y == category, 1, 0)
    return y


class MyDataSet(Dataset):
    """
    data_process load
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def sPRF_2(y_ture, y_pred):
    """
    tow class
    """
    tp = torch.sum((y_ture == 1) & (y_pred == 1)).data
    fp = torch.sum((y_ture == 0) & (y_pred == 1)).data
    fn = torch.sum((y_ture == 1) & (y_pred == 0)).data
    if tp + fp == 0:
        pre = 0
    else:
        pre = tp / (tp + fp)
    if tp + fn == 0:
        rec = 0
    else:
        rec = tp / (tp + fn)
    if pre + rec == 0:
        f1 = 0
    else:
        f1 = 2 * pre * rec / (pre + rec)

    return pre, rec, f1


def early_stop(loss_list, patience=10, min_delta=0.001):
    """
    early stop
    :param loss_list:
    :param patience:
    :param min_delta:
    :return:
    """
    if len(loss_list) < patience + 1:
        return False
    else:
        if np.abs(np.mean(loss_list[-patience:]) - loss_list[-1]) < min_delta:
            return True
        else:
            return False


