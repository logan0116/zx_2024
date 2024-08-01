#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：zx_2 
@File    ：0_7_make_train_data.py
@IDE     ：PyCharm 
@Author  ：Logan
@Date    ：2023/12/3 上午10:36 
"""

# 手上的数据数2019年的数据，所以用2019年的数据来做train_set

import pandas as pd
import json
import numpy as np


def load_node_list():
    with open('data/node_id_list.json', 'r', encoding='utf-8') as f:
        c2node_id_list = json.load(f)

    c2node_id_list = sorted(c2node_id_list.items(), key=lambda x: len(x[1]), reverse=True)

    node2label = dict()
    for c, node_id_list in c2node_id_list:
        if c == '高端装备':
            label = 0
        elif c == '节能环保':
            label = 1
        elif c == '生物':
            label = 2
        elif c == '数字创意':
            label = 3
        elif c == '新材料':
            label = 4
        elif c == '新能源':
            label = 5
        elif c == '新能源汽车':
            label = 6
        elif c == '新一代信息技术':
            label = 7
        else:
            label = -1
        for node_id in node_id_list:
            node2label[node_id] = label

    return node2label


def get_inputs(time_span):
    # load by json
    with open(f'data/node2vec_{time_span}.json', 'r', encoding='utf-8') as f:
        node2vec = json.load(f)
    node2label = load_node_list()

    x_list = []
    y_list = []

    for node, vec in node2vec.items():
        node = node.split('_')[0]
        if node not in node2label:
            continue
        x_list.append(vec)
        y_list.append(node2label[node])

    # save by numpy
    x_list = np.array(x_list)
    y_list = np.array(y_list)
    np.save(f'data/x_{time_span}', x_list)
    np.save(f'data/y_{time_span}', y_list)


if __name__ == '__main__':
    for time_span in ['125', '135', '145']:
        get_inputs(time_span)
