#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：zx_2 
@File    ：3_predict_analysis.py
@IDE     ：PyCharm 
@Author  ：Logan
@Date    ：2023/12/9 上午12:42 
"""

import pandas as pd
import json
import numpy as np
from model import MyDnn, MyDnnSimple, FocalLoss
import torch
from tqdm import tqdm


def get_inputs(year, time_span):
    device = 'cuda:0'
    node2predict_c = {}

    for category in tqdm(range(8), desc=str(year)):
        node2vec_path = 'data/node2vec_{}_v0.json'.format(year)
        # load node2vec by json
        with open(node2vec_path, 'r', encoding='utf-8') as f:
            node2vec = json.load(f)
        x_list = list(node2vec.values())
        x_list = torch.Tensor(x_list)
        x_list = x_list.to(device)
        node_list = list(node2vec.keys())
        # load model
        model = MyDnnSimple()
        model_state_dict = torch.load(f'model/Dnn_{time_span}_{category}.pt')
        c = np.load(f'../dnn/data/category2vec_{time_span}.npy')
        c = torch.Tensor(c[category])
        c = c.to(device)
        model.load_state_dict(model_state_dict)
        model.to(device)
        model.eval()
        # predict
        with torch.no_grad():
            y_pred = model(x_list, c).squeeze()
            y_pred = torch.sigmoid(y_pred)
            y_pred = y_pred.cpu().numpy().tolist()
        # node2pred
        node2pred = {node: label for node, label in zip(node_list, y_pred)}
        node2predict_c[category] = node2pred

    # save by json
    with open('result/node2predict_c_{}.json'.format(year), 'w', encoding='utf-8') as f:
        json.dump(node2predict_c, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    for year in range(2011, 2016):
        get_inputs(year, '125')
    for year in range(2016, 2021):
        get_inputs(year, '135')
    for year in range(2021, 2024):
        get_inputs(year, '145')
