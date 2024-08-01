#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：zx_2 
@File    ：3_predict_125.py
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


def get_inputs(year):
    device = 'cuda:0'
    node2predict_c = {}
    c2threshold = {
        0: 0.5,
        1: 0.5,
        2: 0.5,
        3: 0.5,
        4: 0.5,
        5: 0.5,
        6: 0.5,
        7: 0.5
    }

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
        model_state_dict = torch.load('model/Dnn_125_{}.pt'.format(category))
        c = np.load('../dnn/data/category2vec_125.npy')
        c = torch.Tensor(c[category])
        c = c.to(device)

        model.load_state_dict(model_state_dict)
        model.to(device)
        model.eval()

        # predict
        y_pred = model(x_list, c).squeeze()
        y_pred = torch.sigmoid(y_pred)

        # 根据y0和y1的差异进行预测
        y_pred = torch.where(y_pred > c2threshold[category], torch.ones_like(y_pred), torch.zeros_like(y_pred))
        # print(y_pred)
        # y_pred = torch.argmax(y_pred, dim=1)
        y_pred = y_pred.cpu().numpy().tolist()
        # node2pred
        node2pred = {node: label for node, label in zip(node_list, y_pred)}
        node2predict_c[category] = node2pred

    # save by json
    with open('result/node2predict_c_{}.json'.format(year), 'w', encoding='utf-8') as f:
        json.dump(node2predict_c, f, ensure_ascii=False, indent=4)


def analysis():
    # load node2predict_c
    node2predict_c = {}
    for year in range(2011, 2016):
        with open('result/node2predict_c_{}.json'.format(year), 'r', encoding='utf-8') as f:
            node2predict_c[year] = json.load(f)

    count = {}
    count_sum = {}
    for year in range(2011, 2016):
        count[year] = {}
        count_sum[year] = set()
        for category in range(8):
            count[year][category] = sum(node2predict_c[year][str(category)].values())
            pos_node_set = [node for node, pred in node2predict_c[year][str(category)].items() if pred == 1]
            count_sum[year] = count_sum[year] | set(pos_node_set)

    count_sum = {year: len(node_set) for year, node_set in count_sum.items()}
    print(count_sum.values())
    print(count)
    # save count by pandas
    df = pd.DataFrame(count)
    df.to_excel('result/count_125.xlsx')


def output():
    # load node2predict_c
    node2predict_c = {}
    for year in range(2011, 2016):
        with open('result/node2predict_c_{}.json'.format(year), 'r', encoding='utf-8') as f:
            node2predict_c[year] = json.load(f)

    df = pd.read_csv('全部AB股.csv', dtype=str)
    node_list = df['股票代码'].to_list()
    node_list = ['0' * (6 - len(node)) + node for node in node_list]

    node_name_list = df['公司中文名称'].to_list()
    node_easy_name_list = df['证券名称'].to_list()
    node2node_name = {node: node_name for node, node_name in zip(node_list, node_name_list)}
    node2node_easy_name = {node: node_easy_name for node, node_easy_name in zip(node_list, node_easy_name_list)}

    category_trans = {0: '高端装备', 1: '节能环保', 2: '生物', 3: '数字创意',
                      4: '新材料', 5: '新能源', 6: '新能源汽车', 7: '新一代信息技术'}

    for year in range(2011, 2016):
        result_list = []
        for category in range(8):
            for node, pred in node2predict_c[year][str(category)].items():
                if pred == 1:
                    try:
                        result_list.append([category_trans[category], node,
                                            node2node_name[node], node2node_easy_name[node]])
                    except KeyError:
                        result_list.append([category_trans[category], node, '', ''])

        # save by pandas
        df = pd.DataFrame(result_list, columns=['战新产业分类', '股票代码', '公司中文名称', '证券名称'])
        df.to_excel('output/{}_result.xlsx'.format(year), index=False)


def test():
    # load node2predict_c
    node2predict_c = {}
    with open('result/node2predict_c_2015.json', 'r', encoding='utf-8') as f:
        node2predict_c = json.load(f)
    df = pd.read_csv('全部AB股.csv', dtype=str)
    node_list = df['股票代码'].to_list()
    node_list = ['0' * (6 - len(node)) + node for node in node_list]

    node_easy_name_list = df['证券名称'].to_list()
    node2node_easy_name = {node: node_easy_name for node, node_easy_name in zip(node_list, node_easy_name_list)}

    result_list = []
    for category in range(8):
        for node, pred in node2predict_c[str(category)].items():
            if pred == 1:
                try:
                    result_list.append(node2node_easy_name[node])
                except KeyError:
                    continue

    test_list = ['大华股份', '比亚迪', '歌尔股份', '海康威视', '烽火通信', '科大讯飞', '天准科技', '长飞光纤',
                 '横店东磁', '超声电子', '西部超导', '视源股份', '精测电子', '锐科激光', '中天科技', '中航光电',
                 '江丰电子', '中兴通讯', '大族激光', '长亮科技', '汇顶科技', '福光股份', '深信服', '安恒信息',
                 '鼎信通讯', '亿纬锂能', '迪普科技', '飞天诚信', '浪潮信息', '交控科技', '恒生电子', '海兰信',
                 '光迅科技', '龙软科技', '航天信息', '容百科技', '中国天楹', '彩虹股份', '川大智胜', '三环集团',
                 '绿盟科技', '隆利科技', '信维通信', '远光软件', '浪潮软件', '长盈精密', '太极股份', '永鼎股份',
                 '卫士通', '美亚柏科']
    print(len(test_list))
    count = 0
    for node in test_list:
        if node not in result_list:
            print(node)
            count += 1
    print(count)


if __name__ == '__main__':
    for year in range(2011, 2016):
        get_inputs(year)
    analysis()
    output()
    test()
