#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：zx_2_v2 
@File    ：0_1_node_list.py
@IDE     ：PyCharm 
@Author  ：Logan
@Date    ：2023/12/18 下午2:35 
"""

import pandas as pd
import json
from collections import defaultdict


def get_index2label_add():
    node_list_1 = pd.read_excel('data/战新标签-add.xlsx', dtype=str, sheet_name='高端装备')['企业名称'].values.tolist()
    node_list_2 = pd.read_excel('data/战新标签-add.xlsx', dtype=str, sheet_name='新能源汽车')[
        '企业名称'].values.tolist()
    node_list_3 = pd.read_excel('data/战新标签-add.xlsx', dtype=str, sheet_name='生物')['企业名称'].values.tolist()
    node_list_4 = pd.read_excel('data/战新标签-add.xlsx', dtype=str, sheet_name='新能源')['企业名称'].values.tolist()
    node_list_5 = pd.read_excel('data/战新标签-add.xlsx', dtype=str, sheet_name='新材料')['企业名称'].values.tolist()

    node_list_1 = list(set(node_list_1))
    node_list_2 = list(set(node_list_2))
    node_list_3 = list(set(node_list_3))
    node_list_4 = list(set(node_list_4))
    node_list_5 = list(set(node_list_5))

    # node_listed_list
    node_listed_list = pd.read_csv('data/全部AB股.csv')['公司中文名称'].values.tolist()
    node_id_listed_list = pd.read_csv('data/全部AB股.csv', dtype=str)['股票代码'].values.tolist()
    node_id_listed_list = ['0' * (6 - len(node)) + node for node in node_id_listed_list]
    node2node_id = {node: node_id for node, node_id in zip(node_listed_list, node_id_listed_list)}

    # filter
    node_list_1 = [node2node_id[node] for node in node_list_1 if node in node2node_id]
    node_list_2 = [node2node_id[node] for node in node_list_2 if node in node2node_id]
    node_list_3 = [node2node_id[node] for node in node_list_3 if node in node2node_id]
    node_list_4 = [node2node_id[node] for node in node_list_4 if node in node2node_id]
    node_list_5 = [node2node_id[node] for node in node_list_5 if node in node2node_id]

    print('-------------add-----------')
    print(len(node_list_1))
    print(len(node_list_2))
    print(len(node_list_3))
    print(len(node_list_4))
    print(len(node_list_5))

    return node_list_1, node_list_2, node_list_3, node_list_4, node_list_5


def get_index2label():
    df = pd.read_excel('data/战新上市公司2019和2021对比-230112.xlsx', dtype=str)
    node_id_list = df['股票代码'].values.tolist()
    category_list_1 = df['所属战新八大领域'].values.tolist()
    category_list_2 = df['所属战新八大领域_2'].values.tolist()

    node_list_1 = []  # 高端装备
    node_list_2 = []  # 新能源汽车
    node_list_3 = []  # 生物
    node_list_4 = []  # 新能源
    node_list_5 = []  # 新材料
    node_list_6 = []  # 数字创意
    node_list_7 = []  # 节能环保
    node_list_8 = []  # 新一代信息技术

    for node_id, category_1, category_2 in zip(node_id_list, category_list_1, category_list_2):
        if str(category_1) == '高端装备' or str(category_2) == '高端装备':
            node_list_1.append(node_id)
        if str(category_1) == '新能源汽车' or str(category_2) == '新能源汽车':
            node_list_2.append(node_id)
        if str(category_1) == '生物' or str(category_2) == '生物':
            node_list_3.append(node_id)
        if str(category_1) == '新能源' or str(category_2) == '新能源':
            node_list_4.append(node_id)
        if str(category_1) == '新材料' or str(category_2) == '新材料':
            node_list_5.append(node_id)
        if str(category_1) == '数字创意' or str(category_2) == '数字创意':
            node_list_6.append(node_id)
        if str(category_1) == '节能环保' or str(category_2) == '节能环保':
            node_list_7.append(node_id)
        if str(category_1) == '新一代信息技术' or str(category_2) == '新一代信息技术':
            node_list_8.append(node_id)

    print('-------------base-----------')
    print(len(node_list_1))
    print(len(node_list_2))
    print(len(node_list_3))
    print(len(node_list_4))
    print(len(node_list_5))
    print(len(node_list_6))
    print(len(node_list_7))
    print(len(node_list_8))

    node_list_1_add, node_list_2_add, node_list_3_add, node_list_4_add, node_list_5_add = get_index2label_add()
    node_list_1.extend(node_list_1_add)
    node_list_2.extend(node_list_2_add)
    node_list_3.extend(node_list_3_add)
    node_list_4.extend(node_list_4_add)
    node_list_5.extend(node_list_5_add)

    node_list_1 = list(set(node_list_1))
    node_list_2 = list(set(node_list_2))
    node_list_3 = list(set(node_list_3))
    node_list_4 = list(set(node_list_4))
    node_list_5 = list(set(node_list_5))
    node_list_6 = list(set(node_list_6))
    node_list_7 = list(set(node_list_7))
    node_list_8 = list(set(node_list_8))

    print('-------------all-----------')
    print(len(node_list_1))
    print(len(node_list_2))
    print(len(node_list_3))
    print(len(node_list_4))
    print(len(node_list_5))
    print(len(node_list_6))
    print(len(node_list_7))
    print(len(node_list_8))

    # save by json
    with open('data/node_id_list.json', 'w', encoding='utf-8') as f:
        json.dump({
            '高端装备': node_list_1,
            '新能源汽车': node_list_2,
            '生物': node_list_3,
            '新能源': node_list_4,
            '新材料': node_list_5,
            '数字创意': node_list_6,
            '节能环保': node_list_7,
            '新一代信息技术': node_list_8
        }, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    get_index2label()
