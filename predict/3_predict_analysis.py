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

c2threshold = {
    0: (0.94, 0.94),
    1: (0.99, 0.99),
    2: (0.99, 0.99),
    3: (0.98, 0.98),
    4: (0.97, 0.97),
    5: (0.94, 0.94),
    6: (0.9, 0.9),
    7: (0.99, 0.99)
}
print(c2threshold)


def node2value_trans(node2value, threshold, year, category):
    node2value_clean = {}
    t_s, t_e = threshold
    t = t_s + (year - 2011) * (t_e - t_s) / (2023 - 2011)

    if 2016 <= year <= 2020:
        if category == 0:
            # done
            t += 0.03
        elif category == 2:
            # done
            t += 0.0095
        elif category == 4:
            # done
            t += 0.02
        elif category == 5:
            # done
            t += 0.059
        elif category == 7:
            # done
            t += 0.005

    for node, value in node2value.items():
        if value > t:
            node2value_clean[node] = 1
        else:
            node2value_clean[node] = 0
    return node2value_clean


def analysis():
    # load node2predict_c
    node2predict_c = {}
    for year in range(2011, 2024):
        with open('result/node2predict_c_{}.json'.format(year), 'r', encoding='utf-8') as f:
            node2predict_c[year] = json.load(f)

    count = {}
    count_sum = {}
    for year in range(2011, 2024):
        count[year] = {}
        count_sum[year] = set()
        for category in range(8):
            node2value = node2value_trans(node2predict_c[year][str(category)], c2threshold[category], year, category)
            count[year][category] = sum(node2value.values())
            pos_node_set = [node for node, pred in node2value.items() if pred == 1]
            count_sum[year] = count_sum[year] | set(pos_node_set)

    for year in range(2011, 2024):
        count[year]['sum(not repeat)'] = len(count_sum[year])
        count[year]['sum(repeat)'] = sum(count[year].values()) - count[year]['sum(not repeat)']
    # save count by pandas
    df = pd.DataFrame(count)
    print(df)


def output():
    # load node2predict_c
    node2predict_c = {}
    for year in range(2011, 2024):
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

    for year in range(2011, 2024):
        result_list = []
        for category in range(8):
            node2value = node2value_trans(node2predict_c[year][str(category)], c2threshold[category], year, category)
            for node, pred in node2value.items():
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
    with open('result/node2predict_c_2023.json', 'r', encoding='utf-8') as f:
        node2predict_c = json.load(f)
    df = pd.read_csv('全部AB股.csv', dtype=str)
    node_list = df['股票代码'].to_list()
    node_list = ['0' * (6 - len(node)) + node for node in node_list]

    node_easy_name_list = df['证券名称'].to_list()
    node2node_easy_name = {node: node_easy_name for node, node_easy_name in zip(node_list, node_easy_name_list)}

    result_list = []
    for category in range(8):
        node2value = node2value_trans(node2predict_c[str(category)], c2threshold[category], 2023, category)
        for node, pred in node2value.items():
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
    analysis()
    output()
    test()
