# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：zx_2_v2
@File    ：get_company.py
@IDE     ：PyCharm
@Author  ：Logan
@Date    ：2023/12/13 下午3:50

# 整体我们有三个列表的数据来源
# 1. 隐性冠军（）
# 2. 战新百强（2019-2023）
# 3. 企业分类()

"""

import pandas as pd
# 来文斯坦
import Levenshtein
import os


def get_node_id_list_listed():
    node_list_listed = pd.read_csv('data/node_list/全部AB股.csv', dtype=str)['公司中文名称'].values.tolist()
    node_id_list_listed = pd.read_csv('data/node_list/全部AB股.csv', dtype=str)['证券代码'].values.tolist()
    node_id_list_listed = [node_id.split('.')[0] for node_id in node_id_list_listed]
    node2node_id = {node: node_id for node, node_id in zip(node_list_listed, node_id_list_listed)}
    return node2node_id


def get_node_list_1():
    """
    隐性冠军
    """
    year_list = [2016, 2017, 2018, 2019, 2020, 2021, 2022]
    node2node_id = get_node_id_list_listed()
    year2node_list = {}

    for year in year_list:
        df = pd.read_excel('data/node_list/node_list_1/隐形单项冠军.xlsx', sheet_name=str(year), dtype=str)
        node_1_list = df['示范企业'].values.tolist()
        node_2_list = df['培育企业'].values.tolist()
        node_3_list = df['单项冠军产品-生产企业'].values.tolist()
        # remove nan
        node_1_list = [node for node in node_1_list if str(node) != 'nan']
        node_2_list = [node for node in node_2_list if str(node) != 'nan']
        node_3_list = [node for node in node_3_list if str(node) != 'nan']
        # listed
        node_1_list = [node for node in node_1_list if node in node2node_id]
        node_2_list = [node for node in node_2_list if node in node2node_id]
        node_3_list = [node for node in node_3_list if node in node2node_id]

        node_list_combine = list(set(node_1_list + node_2_list + node_3_list))
        node_list_combine = [(node, node2node_id[node]) for node in node_list_combine]
        year2node_list[year] = node_list_combine

    # save excel by pandas
    for year, node_list in year2node_list.items():
        df = pd.DataFrame(node_list, columns=['node', 'node_id'])
        df.to_excel(f'data/node_list/node_list_1/{year}.xlsx', index=False)


def node_clean(node, org_list):
    """
    清洗企业名称
    :param node:
    :param org_list:
    :return:
    """
    for org in org_list:
        if node.endswith(org):
            node = node.replace(org, '')
    return node


def get_node_list_2():
    """
    战新百强
    """
    year_list = [2019, 2020, 2021, 2022, 2023]
    node2node_id = get_node_id_list_listed()
    year2node_list = {}

    org_list = ['有限公司', '股份有限公司', '集团有限公司', '集团股份有限公司',
                '有限责任公司', '股份有限责任公司', '集团有限责任公司', '集团股份有限责任公司']
    # long 2 short
    org_list = sorted(org_list, key=lambda x: len(x), reverse=True)
    print(org_list)

    for year in year_list:
        print(f'战新匹配结果{year}.csv')
        df = pd.read_csv(f'data/node_list/node_list_2/战新匹配结果{year}.csv', dtype=str)
        node_list = df['企业名称'].values.tolist()
        node_list_new = []
        # listed 这里可能需要加一个模糊匹配
        for node in node_list:
            node = node.replace('（', '(').replace('）', ')')
            for node_ in node2node_id.keys():
                sim_2 = Levenshtein.ratio(node_clean(node, org_list), node_clean(node_, org_list))
                if sim_2 > 0.95:
                    node_list_new.append(node_)

        node_list = [(node, node2node_id[node]) for node in node_list_new]
        year2node_list[year] = node_list

    # save excel by pandas
    for year, node_list in year2node_list.items():
        df = pd.DataFrame(node_list, columns=['node', 'node_id'])
        df.to_excel(f'data/node_list/node_list_2/{year}.xlsx', index=False)


def get_node_list_3():
    """
    企业分类
    """
    with open('data/node_list/node_list_3/战新行业代码.txt', 'r', encoding='utf-8') as f:
        industry_code = f.readlines()
    industry_code = [code.strip() for code in industry_code]
    industry_code = [code.replace('*', '') for code in industry_code]
    print(industry_code)

    node2node_id = get_node_id_list_listed()
    node_id2node = {node_id: node for node, node_id in node2node_id.items()}

    # load 全部A股-行业代码.xlsx
    df = pd.read_excel('data/node_list/node_list_3/全部A股-行业代码.xlsx', dtype=str)
    node_id_list = df['证券代码'].values.tolist()
    node_code_list = df['所属国民经济行业代码(2017)'].values.tolist()

    node_id_list_clean = []

    for node_id, node_code in zip(node_id_list, node_code_list):
        code_split = node_code.split('-')
        code = code_split[-1][1:]
        print(code)
        if code in industry_code:
            node_id_list_clean.append(node_id.split('.')[0])

    node_list = [(node_id2node[node_id], node_id) for node_id in node_id_list_clean if node_id in node_id2node]
    df = pd.DataFrame(node_list, columns=['node', 'node_id'])
    df.to_excel(f'data/node_list/node_list_3/all.xlsx', index=False)


def node_combine():
    """
    企业合并
    按照125(2011-2015),135(2016-2020),145(2021-2025)进行合并
    """
    df = pd.read_excel('data/node_list/node_list_3/all.xlsx', dtype=str)
    node_list_3 = df['node'].values.tolist()

    node2node_id = get_node_id_list_listed()

    timespan2year_list = {
        '125': [2011, 2012, 2013, 2014, 2015],
        '135': [2016, 2017, 2018, 2019, 2020],
        '145': [2021, 2022, 2023, 2024, 2025]
    }

    for time_span, year_list in timespan2year_list.items():
        node_list = []
        for year in year_list:
            # file exist check
            # node_1
            if os.path.exists(f'data/node_list/node_list_1/{year}.xlsx'):
                print(f'load node1 {year}')
                df = pd.read_excel(f'data/node_list/node_list_1/{year}.xlsx', dtype=str)
                node_list.extend(df['node'].values.tolist())
            # node_2
            if os.path.exists(f'data/node_list/node_list_2/{year}.xlsx'):
                print(f'load node2 {year}')
                df = pd.read_excel(f'data/node_list/node_list_2/{year}.xlsx', dtype=str)
                node_list.extend(df['node'].values.tolist())
            # node_3
            node_list.extend(node_list_3)

        node_list = sorted(list(set(node_list)))
        print('num of node:', len(node_list))
        node_list = [(node, node2node_id[node]) for node in node_list]
        df = pd.DataFrame(node_list, columns=['node', 'node_id'])
        df.to_excel(f'data/node_list/node_list_{time_span}.xlsx', index=False)


if __name__ == '__main__':
    # get_node_list_1()
    # get_node_list_2()
    # get_node_list_3()
    node_combine()
