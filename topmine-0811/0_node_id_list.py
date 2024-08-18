# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：zx_2_v2
@File    ：get_company.py
@IDE     ：PyCharm
@Author  ：Logan
@Date    ：2023/12/13 下午3:50

# 整体我们有两个
# 1. 战新百强（2019-2023）
# 2. 年报与发改委目录的匹配

"""

import pandas as pd
# 来文斯坦
import Levenshtein
import os
import json
from collections import defaultdict, Counter

from texttable import Texttable
import multiprocessing as mp
from functools import partial

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def get_node_id_list_listed():
    node_list_listed = pd.read_csv('data/node_list/全部AB股.csv', dtype=str)['公司中文名称'].values.tolist()
    node_id_list_listed = pd.read_csv('data/node_list/全部AB股.csv', dtype=str)['证券代码'].values.tolist()
    node_id_list_listed = [node_id.split('.')[0] for node_id in node_id_list_listed]
    node2node_id = {node_id: node for node, node_id in zip(node_list_listed, node_id_list_listed)}
    return node2node_id


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


def get_node_list_add():
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
        df = pd.read_csv(f'data/node_list/node_list_add/战新匹配结果{year}.csv', dtype=str)
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
        df.to_excel(f'data/node_list/node_list_add/{year}.xlsx', index=False)


def word_count_single(word2category, txt_file_path, file):
    """
    每个企业的每个领域的关键词的数量和种类
    """
    txt_path = os.path.join(txt_file_path, file)
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read()
    category2num_word = defaultdict(int)
    category2word_set = defaultdict(set)
    for word, category in word2category.items():
        num = text.count(word)
        category2num_word[category] += num
        if num > 0:
            category2word_set[category].add(word)
    category2num_type = {category: len(word_set) for category, word_set in category2word_set.items()}
    company_name = file[:6]
    return company_name, category2num_word, category2num_type


def get_node_list_core_prepare():
    """
    每个企业的每个领域的关键词的数量
    """

    print('---------------------deal_3---------------------')
    input_file_path = 'data/clean_2/'
    input_file_list = os.listdir(input_file_path)
    input_file_list = sorted(input_file_list, key=lambda x: int(x[:4]))
    timespan2year_list = {
        '125': ['2011', '2012', '2013', '2014', '2015'],
        '135': ['2016', '2017', '2018', '2019', '2020'],
        '145': ['2021', '2022', '2023', '2024', '2025']
    }
    year2time_span = {}
    for time_span, year_list in timespan2year_list.items():
        for year in year_list:
            year2time_span[year] = time_span

    for input_file in input_file_list:
        time_span = year2time_span[input_file]
        # load word base
        with open(f'data/word/word_base/word_base_{time_span}.json', 'r', encoding='utf-8') as f:
            category2word = json.load(f)
        word2category = {}
        for category, word_list in category2word.items():
            for word in word_list:
                word2category[word] = category
            word2category[category] = category

        load_file_path = os.path.join(input_file_path, input_file)
        load_file_list = os.listdir(load_file_path)

        print('start deal with {} ...'.format(input_file), 'pdf num: ', len(load_file_list))
        # multi process
        pool = mp.Pool()
        func = partial(word_count_single, word2category, load_file_path)
        result = pool.map(func, load_file_list)
        pool.close()
        pool.join()

        company2category2num_word = {}
        company2category2num_type = {}

        for company_name, category2num_word, category2num_type in result:
            company2category2num_word[company_name] = category2num_word
            company2category2num_type[company_name] = category2num_type
        # save to excel by pandas in ../data/node_list/node_list_core/
        df = pd.DataFrame(company2category2num_word).T
        df = df.fillna(0)
        df.to_excel(f'data/node_list/node_list_core/{input_file}_num_word.xlsx')
        df = pd.DataFrame(company2category2num_type).T
        df = df.fillna(0)
        df.to_excel(f'data/node_list/node_list_core/{input_file}_num_type.xlsx')


def df2dict(company2category2num, num_th_list, category2index):
    """
    df to dict
    """
    num_th2category2num = np.zeros((len(num_th_list), len(category2index)))
    for company, category2num in company2category2num.items():
        for category, num in category2num.items():
            for num_th_index, num_th in enumerate(num_th_list):
                if num >= num_th:
                    num_th2category2num[num_th_index][category2index[category]] += 1
    return num_th2category2num


def draw_num_th2category2num(label, num_th2category2num, num_th_list, category2index, year):
    """
    画热力图
    """
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=num_th2category2num,
            y=[str(num_th) for num_th in num_th_list],
            x=list(category2index.keys()),
            colorscale='Viridis',
            text=num_th2category2num,
            texttemplate="%{text}",
        )
    )
    fig.update_layout(
        title=f'{year}',
        xaxis_title='category',
        yaxis_title='num_th',
        width=800,
        height=800
    )
    fig.write_image(f'data/node_list/node_list_core_img/{year}_num_{label}.png')


def get_node_list_core_analysis():
    """
    每年画一个热力图
    """
    category2index = {'高端装备制造': 0,
                      '节能环保': 1,
                      '生物': 2,
                      '数字创意': 3,
                      '新材料': 4,
                      '新能源': 5,
                      '新能源汽车': 6,
                      '新一代信息技术': 7}

    num_th_list = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30, 50]

    for year in range(2011, 2024):
        # load 2 dict: company2category2num
        print(f'year: {year}')
        df = pd.read_excel(f'data/node_list/node_list_core/{year}_num_word.xlsx')
        company2category2num_word = df.set_index('Unnamed: 0').T.to_dict()
        # num_th2category2num_word = df2dict(company2category2num_word, num_th_list, category2index)
        # draw_num_th2category2num('word', num_th2category2num_word, num_th_list, category2index, year)

        df = pd.read_excel(f'data/node_list/node_list_core/{year}_num_type.xlsx')
        company2category2num_type = df.set_index('Unnamed: 0').T.to_dict()
        # num_th2category2num_type = df2dict(company2category2num_type, num_th_list, category2index)
        # draw_num_th2category2num('type', num_th2category2num_type, num_th_list, category2index, year)

        category2scatter = {category: [] for category in category2index.keys()}
        for company in company2category2num_word.keys():
            for category in category2index.keys():
                n_w = company2category2num_word[company][category]
                n_t = company2category2num_type[company][category]
                if n_w == 0 and n_t == 0:
                    continue
                category2scatter[category].append((n_w, n_t))

        fig = make_subplots(rows=4, cols=2, subplot_titles=list(category2index.keys()))
        for i, category in enumerate(category2index.keys()):
            x = [scatter[0] for scatter in category2scatter[category]]
            y = [scatter[1] for scatter in category2scatter[category]]
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode='markers',
                    name=category
                ),
                row=i // 2 + 1, col=i % 2 + 1
            )

        fig.update_layout(title=f'{year}', width=800, height=800)
        fig.write_image(f'data/node_list/node_list_core_img/{year}_scatter.png')


def get_th_80(num_type_count, rate):
    """
    获取80%的阈值
    """
    total = sum(num_type_count.values())
    th_80 = 0
    th_80_count = 0
    for num, count in num_type_count.items():
        th_80_count += count
        # if 0.75 < th_80_count / total < 0.85:
        if th_80_count / total > rate:
            th_80 = num
            break
    return th_80


def get_node_list_core_clean():
    """
    企业合并
    按照125(2011-2015),135(2016-2020),145(2021-2025)进行合并
    """
    # core
    year2category2node_list = {}
    year2category2num_node = defaultdict(lambda: defaultdict(int))

    category2index = {'高端装备制造': 0,
                      '节能环保': 1,
                      '生物': 2,
                      '数字创意': 3,
                      '新材料': 4,
                      '新能源': 5,
                      '新能源汽车': 6,
                      '新一代信息技术': 7}

    for year in range(2011, 2024):
        # load 2 dict: company2category2num
        print(f'year: {year}')
        df = pd.read_excel(f'data/node_list/node_list_core/{year}_num_type.xlsx', dtype=str)
        category2company2num_type = df.set_index('Unnamed: 0').to_dict()
        df = pd.read_excel(f'data/node_list/node_list_core/{year}_num_word.xlsx', dtype=str)
        category2company2num_word = df.set_index('Unnamed: 0').to_dict()
        category2node_list = {}
        for category in category2index:
            company2num_type = category2company2num_type[category]
            company2num_word = category2company2num_word[category]
            # value2int
            company2num_type = {company: int(num_type) for company, num_type in company2num_type.items()}
            company2num_word = {company: int(num_word) for company, num_word in company2num_word.items()}
            # 根据 80-20 获取阈值
            num_type_count = Counter(company2num_type.values())
            num_word_count = Counter(company2num_word.values())
            # sorted by key
            num_type_count = dict(sorted(num_type_count.items(), key=lambda x: x[0]))
            num_word_count = dict(sorted(num_word_count.items(), key=lambda x: x[0]))
            # remove 0
            num_type_count.pop(0, None)
            num_word_count.pop(0, None)
            num_type_th_80 = get_th_80(num_type_count, rate=0.8)
            num_word_th_80 = get_th_80(num_word_count, rate=0.8)
            # get node_list and add to dict
            node_list_num_type = [node for node, num_type in company2num_type.items() if num_type > num_type_th_80]
            node_list_num_word = [node for node, num_word in company2num_word.items() if num_word > num_word_th_80]
            node_list = list(set(node_list_num_type) | set(node_list_num_word))
            category2node_list[category] = node_list
            # record
            year2category2num_node[year][category] = len(node_list)

        year2category2node_list[year] = category2node_list

    year2node_list = {}
    for year, category2node_list in year2category2node_list.items():
        node_list = []
        for category, node_list_ in category2node_list.items():
            node_list.extend(node_list_)
        node_list = list(set(node_list))
        year2node_list[year] = node_list

    # save by json
    with open('data/node_list/node_list_core/year2node_list.json', 'w', encoding='utf-8') as f:
        json.dump(year2node_list, f)


def node_combine():
    """
    对core和add进行合并
    """
    node_id2node = get_node_id_list_listed()

    timespan2year_list = {
        '125': [2011, 2012, 2013, 2014, 2015],
        '135': [2016, 2017, 2018, 2019, 2020],
        '145': [2021, 2022, 2023]
    }
    with open('data/node_list/node_list_core/year2node_list.json', 'r', encoding='utf-8') as f:
        year2node_list_core = json.load(f)

    for time_span, year_list in timespan2year_list.items():
        node_list = []
        for year in year_list:
            # file exist check
            # node_add
            if os.path.exists(f'data/node_list/node_list_add/{year}.xlsx'):
                print(f'load node_add {year}')
                df = pd.read_excel(f'data/node_list/node_list_add/{year}.xlsx', dtype=str)
                node_list.extend(df['node_id'].values.tolist())
            # node_core
            node_list.extend(year2node_list_core[str(year)])

        node_list = sorted(list(set(node_list)))
        print('num of node:', len(node_list))
        node_list = [(node, node_id2node[node]) for node in node_list if node in node_id2node]
        df = pd.DataFrame(node_list, columns=['node_id', 'node'])
        df.to_excel(f'data/node_list/node_list_{time_span}.xlsx', index=False)


if __name__ == '__main__':
    # get_node_list_core_prepare()
    # get_node_list_core_analysis()
    get_node_list_core_clean()
    # get_node_list_add()
    node_combine()
