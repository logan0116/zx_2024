#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：zx_2 
@File    ：0_5_node2vec.py
@IDE     ：PyCharm 
@Author  ：Logan
@Date    ：2023/12/3 下午3:42 
"""

import json
from collections import defaultdict
from tqdm import tqdm
import numpy as np


def get_index_file_s2vev_file_list(year, version):
    index_file_list = [
        'data/doc2vec_{}/id_64.json'.format(year),
        'data/doc2vec_{}/id_128.json'.format(year),
        'data/doc2vec_{}/id_256.json'.format(year),
        'data/doc2vec_{}/id_512.json'.format(year)
    ]
    index_list_list = []
    for index_file in index_file_list:
        with open(index_file, 'r', encoding='utf-8') as file:
            index_list = json.load(file)
        index_list = [index.split('_')[0] for index in index_list]
        index_list_list.append(index_list)

    s2vec_file_list = [
        'data/doc2vec_{}/patent_feature_v{}_64.npy'.format(year, version),
        'data/doc2vec_{}/patent_feature_v{}_128.npy'.format(year, version),
        'data/doc2vec_{}/patent_feature_v{}_256.npy'.format(year, version),
        'data/doc2vec_{}/patent_feature_v{}_512.npy'.format(year, version)
    ]

    return index_list_list, s2vec_file_list


def get_node2vec(year, version):
    """
    :return:
    """

    node2vec_set = defaultdict(list)

    id_list_list, s2vec_file_list = get_index_file_s2vev_file_list(year, version)

    for id_list, s2vec_file in zip(id_list_list, s2vec_file_list):
        print(s2vec_file)
        s2vec = np.load(s2vec_file)
        # sample check
        if len(id_list) == s2vec.shape[0]:
            print('safe')

        s2vec = s2vec.tolist()

        for id_, s_vec in zip(id_list, s2vec):
            node2vec_set[id_].append(s_vec)

    node2vec = dict()

    for node, vec_list in node2vec_set.items():
        # max pooling
        node2vec[node] = np.max(vec_list, axis=0).tolist()

    # save by json
    with open('data/node2vec_{}_v{}.json'.format(year, version), 'w', encoding='utf-8') as f:
        json.dump(node2vec, f, ensure_ascii=False)


if __name__ == '__main__':
    # get_node_year2year()
    # get_index2node_year()
    for year in range(2011, 2023):
        get_node2vec(year, version=0)
