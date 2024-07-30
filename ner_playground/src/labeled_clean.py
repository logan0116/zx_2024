#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：zx_2 
@File    ：4_GPT_labeled.py
@IDE     ：PyCharm 
@Author  ：Logan
@Date    ：2023/11/28 下午5:07 
"""

import pandas as pd
import json
from collections import defaultdict


def label_clean(time_span, i):
    df = pd.read_excel(f'data/战新词表_topmine_top50_{time_span}_labeled_{i}.xlsx', dtype=str)
    category_list = df['category'].values.tolist()
    word_list = df['word'].values.tolist()
    result_list = df['result'].values.tolist()
    result_list_new = []
    for result in result_list:
        result_set = str(result).split()
        result_set = [r.strip() for r in result_set]
        if '是' in result_set:
            result_list_new.append(1)
        else:
            result_list_new.append(0)

    df_new = pd.DataFrame({'category': category_list, 'word': word_list, 'result': result_list_new})
    df_new.to_excel(f'data/战新词表_topmine_top50_{time_span}_labeled_{i}_clean.xlsx', index=False)


def label_combine(time_span):
    df = pd.read_excel(f'data/战新词表_topmine_top50_{time_span}_labeled_0_clean.xlsx')
    category_list = df['category'].values.tolist()
    word_list = df['word'].values.tolist()
    result_list_1 = df['result'].values.tolist()
    df = pd.read_excel(f'data/战新词表_topmine_top50_{time_span}_labeled_1_clean.xlsx')
    result_list_2 = df['result'].values.tolist()
    df = pd.read_excel(f'data/战新词表_topmine_top50_{time_span}_labeled_2_clean.xlsx')
    result_list_3 = df['result'].values.tolist()
    category2word = defaultdict(list)
    for category, word, result_1, result_2, result_3 in zip(category_list, word_list,
                                                            result_list_1, result_list_2, result_list_3):
        if result_1 + result_2 + result_3 >= 2:
            category2word[category].append(word)

    with open(f'data/战新词表_topmine_top50_{time_span}_labeled_combine.json', 'w', encoding='utf-8') as f:
        json.dump(category2word, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    time_span = '145'
    for i in range(3):
        label_clean(time_span, i)
    label_combine(time_span)
