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
    df = pd.read_excel(f'../data/战新词表_ner_{time_span}_labeled_{i}.xlsx', dtype=str)
    category_list = df['category'].values.tolist()
    word_list = df['word'].values.tolist()
    freq_list = df['freq'].values.tolist()
    result_list = df['result'].values.tolist()
    result_list_new = []
    for result in result_list:
        result_set = str(result).split()
        result_set = [r.strip() for r in result_set]
        if '是' in result_set:
            result_list_new.append(1)
        else:
            result_list_new.append(0)

    category2word = defaultdict(list)
    for category, word, freq, result in zip(category_list, word_list, freq_list, result_list_new):
        if result == 1 and int(freq) > 1:
            category2word[category].append(word)

    for category, word_list in category2word.items():
        print(category, len(word_list))

    with open(f'../data/战新词表_ner_{time_span}_labeled.json', 'w', encoding='utf-8') as f:
        json.dump(category2word, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    for time_span in ['125', '135', '145']:
        label_clean(time_span, 0)
