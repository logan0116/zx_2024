#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：zx_2_v2 
@File    ：1_base_word.py
@IDE     ：PyCharm 
@Author  ：Logan
@Date    ：2023/12/13 下午7:37 
"""

import re
import pandas as pd
import json


def word_clean(word):
    """
    清洗词语
    :param word:
    :return:
    """
    word = re.sub(r'[^\u4e00-\u9fa5]', '', word)
    word = word.replace('产业', '')
    return word


def get_word_base():
    df = pd.read_excel('data/word_base.xlsx', dtype=str, header=None)

    category2word_list = {}

    for i in range(8):
        word_list = df[i].values.tolist()
        word_list = [word for word in word_list if str(word) != 'nan']
        word_list = [word_clean(word) for word in word_list]
        category2word_list[word_list[0]] = word_list[1:]

    # save by json
    with open('data/word_base.json', 'w', encoding='utf-8') as f:
        json.dump(category2word_list, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    get_word_base()
