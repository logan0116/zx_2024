#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：zx_2 
@File    ：0_data_maker.py
@IDE     ：PyCharm 
@Author  ：Logan
@Date    ：2023/11/27 下午2:31 
"""

import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import json


def load_data(filename):
    """
    数据的输入
    :param filename:
    :return:
    """
    with open(filename, encoding='utf-8') as f:
        data = f.readlines()

    text_list = []
    for d in data:
        d = json.loads(d)
        text_list.append(d['text'].replace(' ', ''))
    return text_list


def load_word2category():
    """
    :return:
    """
    category_set = set()
    word2category_list = defaultdict(list)

    # load word list by pandas from excel
    df = pd.read_excel('data/战新词表_topmine_top50_labeled_n_combine.xlsx')
    category_list = df['category'].values.tolist()
    word_list = df['word'].values.tolist()
    label_list = df['label'].values.tolist()

    for category, word, label in zip(category_list, word_list, label_list):
        category_set.add(category)
        if label >= 2:
            word2category_list[word].append(category)

    # 引入word_base，将word_base中的词加入到word2category_list中
    with open('data/word_base.json', encoding='utf-8') as f:
        category2word_list_base = json.load(f)

    for category, word_list in category2word_list_base.items():
        for word in word_list:
            word2category_list[word].append(category)

    print(len(word2category_list))
    print(word2category_list)
    return word2category_list, list(category_list)


def get_word_bit(word, text):
    """
    返回list:(word_start_bit，word_end_bit)
    :param word:
    :param text:
    :return:
    """
    word_bit_list = []
    for i in range(len(text)):
        if text[i:i + len(word)] == word:
            word_bit_list.append((i, i + len(word) - 1))
    return word_bit_list


def make_data(file_path):
    """
    example:
    {"text": "万通地产设计总监刘克峰；", "label": {"name": {"刘克峰": [[8, 10]]}, "company": {"万通地产": [[0, 3]]}, "position": {"设计总监": [[4, 7]]}}}
    :return:
    """

    data = load_data(file_path)
    word2category_list, category_list = load_word2category()
    print(len(word2category_list))

    write_list = []

    print('data_process processing...')
    for text in tqdm(data):
        word_list_temp = [word for word in word2category_list if word in text]
        if not word_list_temp:
            continue
        word_category_list_temp = [(word, word2category_list[word]) for word in word_list_temp]

        label = defaultdict(dict)
        for word, category_list in word_category_list_temp:
            word_bit_list = get_word_bit(word, text)
            for category in category_list:
                label[category][word] = word_bit_list

        write_list.append({'text': text, 'label': label})
    # write
    with open('data/data_ner_1213.json', 'w', encoding='utf-8') as f:
        for l in write_list:
            f.write(json.dumps(l, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    make_data('data/text_1213.json')
