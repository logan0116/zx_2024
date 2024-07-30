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
import random


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


def load_word2category(time_span):
    """
    :return:
    """
    # load word_base_{}.json  战新词表_topmine_top50_{}_labeled_combine.json
    with open(f'../data/word_base_{time_span}.json', encoding='utf-8') as f:
        category2word_list_base = json.load(f)
    with open(f'../data/战新词表_topmine_top50_{time_span}_labeled_combine.json', encoding='utf-8') as f:
        category2word_list_core = json.load(f)
    word2category_list = defaultdict(list)
    # add 2 category to word2category_list
    for category, word_list in category2word_list_base.items():
        for word in word_list:
            word2category_list[word].append(category)
    for category, word_list in category2word_list_core.items():
        for word in word_list:
            word2category_list[word].append(category)

    # category_list
    category_list = category2word_list_base.keys()

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


def make_data(time_span):
    """
    example:
    {"text": "万通地产设计总监刘克峰；", "label": {"name": {"刘克峰": [[8, 10]]}, "company": {"万通地产": [[0, 3]]}, "position": {"设计总监": [[4, 7]]}}}
    :return:
    """
    file_path = f'../data/text_{time_span}.json'

    data = load_data(file_path)
    word2category_list, category_list = load_word2category(time_span)
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

    random.seed(int(time_span))
    # 70% train, 30% dev
    random.shuffle(write_list)
    train_num = int(0.7 * len(write_list))
    write_list_train = write_list[:train_num]
    write_list_dev = write_list[train_num:]

    # write
    with open(f'../data/data_ner_{time_span}.json', 'w', encoding='utf-8') as f:
        for l in write_list:
            f.write(json.dumps(l, ensure_ascii=False) + '\n')

    with open(f'../data/train_{time_span}.json', 'w', encoding='utf-8') as f:
        for l in write_list_train:
            f.write(json.dumps(l, ensure_ascii=False) + '\n')

    with open(f'../data/dev_{time_span}.json', 'w', encoding='utf-8') as f:
        for l in write_list_dev:
            f.write(json.dumps(l, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    for time_span in ['125', '135', '145']:
        make_data(time_span)
