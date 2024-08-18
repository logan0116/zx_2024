#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/12/3 上午9:34
# @Author  : liu yuhan
# @FileName: 2_topmine_add.py
# @Software: PyCharm

from collections import Counter
from tqdm import tqdm
import json

import pandas as pd


class TopmineAdd():
    def __init__(self, index_path, seg_path, min_freq):
        self.seg_path = seg_path
        self.index_path = index_path
        self.min_freq = min_freq
        # 构建索引
        self.index2word = dict()
        self.word2index = dict()
        self.get_index2word()
        self.word_dict_s = Counter()
        self.word_dict_m = Counter()
        self.get_frequency()

    def get_index2word(self):
        '''
        Returns a list of stopwords.
        '''
        f = open(self.index_path)
        words = []
        for line in f:
            words.append(line.strip())
        self.index2word = dict(zip([i for i in range(len(words))], words))
        self.word2index = dict(zip(words, [i for i in range(len(words))]))

    def get_frequency(self):
        '''
        计算词和词组的词频
        词组和单词貌似没有必要分开进行计算
        20211206
        词组和单词需要分开进行计算
        :return:
        '''
        docs = open(self.seg_path, 'r', encoding='UTF-8')

        for doc in tqdm(docs):
            doc = doc.strip()
            words = doc.split(', ')
            for word in words:
                if not word:
                    continue
                if ' ' in word:
                    self.word_dict_m[word] += 1
                else:
                    self.word_dict_s[word] += 1

    def get_keywords(self, save_path_s, save_path_m):
        """
        根据词频进行筛选
        这一阶段的词频筛选为第一阶段的词频筛选，选择一个小词频，用于术语的合并
        :return:
        """
        # 存储词组 by pandas

        word_list_s = []
        word_list_m = []

        count = 0
        for word, freq in self.word_dict_m.items():
            if freq > self.min_freq:
                word_trans = ' '.join([self.index2word[int(index)] for index in word.split()])
                word_list_m.append((word_trans, freq))
                count += 1
        print('num-keyword-m:', count)
        # save
        df = pd.DataFrame(word_list_m, columns=['word', 'freq'])
        df.to_excel(save_path_m, index=False)

        # 存储单词
        count = 0
        for word, freq in self.word_dict_s.items():
            if freq > self.min_freq:
                word_trans = self.index2word[int(word)]
                word_list_s.append((word_trans, freq))
                count += 1
        print('num-keyword-s:', count)
        # save
        df = pd.DataFrame(word_list_s, columns=['word', 'freq'])
        df.to_excel(save_path_s, index=False)


if __name__ == '__main__':
    time_span = '145'
    seg_path = f'data/topmine/partitioneddocs_{time_span}.txt'
    index_path = f'data/topmine/vocab_{time_span}.txt'
    save_path_s = f'data/topmine/keywords_single_{time_span}.xlsx'
    save_path_m = f'data/topmine/keywords_multiple_{time_span}.xlsx'

    # for min_freq in range(0, 500, 25):
    #     print('min_freq:', min_freq)
    #     topmine_add = TopmineAdd(index_path, seg_path, min_freq)
    #     # topmine_add.add_base_keywords(keyword_base_path)
    #     topmine_add.get_keywords(save_path_s, save_path_m)

    min_freq = 100
    print('min_freq:', min_freq)
    topmine_add = TopmineAdd(index_path, seg_path, min_freq)
    # topmine_add.add_base_keywords(keyword_base_path)
    topmine_add.get_keywords(save_path_s, save_path_m)
