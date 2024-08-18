# pdf trans to text
import re

import pdfplumber
import os
from tqdm import tqdm
# multi process
import multiprocessing as mp
from functools import partial
import json
import pandas as pd

import jieba

from collections import Counter


def sentence_clean(file_s):
    """
    句子的处理
    :param s:
    :return:
    """
    file, s = file_s.split(' ******** ')
    # 正则去除括号中的内容
    pattern = re.compile(r'[\(].+[\)]')
    s = re.sub(pattern, '', s)
    pattern = re.compile(r'[（].+[）]')
    s = re.sub(pattern, '', s)
    # 开头的部分
    # 大写数字
    pattern_2 = re.compile(r'[零一二三四五六七八九○０Ｏ]+、')
    s = re.sub(pattern_2, '', s)
    pattern_2 = re.compile(r'[零一二三四五六七八九○０Ｏ]+是')
    s = re.sub(pattern_2, '', s)
    # 小写数字
    pattern_2 = re.compile(r'[0-9]+[、．.]')
    s = re.sub(pattern_2, '', s)
    pattern_2 = re.compile(r'^[0-9]+')
    s = re.sub(pattern_2, '', s)
    # 条
    pattern_2 = re.compile(r'[第].+[条章]')
    s = re.sub(pattern_2, '', s)
    # 年月日
    pattern_3 = re.compile(r'[零一二三四五六七八九○０Ｏ]+年[一二三四五六七八九十]+月[一二三四五六七八九十]+日')
    s = re.sub(pattern_3, '', s)
    pattern_3 = re.compile(r'[0-9０]+年[0-9]+月[0-9０]+日')
    s = re.sub(pattern_3, '', s)
    # 标点
    pattern_4 = re.compile(r'[，。；：？！…—～·《》〈〉“”‘’【】『』〔〕〖〗〘〙〚〛〝〞〟、%（）]')
    s = re.sub(pattern_4, ' ', s)
    # 数字
    pattern_5 = re.compile(r'[0-9０１２３４５６７８９①②③④⑤⑥⑦⑧⑨⑩.]+')
    s = re.sub(pattern_5, ' ', s)
    # 去除空格
    s = s.strip()
    if s:
        # jieba分词
        word_list = jieba.cut(s, cut_all=False)
        s = ' '.join(word_list)
        s = ' '.join(s.split())
        # s = ' '.join(s.split())

    return file + ' ******** ' + s


def load_node_list(time_span):
    df = pd.read_excel(f'../data/node_list/node_list_{time_span}.xlsx', dtype=str)
    node_id_list = df['node_id'].values.tolist()
    return node_id_list


def deal_5(time_span):
    """
    合并所有文本并分词
    :return:
    """
    input_file_path = '../data/clean_2/'
    input_file_list = os.listdir(input_file_path)
    input_file_list = sorted(input_file_list, key=lambda x: int(x[:4]))

    if time_span == '125':
        input_file_range = ['2011', '2012', '2013', '2014', '2015']
    elif time_span == '135':
        input_file_range = ['2016', '2017', '2018', '2019', '2020']
    elif time_span == '145':
        input_file_range = ['2021', '2022', '2023', '2024', '2025']
    else:
        raise ValueError('time_span error')

    # load node_id_list_chosen
    node_id_list_chosen = load_node_list(time_span)

    text_list = []
    for input_file in input_file_list:
        if input_file not in input_file_range:
            continue

        load_file_path = os.path.join(input_file_path, input_file)
        load_file_list = os.listdir(load_file_path)
        # clean
        load_file_list = [file for file in load_file_list if file[:6] in node_id_list_chosen]
        for file in tqdm(load_file_list):
            txt_path = os.path.join(load_file_path, file)
            with open(txt_path, 'r', encoding='utf-8') as f:
                text_ = f.read()

            text_list_temp = text_.split('\n\n')
            for text in text_list_temp:
                t = text.replace('\n', '')
                t_list = t.split('。')
                for t_ in t_list:
                    text_list.append(file + ' ******** ' + t_)

    # jieba
    pool = mp.Pool()
    text_list_clean = pool.map(sentence_clean, text_list)
    pool.close()
    pool.join()
    text_list_clean = [file_s.split(' ******** ') for file_s in text_list_clean
                       if len(file_s.split(' ******** ')[1]) > 20]

    print(len(text_list_clean))

    length_counter = []

    # save by json
    text_cut_path = f'../data/text_{time_span}.json'
    with open(text_cut_path, 'w', encoding='utf-8') as f:
        for file, s in text_list_clean:
            length_counter.append(len(s))
            f.write(json.dumps({'file': file, 'text': s}, ensure_ascii=False) + '\n')
    # save by txt for topmine
    text_cut_path = f'../data/text_{time_span}.txt'
    with open(text_cut_path, 'w', encoding='utf-8') as f:
        for file, s in text_list_clean:
            f.write(s + '\n')


if __name__ == "__main__":
    deal_5(time_span='125')
    deal_5(time_span='135')
    deal_5(time_span='145')
