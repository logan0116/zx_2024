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


def pdf2text(pdf_file_path, save_path, file):
    """

    :param pdf_file_path:
    :param save_path:
    :param file:
    :return:
    """
    # pdf_path = 'test.pdf'
    pdf_path = os.path.join(pdf_file_path, file)
    try:
        pdf = pdfplumber.open(pdf_path)
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
            page.flush_cache()
        pdf.close()
        txt_path = os.path.join(save_path, file.split('.')[0] + '.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
    except:
        # save to error file
        error_file_path = 'error.txt'
        with open(error_file_path, 'a', encoding='utf-8') as f:
            f.write(file + '\n')


def deal_1():
    input_file_path = 'data/input/'
    output_file_path = 'data/output/'
    input_file_list = os.listdir(input_file_path)
    input_file_list = sorted(input_file_list, key=lambda x: int(x[:4]))

    for input_file in input_file_list:
        load_file_path = os.path.join(input_file_path, input_file)
        load_file_list = os.listdir(load_file_path)
        # save path
        save_file_path = os.path.join(output_file_path, input_file)
        if not os.path.exists(save_file_path):
            os.mkdir(save_file_path)
        print('start deal with {} ...'.format(input_file), 'pdf num: ', len(load_file_list))
        # multi process
        pool = mp.Pool()
        func = partial(pdf2text, load_file_path, save_file_path)
        pool.map(func, load_file_list)
        pool.close()
        pool.join()


def title_check(l):
    """
    检查一行文字是否为标题
    :param l:
    :return:
    """
    flag = False
    # 第一节 第一章
    if re.search(r'第[一二三四五六七八九十]+[节章]', l[:5]):
        flag = True
    # （一）
    if re.search(r'[（(][一二三四五六七八九十]+[)）]', l[:5]):
        flag = True
    # 一、 一. 一)
    if re.search(r'[一二三四五六七八九十]+[.、) ]', l[:5]):
        flag = True
    if len(l) > 30:
        flag = False
    return flag


def table_check(l):
    """
    检查一行文字是否为表格
    :param l:
    :return:
    """
    flag = False
    if re.search(r'[-0-9.%]+[ ,][ \t]*[-0-9.%]+', l):
        flag = True
    return flag


def text_cut_single(txt_file_path, save_path, file):
    """
    :param txt_file_path:
    :param save_path:
    :param file:
    :return:
    """
    # pdf_path = 'test.pdf'
    txt_path = os.path.join(txt_file_path, file)
    title_list = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines_cut = []
    for l in lines:
        l = l.strip()
        # table
        if table_check(l):
            continue
        # title
        if title_check(l):
            lines_cut.append('')
            title_list.append(l)
        lines_cut.append(l)

    txt_path_cut = os.path.join(save_path, file)
    with open(txt_path_cut, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines_cut))

    return title_list


def load_node_list(time_span):
    df = pd.read_excel(f'data/node_list/node_list_{time_span}.xlsx', dtype=str)
    node_id_list = df['node_id'].values.tolist()
    return node_id_list


def deal_2(time_span):
    """

    """
    print('---------------------deal_2---------------------')
    input_file_path = 'data/output/'
    output_file_path = 'data/clean_1/'

    # load node_id_list_chosen
    node_id_list_chosen = load_node_list(time_span)

    input_file_list = os.listdir(input_file_path)
    input_file_list = sorted(input_file_list, key=lambda x: int(x[:4]))

    title_list = []

    timespan2year_list = {
        '125': ['2011', '2012', '2013', '2014', '2015'],
        '135': ['2016', '2017', '2018', '2019', '2020'],
        '145': ['2021', '2022', '2023', '2024', '2025']
    }

    year_list = timespan2year_list[time_span]

    for input_file in input_file_list:
        if input_file not in year_list:
            continue
        load_file_path = os.path.join(input_file_path, input_file)
        load_file_list = os.listdir(load_file_path)

        # clean
        load_file_list = [file for file in load_file_list if file.split('_')[0] in node_id_list_chosen]
        # save path
        save_file_path = os.path.join(output_file_path, input_file)
        if not os.path.exists(save_file_path):
            os.mkdir(save_file_path)

        print('start deal with {} ...'.format(input_file), 'pdf num: ', len(load_file_list))
        # multi process
        pool = mp.Pool()
        func = partial(text_cut_single, load_file_path, save_file_path)
        title_list_list = pool.map(func, load_file_list)
        pool.close()
        pool.join()

        for title_list_ in title_list_list:
            title_list += title_list_
    # save title
    with open(f'title_{time_span}.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(title_list))


def text_extract_single(title_screen_list, txt_file_path, save_file_path, file):
    """
    :param txt_file_path:
    :param save_file_path:
    :param file:
    :return:
    """
    # pdf_path = 'test.pdf'
    txt_path = os.path.join(txt_file_path, file)
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read()
    text_piece_list = text.split('\n\n')
    text_piece_list_clean = []
    for text_piece in text_piece_list:
        if text_piece.count('\n') < 1:
            # 只有一行
            continue
        title = text_piece.split('\n')[0]
        if title in title_screen_list:
            text_piece_list_clean.append(text_piece)
    text_clean = '\n\n'.join(text_piece_list_clean)
    txt_path_extract = os.path.join(save_file_path, file)
    with open(txt_path_extract, 'w', encoding='utf-8') as f:
        f.write(text_clean)


def deal_3():
    print('---------------------deal_3---------------------')
    input_file_path = 'data/clean_1/'
    output_file_path = 'data/clean_2/'
    input_file_list = os.listdir(input_file_path)
    input_file_list = sorted(input_file_list, key=lambda x: int(x[:4]))
    # title_screen

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
        with open(f'title_screen_{time_span}.txt', 'r', encoding='utf-8') as f:
            title_screen_list = f.readlines()
        title_screen_list = set([l.strip() for l in title_screen_list])
        load_file_path = os.path.join(input_file_path, input_file)
        load_file_list = os.listdir(load_file_path)
        # save path
        save_file_path = os.path.join(output_file_path, input_file)
        if not os.path.exists(save_file_path):
            os.mkdir(save_file_path)

        print('start deal with {} ...'.format(input_file), 'pdf num: ', len(load_file_list))
        # multi process
        pool = mp.Pool()
        func = partial(text_extract_single, title_screen_list, load_file_path, save_file_path)
        pool.map(func, load_file_list)
        pool.close()
        pool.join()


def make_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


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


def deal_4():
    """
    合并所有文本并分词
    :return:
    """
    input_file_path = '../data/clean_2/'
    input_file_list = os.listdir(input_file_path)
    input_file_list = sorted(input_file_list, key=lambda x: int(x[:4]))

    text_piece_list = []
    for input_file in input_file_list:
        load_file_path = os.path.join(input_file_path, input_file)
        load_file_list = os.listdir(load_file_path)
        for file in load_file_list:
            txt_path = os.path.join(load_file_path, file)
            with open(txt_path, 'r', encoding='utf-8') as f:
                text = f.read()
            text_piece_list_ = text.split('\n\n')
            text_piece_list.extend(text_piece_list_)

    text = ''
    for text_piece in text_piece_list:
        text += text_piece + '\n'

    # jieba
    sentence_list = text.split('\n')
    sentence_list_cut = []
    pool = mp.Pool()
    sentence_list_cut = pool.map(sentence_clean, sentence_list)
    pool.close()
    pool.join()
    # save
    text_cut_path = '../data/data.txt'
    with open(text_cut_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sentence_list_cut))


def deal_5(time_span):
    """
    合并所有文本并分词
    :return:
    """
    input_file_path = 'data/clean_2/'
    input_file_list = os.listdir(input_file_path)
    input_file_list = sorted(input_file_list, key=lambda x: int(x[:4]))

    # load node_id_list_chosen
    node_id_list_chosen = load_node_list(time_span)

    if time_span == '125':
        input_file_range = ['2011', '2012', '2013', '2014', '2015']
    elif time_span == '135':
        input_file_range = ['2016', '2017', '2018', '2019', '2020']
    elif time_span == '145':
        input_file_range = ['2021', '2022', '2023', '2024', '2025']
    else:
        raise ValueError('time_span error')

    text_list = []
    for input_file in input_file_list:
        if input_file not in input_file_range:
            continue

        load_file_path = os.path.join(input_file_path, input_file)
        load_file_list = os.listdir(load_file_path)
        load_file_list = [file for file in load_file_list if file.split('_')[0] in node_id_list_chosen]

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
    text_cut_path = f'data/text_{time_span}.json'
    with open(text_cut_path, 'w', encoding='utf-8') as f:
        for file, s in text_list_clean:
            length_counter.append(len(s))
            f.write(json.dumps({'file': file, 'text': s}, ensure_ascii=False) + '\n')
    # save by txt for topmine
    text_cut_path = f'data/text_{time_span}.txt'
    with open(text_cut_path, 'w', encoding='utf-8') as f:
        for file, s in text_list_clean:
            f.write(s + '\n')


if __name__ == "__main__":
    # deal_1()
    # deal_2('125')
    # deal_2('135')
    # deal_2('145')
    # deal_3()
    deal_5(time_span='125')
    deal_5(time_span='135')
    deal_5(time_span='145')
