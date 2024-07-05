#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：zx_2 
@File    ：4_GPT_labeled.py
@IDE     ：PyCharm 
@Author  ：Logan
@Date    ：2023/11/28 下午5:07 
"""

import openai
import time
import json
import pandas as pd
import os
from tqdm import tqdm


def text_generation(each_prompt: list):
    """
    给定一个prompt，返回一个message
    :param prompt:
    :return:
    """
    model_engine = "gpt-4-1106-preview"
    openai.api_key = ""

    start_time = time.time()
    completions = openai.ChatCompletion.create(
        model=model_engine,
        messages=each_prompt
    )
    end_time = time.time()
    message = completions.choices[0].message.content
    # print('time: ', end_time - start_time)
    # print(message)
    return message


def load_word2category():
    """
    :return:
    """
    category2word_list = dict()

    # load word list by pandas from excel
    df = pd.read_excel('data/战新词表_topmine_top50.xlsx', header=None)

    for i in range(8):
        word_list = df[i].values.tolist()
        word_list = [word for word in word_list if str(word) != 'nan']
        category2word_list[word_list[0]] = word_list[1:]

    return category2word_list


def make_prompt(category):
    category2example = {
        '新一代信息技术': {
            'positive':
                ['人工智能', '公共事业信息服务', 'IT服务', '集成电路', '网络运营服务'],
            'negative':
                ['多项技术', '产品结构', '国家工信部', '新兴行业', '国家级高新技术企业']
        },
        '新材料': {
            'positive':
                ['高分子材料', '表面功能材料', '负极材料', '先进结构材料', '新型膜材料'],
            'negative':
                ['一种新型', '客户个性化需求', '同比增长', '创新业务', '创新成果']
        },
        '生物': {
            'positive':
                ['生物医药', '新型疫苗', '生物医学工程', '生物农业', '生物育种'],
            'negative':
                ['全球多个国家地区', '处于行业', '多个领域', '生产组织', '分行业本期']
        },
        '高端装备制造': {
            'positive':
                ['航空装备', '航空材料', '轨道交通装备', '海洋工程装备', '服务机器人'],
            'negative':
                ['提高设备', '快速发展', '高科技企业', '高端品牌', '高新技术企业']
        },
        '节能环保': {
            'positive':
                ['高效节能', '智能水务', '水污染防治装备', '绿色建筑材料', '资源再生利用'],
            'negative':
                ['保证产品质量', '确保各项', '降低成本', '性能稳定', '质量保障']
        },
        '新能源汽车': {
            'positive':
                ['新能源汽车', '混合动力', '智能驾驶', '智能交通', '绿色节能'],
            'negative':
                ['高新技术产品', '创新驱动', '车型类别', '自主创新', '一种新型']
        },
        '新能源': {
            'positive':
                ['核电技术', '氢能源', '光伏发电', '生物质能', '绿色低碳'],
            'negative':
                ['新兴市场', '新型专利', '一种新型', '新兴业务', '新冠疫情']
        },
        '数字创意': {
            'positive':
                ['数字文化创意', '新型媒体服务', '设计服务', '人居环境设计服务', '工业设计服务'],
            'negative':
                ['创新团队', '数据统计', '基于数据', '数据调整', '发展战略']
        }
    }

    prompt = [{"role": "system",
               "content": "您是一个{}产业的专家。我将给您展示一些术语，请判断我给出的术语是否与{}产业相关，特别是该术语是否与{}产业的产品相关。" +
                          "如果可以请回复“是”，不可以则回复“否”。" +
                          "请注意，类似于“新兴技术、创新成果、快速发展”这样词语可能会出现在多个领域，所以判定其不具备描述特定产业的能力"
                          .format(category, category)}]

    example = category2example[category]
    for word_pos, word_neg in zip(example['positive'], example['negative']):
        prompt.append({"role": "user", "content": "术语：{}".format(word_pos)})
        prompt.append({"role": "assistant", "content": '是'})
        prompt.append({"role": "user", "content": "术语：{}".format(word_neg)})
        prompt.append({"role": "assistant", "content": '否'})

    return prompt


def load_already_get(already_get_path):
    if os.path.exists(already_get_path):
        with open(already_get_path, 'r', encoding='utf-8') as f:
            already_get_set = json.load(f)
    else:
        already_get_set = dict()
        with open(already_get_path, 'w', encoding='utf-8') as f:
            json.dump(already_get_set, f)
    return already_get_set


def labeled():
    """
    通过gpt3打标签
    :return:
    """
    category2word_list = load_word2category()
    already_get_set = load_already_get('data/label_already_get.json')

    # category2word_list = {'新一代信息技术': ['制品制造业', '技术装备', '产品研发制造', '设计制造', '制造能力'],
    #                       '新材料': ['新能源汽'],
    #                       '生物': ['生存发展']}

    for category, word_list in category2word_list.items():
        prompt = make_prompt(category)
        for word in tqdm(word_list):
            if category + " " + word in already_get_set:
                continue
            # try:
            each_prompt = prompt.copy()
            each_prompt.append({"role": "user", "content": "术语：{}".format(word)})
            each_prompt.append({"role": "assistant", "content": ''})

            message = text_generation(each_prompt)
            already_get_set[category + " " + word] = message
            # except:
            #     print('error', category, word)
            #     # save already get
            #     with open('data/label_already_get.json', 'w', encoding='utf-8') as f:
            #         json.dump(already_get_set, f)
            #     continue

    with open('data/label_already_get.json', 'w', encoding='utf-8') as f:
        json.dump(already_get_set, f)
    # save by pandas
    result_list = [{'category': c_w.split(' ')[0], 'word': c_w.split(' ')[1], 'result': res}
                   for c_w, res in already_get_set.items()]
    result_list = pd.DataFrame(result_list)
    result_list.to_excel('data/战新词表_topmine_top50_labeled_n3.xlsx', index=False)


if __name__ == '__main__':
    labeled()
