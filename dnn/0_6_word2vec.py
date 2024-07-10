#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：zx_2 
@File    ：0_6_word2vec.py
@IDE     ：PyCharm 
@Author  ：Logan
@Date    ：2023/12/3 下午7:15 
"""

import pandas as pd
from collections import defaultdict
import json
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch
import os
import numpy as np


def model_load(model_file):
    tokenizer = AutoTokenizer.from_pretrained(model_file)
    model = AutoModel.from_pretrained(model_file)
    return tokenizer, model


def get_category2word_list():
    """
    获取类别到词集合的映射
    :return:
    """
    df = pd.read_excel('data/战新词表_1215_simple.xlsx', dtype=str, header=None)
    category2word_list = dict()
    for i in range(8):
        word_list = df[i + 1].values.tolist()
        word_list = [word for word in word_list if str(word) != 'nan']
        word_len_list = [len(word) for word in word_list]
        category2word_list[word_list[0]] = word_list[1:]

    return category2word_list


def get_word2vec(model_path, version):
    """

    :param model_path:
    :param version:
    :return:
    """
    # load category2word_set
    category2word_list = get_category2word_list()
    # print(category2word_list)
    # for c, word_list in category2word_list.items():
    #     print(c, len(word_list))

    device = torch.device('cuda:0')
    # model_load
    tokenizer, bert_model = model_load(model_path)
    # load checkpoint
    if version == 1:
        model_state_dict = torch.load('pretrain/MLM_v2_20231127-194715_0.pt')
        bert_model.load_state_dict(model_state_dict, strict=False)

    bert_model.to(device)
    bert_model.eval()

    category2vec = []

    category2index = {'高端装备': 0, '节能环保': 1, '生物': 2, '数字创意': 3,
                      '新材料': 4, '新能源': 5, '新能源汽车': 6, '新一代信息技术': 7}

    for category in tqdm(category2index):
        word_list = category2word_list[category]
        print(category, len(word_list))
        # encode
        encoded_input = tokenizer(word_list, padding=True, truncation=True, return_tensors='pt', max_length=20)
        input_ids = encoded_input['input_ids'].to(device)
        attention_mask = encoded_input['attention_mask'].to(device)
        token_type_ids = encoded_input['token_type_ids'].to(device)

        with torch.no_grad():
            model_output = bert_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            # Perform pooling. In this case, cls pooling.
            word_embeds = model_output.pooler_output

        # max pooling
        category2vec.append(torch.max(word_embeds, dim=0)[0].cpu().numpy().tolist())

    # save by numpy
    category2vec = np.array(category2vec)
    print(category2vec.shape)
    np.save('data/category2vec_v{}.npy'.format(version), category2vec)


if __name__ == '__main__':
    # get_category2word_set()
    model_path = 'hfl/chinese-roberta-wwm-ext-large'
    get_word2vec(model_path, version=0)
    get_word2vec(model_path, version=1)
