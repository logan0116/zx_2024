#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：zx_2 
@File    ：0_4_s2vec.py
@IDE     ：PyCharm 
@Author  ：Logan
@Date    ：2023/12/2 上午7:59 
"""

from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch
import os
import numpy as np
import json

import argparse


def parameter_parser():
    parser = argparse.ArgumentParser(description='for GNN')
    # model
    parser.add_argument('--node', help="Please give a value for node_size")
    return parser.parse_args()


def model_load(model_file):
    tokenizer = AutoTokenizer.from_pretrained(model_file)
    model = AutoModel.from_pretrained(model_file)
    return tokenizer, model


# def data_load(filename):
#     """
#     文件写入
#     :param filename:
#     :return:
#     """
#     with open(filename, 'r', encoding='utf-8') as f:
#         patent2doc = json.load(f)
#     data_process = list(patent2doc.values())
#     return data_process


def load_data(filename):
    """
    {"file": "300063_2011_2011年年度报告.txt", "text": "公司直接客户主要是包装印刷企业产品主要适用于瓦楞纸箱高档纸张纸巾卷筒纸及装饰纸等精美包装的印刷"}
    数据的输入
    :param filename:
    :return:
    """
    with open(filename, encoding='utf-8') as f:
        data = f.readlines()

    data = [json.loads(l.strip()) for l in data if l.strip()]
    data = [(l['file'], l['text']) for l in data if l['text']]
    return data


def doc_trans_1(model_path, file_path, output_path):
    # 载入模型
    tokenizer, bert_model = model_load(model_path)
    bert_model.eval()
    bert_model.cuda()
    print('模型载入成功')
    # 载入数据
    data = load_data(file_path)
    # 将data分为4个部分
    print('数据载入成功')
    # 16个一组

    id_64 = []
    token_ids_64 = []
    id_128 = []
    token_ids_128 = []
    id_256 = []
    token_ids_256 = []
    id_512 = []
    token_ids_512 = []

    for index, text in tqdm(data):
        token_ids = tokenizer.encode(text, max_length=512, truncation=True)
        if len(token_ids) <= 64:
            id_64.append(index)
            token_ids_64.append(token_ids)
        elif len(token_ids) <= 128:
            id_128.append(index)
            token_ids_128.append(token_ids)
        elif len(token_ids) <= 256:
            id_256.append(index)
            token_ids_256.append(token_ids)
        else:
            id_512.append(index)
            token_ids_512.append(token_ids)

    print(len(token_ids_64))
    print(len(token_ids_128))
    print(len(token_ids_256))
    print(len(token_ids_512))

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # save
    with open(output_path + '/id_64.json', 'w') as f:
        json.dump(id_64, f)
    with open(output_path + '/id_128.json', 'w') as f:
        json.dump(id_128, f)
    with open(output_path + '/id_256.json', 'w') as f:
        json.dump(id_256, f)
    with open(output_path + '/id_512.json', 'w') as f:
        json.dump(id_512, f)

    with open(output_path + '/token_ids_64.json', 'w') as f:
        json.dump(token_ids_64, f)
    with open(output_path + '/token_ids_128.json', 'w') as f:
        json.dump(token_ids_128, f)
    with open(output_path + '/token_ids_256.json', 'w') as f:
        json.dump(token_ids_256, f)
    with open(output_path + '/token_ids_512.json', 'w') as f:
        json.dump(token_ids_512, f)


def doc_trans_2(model_path, output_path, length, batch_size, version=0):
    print('length', length, 'batch_size', batch_size, 'version', version)
    device = torch.device('cuda:1')
    # model_load
    tokenizer, bert_model = model_load(model_path)
    # load checkpoint
    if version == 1:
        device = torch.device('cuda:1')
        model_state_dict = torch.load('pretrain/MLM_v2_20231127-194715_0.pt')
        bert_model.load_state_dict(model_state_dict, strict=False)

    bert_model.to(device)
    bert_model.eval()
    with open(output_path + '/token_ids_{}.json'.format(length), 'r') as f:
        token_ids_list = json.load(f)

    attention_mask_list = [[1] * len(token_ids) for token_ids in token_ids_list]
    # padding
    token_ids_list = [token_ids + [0] * (length - len(token_ids)) for token_ids in token_ids_list]
    attention_mask_list = [attention_mask + [0] * (length - len(attention_mask)) for attention_mask in
                           attention_mask_list]
    print(len(token_ids_list))

    # encode
    train_x = []
    data_length = len(token_ids_list)
    for i in tqdm(range(0, data_length, batch_size)):
        inputs = token_ids_list[i:i + batch_size]
        attention_mask = attention_mask_list[i:i + batch_size]
        inputs = torch.tensor(inputs).to(device)
        attention_mask = torch.tensor(attention_mask).to(device)
        last_hidden_states = bert_model(inputs, attention_mask=attention_mask).pooler_output.cpu().detach()
        train_x.extend(last_hidden_states.tolist())

    print('数据转化成功')
    print(len(train_x))
    train_x = np.array(train_x)
    # 保存结果
    np.save(output_path + '/patent_feature_v{}_{}.npy'.format(version, length), train_x)
    print('数据保存成功')


# def check(node='node1'):
#     """
#     检查embedding是否完整
#     :param node:
#     测试通过，没有问题。
#     """
#     length_list = [64, 128, 256, 512]
#     result_path = '../data_process/output/{}_feature'.format(node)
#     for length in length_list:
#         print(node, length)
#         id_path = os.path.join(result_path, 'id_{}.json'.format(length))
#         with open(id_path, 'r') as f:
#             id_list = json.load(f)
#         print('id_list', len(id_list))
#         feature_path = os.path.join(result_path, 'patent_feature_{}.npy'.format(length))
#         feature = np.load(feature_path, allow_pickle=True)
#         print('feature shape', feature.shape)


if __name__ == '__main__':
    time_span = '145'
    model_path = 'hfl/chinese-roberta-wwm-ext-large'
    file_path = f'data/text_{time_span}.json'
    output_path = f'data/doc2vec_{time_span}'
    # doc_trans_1(model_path, file_path, output_path)
    for length, batch_size in [(512, 12)]:
        doc_trans_2(model_path, output_path, length, batch_size)
