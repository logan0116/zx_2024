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
import requests
from zhipuai import ZhipuAI


def label_clean(time_span, i):
    df = pd.read_excel(f'data/战新词表_topmine_top50_{time_span}_labeled_{i}.xlsx', dtype=str)
    category_list = df['category'].values.tolist()
    word_list = df['word'].values.tolist()
    result_list = df['result'].values.tolist()
    result_list_new = []
    for result in result_list:
        result_set = str(result).split()
        result_set = [r.strip() for r in result_set]
        if '是' in result_set:
            result_list_new.append(1)
        else:
            result_list_new.append(0)

    df_new = pd.DataFrame({'category': category_list, 'word': word_list, 'result': result_list_new})
    df_new.to_excel(f'data/战新词表_topmine_top50_{time_span}_labeled_{i}_clean.xlsx', index=False)


if __name__ == '__main__':
    label_clean('125', 0)
    label_clean('125', 1)
