import json
from collections import defaultdict
import pandas as pd


def get_category2word_list(time_span):
    """
    获取类别到词集合的映射
    :return:
    """
    # category2word core
    with open(f'data/word_base_{time_span}.json', encoding='utf-8') as f:
        category2word_list_core = json.load(f)
    # category2word base
    with open(f'data/战新词表_topmine_top50_{time_span}_labeled_combine.json', encoding='utf-8') as f:
        category2word_list_base = json.load(f)
    # category2word expand
    with open(f'data/战新词表_ner_{time_span}_labeled.json', encoding='utf-8') as f:
        category2word_list_extra = json.load(f)

    category2word_count = defaultdict(list)
    # base
    for category, word_list in category2word_list_base.items():
        word_list = list(set(word_list))
        category2word_count[category].append(len(word_list))
    # core
    for category, word_list in category2word_list_core.items():
        word_list = list(set(word_list) - set(category2word_list_base[category]))
        category2word_count[category].append(len(word_list))
    # expand
    for category, word_list in category2word_list_extra.items():
        word_list = list(
            set(word_list) - set(category2word_list_base[category]) - set(category2word_list_core[category]))
        category2word_count[category].append(len(word_list))

    # print as table by pd
    # add index 基准 核心 扩展
    df = pd.DataFrame(category2word_count, index=['基准', '核心', '扩展'])
    print(f'-----------------{time_span}-----------------')
    print(df.T)

    return category2word_count


if __name__ == '__main__':
    get_category2word_list('125')
    get_category2word_list('135')
    get_category2word_list('145')
