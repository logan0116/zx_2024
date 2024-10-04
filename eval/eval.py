import pandas as pd
from collections import defaultdict


def get_category2code_list(time):
    """
    获取战新行业代码
    """
    df = pd.read_excel(f'data/zx{time}.xlsx', sheet_name='战新行业代码', dtype=str)
    # 8列数据 header: category
    category2code_list = df.to_dict()
    category2code_list = {category: list(set(code_list.values()))
                          for category, code_list in category2code_list.items()}
    # remove nan
    category2code_list = {category: [code for code in code_list if str(code) != 'nan']
                          for category, code_list in category2code_list.items()}
    # remove *
    category2code_list = {category: [code.replace('*', '') for code in code_list]
                          for category, code_list in category2code_list.items()}

    code2category_set = defaultdict(set)
    for category, code_list in category2code_list.items():
        for code in code_list:
            code2category_set[code].add(category)

    return code2category_set


def get_top100():
    """
    企业分类
    """
    # category2code
    code2category_set_23 = get_category2code_list(23)

    # load 全部A股-行业代码.xlsx
    df = pd.read_excel('data/全部A股-行业代码.xlsx', dtype=str)
    node_id_list = df['证券代码'].values.tolist()
    node_code_list = df['所属国民经济行业代码(2017)'].values.tolist()
    node_id2node_code = {node_id[:6]: node_code.split('-')[-1][1:]
                         for node_id, node_code in zip(node_id_list, node_code_list)}

    # category2node_list
    category2node_list = defaultdict(list)
    for node_id, code in node_id2node_code.items():
        category_set = code2category_set_23.get(code, set())
        for category in category_set:
            category2node_list[category].append(node_id)
    return category2node_list


def eval_node(target):
    category_list_1 = pd.read_excel('data/2023_result.xlsx', dtype=str)['战新产业分类'].values.tolist()
    node_list_1 = pd.read_excel('data/2023_result.xlsx', dtype=str)['股票代码'].values.tolist()
    category2node_list = defaultdict(list)
    for category, node in zip(category_list_1, node_list_1):
        category2node_list[category].append(node)

    if target == 'choice':
        category_list_2 = pd.read_excel('data/choice金融终端整理的战新企业名单（科创板含意向申报）.xlsx', dtype=str)[
            '所属战新行业'].values.tolist()
        node_list_2 = pd.read_excel('data/choice金融终端整理的战新企业名单（科创板含意向申报）.xlsx', dtype=str)[
            '证券代码'].values.tolist()
        # clean for node_list_2
        node_list_2 = [node[:6] for node in node_list_2]
        category2node_list_2 = defaultdict(list)
        for category, node in zip(category_list_2, node_list_2):
            category2node_list_2[category].append(node)

    elif target == 'tongjiju':
        category_list_2 = pd.read_excel('data/战新上市公司2019和2021对比-230112.xlsx', dtype=str)[
            '所属战新八大领域'].values.tolist()
        node_list_2 = pd.read_excel('data/战新上市公司2019和2021对比-230112.xlsx', dtype=str)[
            '证券代码'].values.tolist()
        # clean for node_list_2
        node_list_2 = [node[:6] for node in node_list_2]

        category2node_list_2 = defaultdict(list)
        for category, node in zip(category_list_2, node_list_2):
            category2node_list_2[category].append(node)

    elif target == 'top100':
        category2node_list_2 = get_top100()
    else:
        raise ValueError('target should be choice or tongjiju')

    for category in category2node_list.keys():
        len_1 = len(set(category2node_list[category]))
        len_2 = len(set(category2node_list_2[category]))
        num_pos = len(set(category2node_list[category]) & set(category2node_list_2[category]))
        num_all = len(set(category2node_list_2[category]))
        new = len(set(category2node_list[category]) - set(category2node_list_2[category]))
        if num_all == 0:
            print(category, len_1, "-", "-", new)
        else:
            print(category, len_1, len_2, num_pos / num_all, new)


if __name__ == '__main__':
    eval_node(target='choice')
    eval_node(target='tongjiju')
    eval_node(target='top100')
