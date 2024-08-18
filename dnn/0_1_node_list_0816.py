from collections import Counter, defaultdict
import json
import pandas as pd


def get_th_80(num_type_count, rate):
    """
    获取80%的阈值
    """
    total = sum(num_type_count.values())
    th_80 = 0
    th_80_count = 0
    for num, count in num_type_count.items():
        th_80_count += count
        # if 0.75 < th_80_count / total < 0.85:
        if th_80_count / total > rate:
            th_80 = num
            break
    return th_80


def get_node_list_core():
    """
    企业合并
    按照125(2011-2015),135(2016-2020),145(2021-2025)进行合并
    """
    # core
    year2category2node_list = {}
    year2category2num_node = defaultdict(lambda: defaultdict(int))

    category2index = {'高端装备制造': 0,
                      '节能环保': 1,
                      '生物': 2,
                      '数字创意': 3,
                      '新材料': 4,
                      '新能源': 5,
                      '新能源汽车': 6,
                      '新一代信息技术': 7}

    for year in range(2011, 2024):
        # load 2 dict: company2category2num
        print(f'year: {year}')
        df = pd.read_excel(f'data/node_list/node_list_core/{year}_num_type.xlsx', dtype=str)
        category2company2num_type = df.set_index('Unnamed: 0').to_dict()
        df = pd.read_excel(f'data/node_list/node_list_core/{year}_num_word.xlsx', dtype=str)
        category2company2num_word = df.set_index('Unnamed: 0').to_dict()
        category2node_list = {}
        for category in category2index:
            company2num_type = category2company2num_type[category]
            company2num_word = category2company2num_word[category]
            # value2int
            company2num_type = {company: int(num_type) for company, num_type in company2num_type.items()}
            company2num_word = {company: int(num_word) for company, num_word in company2num_word.items()}
            # 根据 80-20 获取阈值
            num_type_count = Counter(company2num_type.values())
            num_word_count = Counter(company2num_word.values())
            # sorted by key
            num_type_count = dict(sorted(num_type_count.items(), key=lambda x: x[0]))
            num_word_count = dict(sorted(num_word_count.items(), key=lambda x: x[0]))
            # remove 0
            num_type_count.pop(0, None)
            num_word_count.pop(0, None)
            num_type_th_80 = get_th_80(num_type_count, rate=0.8)
            num_word_th_80 = get_th_80(num_word_count, rate=0.8)
            # get node_list and add to dict
            node_list_num_type = [node for node, num_type in company2num_type.items() if num_type > num_type_th_80]
            node_list_num_word = [node for node, num_word in company2num_word.items() if num_word > num_word_th_80]
            node_list = list(set(node_list_num_type) | set(node_list_num_word))
            category2node_list[category] = node_list
            # record
            year2category2num_node[year][category] = len(node_list)

        year2category2node_list[year] = category2node_list

    return year2category2node_list


def get_category2code_list(time):
    """
    获取战新行业代码
    """
    df = pd.read_excel(f'data/node_list/node_list_add/zx{time}.xlsx', sheet_name='战新行业代码', dtype=str)
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


def load_add_node_id_list(year):
    """
    load node_id_list
    """
    df = pd.read_excel(f'data/node_list/node_list_add/{year}.xlsx', dtype=str)
    node_id_list = df['node_id'].values.tolist()
    return node_id_list


def get_node_list_add():
    code2category_set_18 = get_category2code_list(18)
    code2category_set_23 = get_category2code_list(23)

    # load node_list
    year2node_list = {
        2019: load_add_node_id_list(2019),
        2020: load_add_node_id_list(2020),
        2021: load_add_node_id_list(2021),
        2022: load_add_node_id_list(2022),
        2023: load_add_node_id_list(2023)
    }

    year2category2node_list = defaultdict(lambda: defaultdict(list))

    # load 全部A股-行业代码.xlsx
    df = pd.read_excel('data/node_list/node_list_add/全部A股-行业代码.xlsx', dtype=str)
    node_id_list = df['证券代码'].values.tolist()
    node_code_list = df['所属国民经济行业代码(2017)'].values.tolist()
    node_id2node_code = {node_id[:6]: node_code.split('-')[-1][1:]
                         for node_id, node_code in zip(node_id_list, node_code_list)}

    for year, node_list in year2category2node_list.items():
        for node_id in node_list:
            if node_id not in node_id2node_code:
                continue
            code = node_id2node_code[node_id]
            if year in [2019, 2020]:
                category_set = code2category_set_18.get(code, set())
            else:
                category_set = code2category_set_23.get(code, set())
            for category in category_set:
                year2category2node_list[year][category].append(node_id)

    return year2category2node_list


if __name__ == '__main__':
    year2category2node_list_code = get_node_list_core()
    year2category2node_list_add = get_node_list_add()
    # combine
    year2category2node_list = defaultdict(lambda: defaultdict(list))

    for year in range(2011, 2024):
        for category in year2category2node_list_code[year]:
            if year in year2category2node_list_add:
                node_list = year2category2node_list_code[year][category] + year2category2node_list_add[year][category]
            else:
                node_list = year2category2node_list_code[year][category]
            node_list = list(set(node_list))
            print(f'year: {year}, category: {category}, num: {len(node_list)}')
            year2category2node_list[year][category] = node_list

    # with open('data/year2category2node_list.json', 'w', encoding='utf-8') as f:
    #     json.dump(year2category2node_list, f, ensure_ascii=False, indent=4)
