# 处理title
import re
import jieba
from collections import Counter


def load_title():
    """
    加载title
    :return:
    """
    title_path = 'title.txt'
    with open(title_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    title_list = [l.strip() for l in lines]
    print(len(title_list))
    title_list = list(set(title_list))
    print(len(title_list))
    # remove context
    title_list = [l for l in title_list if '………' not in l]
    print(len(title_list))
    key_words_p = ['经营', '产品', '创新', '策略', '业务', '研发', '核心竞争力', '决策', '风险']
    title_list = [l for l in title_list if any([kw in l for kw in key_words_p])]
    key_words_n = ['现金', '金额', '表', '财务', '资产', '负债', '利润', '收入', '成本', '费用', '现金流', '利润表',
                   '资产负债表', '现金流量表']
    title_list = [l for l in title_list if not any([kw in l for kw in key_words_n])]
    print(len(title_list))
    # save
    title_path = 'title_screen.txt'
    with open(title_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(title_list))


if __name__ == '__main__':
    load_title()
    # get_word()
