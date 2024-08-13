# pdf trans to text
import re
import os
import multiprocessing as mp
from functools import partial


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


def deal_2(input_file_path, output_file_path):
    """
    get title
    """
    print('---------------------deal_2---------------------')
    input_file_list = os.listdir(input_file_path)
    input_file_list = sorted(input_file_list, key=lambda x: int(x[:4]))

    title_list = []
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
        func = partial(text_cut_single, load_file_path, save_file_path)
        title_list_list = pool.map(func, load_file_list)
        pool.close()
        pool.join()

        for title_list_ in title_list_list:
            title_list += title_list_
    # save title
    with open(f'title.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(title_list))


def load_title():
    """
    加载title
    :return:
    """
    title_path = f'title.txt'
    with open(title_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    title_list = [l.strip() for l in lines]
    print('title num: {}'.format(len(title_list)))
    title_list = list(set(title_list))
    print('title num(combine): {}'.format(len(title_list)))
    # remove context
    title_list = [l for l in title_list if '………' not in l]
    print('title num(remove context): {}'.format(len(title_list)))
    key_words_p = ['经营', '产品', '创新', '策略', '业务', '研发', '核心竞争力', '决策', '风险']
    title_list = [l for l in title_list if any([kw in l for kw in key_words_p])]
    key_words_n = ['现金', '金额', '表', '财务', '资产', '负债', '利润', '收入', '成本', '费用', '现金流', '利润表',
                   '资产负债表', '现金流量表']
    title_list = [l for l in title_list if not any([kw in l for kw in key_words_n])]
    print('title num(clean): {}'.format(len(title_list)))
    # save
    title_path = f'title_screen.txt'
    with open(title_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(title_list))
    print('save title_screen.txt')


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


def deal_3(input_file_path, output_file_path):
    print('---------------------deal_3---------------------')
    input_file_list = os.listdir(input_file_path)
    input_file_list = sorted(input_file_list, key=lambda x: int(x[:4]))
    # title_screen
    with open(f'title_screen.txt', 'r', encoding='utf-8') as f:
        title_screen_list = f.readlines()

    for input_file in input_file_list:
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


if __name__ == "__main__":
    deal_2(input_file_path='../data/output/', output_file_path='../data/clean_1/')
    load_title()
    deal_3(input_file_path='../data/clean_1/', output_file_path='../data/clean_2/')
