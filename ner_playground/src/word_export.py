import pandas as pd
import json
from openpyxl import Workbook


# 读取JSON数据
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


time_span = '145'
# 假设文件路径
file1 = f'../data/word_base_{time_span}.json'
file2 = f'../data/战新词表_topmine_top50_{time_span}_labeled_combine.json'
file3 = f'../data/战新词表_ner_{time_span}_labeled.json'

data1 = read_json(file1)
data2 = read_json(file2)
data3 = read_json(file3)

# 以文件1的key顺序为基准
keys_order = list(data1.keys())

# 创建Excel工作簿和工作表
wb = Workbook()
ws = wb.active

# 首行写入类型
ws.append([''] + keys_order)  # 第一个单元格留空


# 写入数据，优化空白行处理
def append_data(ws, data, label, keys_order):
    # 使用字典记录每个key对应的最大行数
    max_rows = {key: 0 for key in keys_order}
    data_rows = []

    # 先统计每个类型下的最大词数
    for key in keys_order:
        words = data.get(key, [])
        if words:
            max_rows[key] = len(words)

    # 根据最大词数设置行数据
    for i in range(max(max_rows.values())):
        row = [label if i == 0 else '']
        for key in keys_order:
            words = data.get(key, [])
            if i < len(words):
                row.append(words[i])
            else:
                row.append('')
        data_rows.append(row)

    # 删除全空的行，除了第一个单元格可能有标签
    data_rows = [row for row in data_rows if any(cell for j, cell in enumerate(row) if cell or j == 0)]

    # 写入行数据
    for row in data_rows:
        ws.append(row)


append_data(ws, data1, '基础词表', keys_order)
append_data(ws, data2, '核心词表', keys_order)
append_data(ws, data3, '扩展词表', keys_order)

# 保存文件
wb.save(f'战新词表_{time_span}.xlsx')
