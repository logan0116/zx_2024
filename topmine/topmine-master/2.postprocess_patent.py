import json
import re


def deal(prepare_path, json_write_path):
    txt_read = open(prepare_path, 'r', encoding='UTF-8')
    pattern = re.compile(r"['](.*?)[']", re.S)
    point_list = []
    for each_line in txt_read:
        point_list.append(re.findall(pattern, each_line)[0])

    json.dump(point_list, open(json_write_path, 'w', encoding='UTF-8'))


if __name__ == '__main__':
    prepare_path = 'output/frequent_phrases.txt'
    json_write_path = 'output/phrase_list_patent.json'
    deal(prepare_path, json_write_path)
