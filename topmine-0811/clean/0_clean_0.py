# pdf trans to text
import pdfplumber
import os
import multiprocessing as mp
from functools import partial


def pdf2text(pdf_file_path, save_path, file):
    """

    :param pdf_file_path:
    :param save_path:
    :param file:
    :return:
    """
    # pdf_path = 'test.pdf'
    pdf_path = os.path.join(pdf_file_path, file)
    try:
        pdf = pdfplumber.open(pdf_path)
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
            page.flush_cache()
        pdf.close()
        txt_path = os.path.join(save_path, file.split('.')[0] + '.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
    except:
        # save to error file
        error_file_path = 'error.txt'
        with open(error_file_path, 'a', encoding='utf-8') as f:
            f.write(file + '\n')


def deal_1():
    """
    deal with pdf to text
    """
    input_file_path = 'data/input/'
    output_file_path = 'data/output/'
    input_file_list = os.listdir(input_file_path)
    input_file_list = sorted(input_file_list, key=lambda x: int(x[:4]))

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
        func = partial(pdf2text, load_file_path, save_file_path)
        pool.map(func, load_file_list)
        pool.close()
        pool.join()


if __name__ == "__main__":
    deal_1()
