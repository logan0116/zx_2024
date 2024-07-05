""" Implements the algorithm provided in the following research paper:

El-Kishky, Ahmed, et al. "Scalable topical phrase mining from text corpora." Proceedings of the VLDB Endowment 8.3 (2014): 305-316.
"""
import shlex
import subprocess


def get_output_of(command):
    args = shlex.split(command)
    return subprocess.Popen(args, stdout=subprocess.PIPE).communicate()[0]


if __name__ == '__main__':
    # filename
    file_name = "input/topmine_input.txt"

    # 默认值为 4
    # num_topics = 4

    phrase_mining_cmd = "python3 topmine_src/run_phrase_mining.py {0}".format(file_name)
    print(get_output_of(phrase_mining_cmd))

# phrase_lda_cmd = "python topmine_src/run_phrase_lda.py {0}".format(num_topics)
# print(get_output_of(phrase_lda_cmd))
