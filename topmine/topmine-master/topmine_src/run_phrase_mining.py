import phrase_mining
import sys
import utils

print('Running Phrase Mining...')

file_name = '../../data/text_125.txt'

# represents the minimum number of occurences you want each phrase to have.
min_support = 10

# represents the threshold for merging two words into a phrase. A lower value
# alpha leads to higher recall and lower precision,
alpha = 2

# length of the maximum phrase size
max_phrase_size = 12

phrase_miner = phrase_mining.PhraseMining(file_name, min_support, max_phrase_size, alpha)
partitioned_docs, index_vocab = phrase_miner.mine()
frequent_phrases = phrase_miner.get_frequent_phrases(min_support)
utils.store_partitioned_docs(partitioned_docs, path='../../data/topmine/partitioneddocs.txt')
utils.store_vocab(index_vocab, path='../../data/topmine/vocab.txt')
# utils.store_frequent_phrases(frequent_phrases)
