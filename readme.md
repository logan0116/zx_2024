2.1 频繁短语挖掘

input路径：'../../data/text_125.txt'

output路径：'../../data/topmine/partitioneddocs.txt' '../../data/topmine/vocab.txt'

运行代码：[run_phrase_mining.py](topmine%2Ftopmine-master%2Ftopmine_src%2Frun_phrase_mining.py)

input路径：'../../data/topmine/partitioneddocs.txt' '../../data/topmine/vocab.txt'

input路径：'data/topmine/keywords_multiple.xlsx'

运行代码：[2_topmine_add.py](topmine%2F2_topmine_add.py)

2.2 短语与领域关联

前置条件：需要构建[word_base.xlsx](topmine%2Fdata%2Fword_base.xlsx)

做法：将重点产品服务指导目录的“目录”整理成表

input路径：'data/topmine/keywords_multiple.xlsx' 'data/word_base.xlsx'

output路径：'data/战新词表_topmine_top50.xlsx'

运行代码：[3_embedding.py](topmine%2F3_embedding.py)



