# 筛选出组成词汇表的单词到文件


import numpy as np
import os
import json
from functools import reduce
from collections import Counter
from utils.functions import *

import argparse

parser = argparse.ArgumentParser(description="Demo of argparse")
parser.add_argument('--abbreviation', type=bool, default=False)
parser.add_argument('--yeast', type=bool, default=False)
parser.add_argument('--mix', type=bool, default=False)

args = parser.parse_args()

#获取当前文件所在的目录
current_path = os.path.abspath(__file__)
abs_file_path = os.path.dirname(current_path)

#设定预处理文件类别
abbreviation = args.abbreviation  #是否使用缩写
yeast = args.yeast         #是否使用酵母数据集
mix = args.mix             #是否使用混合数据集
no_unk = True         #确保term_name中的单词全部进入词汇表

if abbreviation:
    ab_append = "@"
else:
    ab_append = ""

if yeast:
    yeast_append = "_yeast"
else:
    yeast_append = ""

if mix:
    mix_append = "_mix"
else:
    mix_append = ""

# 加载预处理文件
term2Gene_file = open(abs_file_path + "/data/processed_data/shuffle_Onto2Gene"+yeast_append+mix_append+".json", "r", encoding='utf-8')  #本体代号 -> 基因代号
geneDe_file = open(abs_file_path + "/data/processed_data/all_geneDe"+ab_append+yeast_append+mix_append+".json", "r", encoding='utf-8')  #基因代号 -> 基因描述
idName_file = open(abs_file_path + "/data/processed_data/idName"+ab_append+yeast_append+mix_append+".json", "r", encoding='utf-8')      #本体代号 -> 本体名称
term2gene = json.loads(term2Gene_file.read())
geneId2content = json.loads(geneDe_file.read())
termId2tn = json.loads(idName_file.read())

# 设置词汇表保存路径
save_path = abs_file_path + "/data/vocabulary/vocab" + ab_append + yeast_append + mix_append + ".txt"

#读取训练集，常用命名空间应当在训练集上构造
train_output_path = abs_file_path + "/data/input_data/gcn_train_dataset"+ab_append+yeast_append+mix_append+".txt"
train_data_set = load_corpus(train_output_path)
train_data = train_data_set


#从原始数据中筛选出训练集的数据
term_Ids = []
gene_Ids = []

for (termId, vocab, adj) in train_data:
    term_Ids.append(termId)
    geneIds = term2gene[termId]
    gene_Ids += geneIds

gene_Ids = list(set(gene_Ids))
train_geneId2content = {}
train_termId2tn = {}

for geneId, (geneName, geneDe) in geneId2content.items():
    if geneId in gene_Ids:
        train_geneId2content[geneId] = (geneName, geneDe)

for termId, termName in termId2tn.items():
    if termId in term_Ids:
        train_termId2tn[termId] = termName


#设置筛选规则
min_appera = 1     #设置单词能够进入词汇表的最小词频数


gene_words = []    #保存所有筛选出来的单词
term_words = []    #保存所有term_name中的单词，方便对这些单词做特殊的处理

all_words = [] #保存所有单词


#获取数据集中所有单词
for geneId, (geneName, geneDe) in train_geneId2content.items():
    all_words += set(geneName.split() + geneDe.split())
    gene_words += geneName.split() + geneDe.split()

all_words_count = Counter(all_words).items()
all_words_idf = []
gene_nums = len(train_geneId2content.items())
for word, weight in all_words_count:
    all_words_idf.append((word, np.log(gene_nums/(weight+1))))

all_words_idf = sorted(all_words_idf, key = lambda x:x[1], reverse=True)


for termId, termName in train_termId2tn.items():
    # words += termName.split()
    term_words += termName.split()

term_words = list(set(term_words))


#这里把单词排序后单独输入一个文件，每一行为 word idf-weights
sorted_vocab_path = abs_file_path + "/data/vocabulary/sorted_vocab" + ab_append + yeast_append + mix_append + ".txt"
gene_words = Counter(gene_words)
count = 0
with open(sorted_vocab_path, 'w', encoding='utf-8') as f:
    for word in term_words:
        f.write(word + '\t' + '<term>' + '\n')
        count += 1
    for word, weights in all_words_idf:
        if word not in term_words and gene_words[word] >= min_appera:
            f.write(word + '\t' + str(weights) + '\n')
            count += 1
print(count)


