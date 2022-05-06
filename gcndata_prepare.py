
# 从文本数据集构造用于GCN模型的数字数据集

import os
import numpy as np
import json
import random
import scipy.sparse as sp
import pickle as pkl
from math import log
from tqdm import tqdm
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
yeast = args.yeast        #是否使用酵母数据集
mix = args.mix         #是否使用混合数据集

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


# 设置划分数据集的超参数
# most_seq_nodes = 10 #单个gene_text最多允许的句子节点数量
# most_token_nodes = 100 #在一个图中最多允许存在的单词节点数量
most_seq = 200  #gene_text的最大长度
max_oovs = 100  #最大允许的oov单词的数量，影响embeding层大小
window_size = 20 #计算单词之间共现次数的窗口大小


# 加载预处理文件
term2Gene_file = open(abs_file_path + "/data/processed_data/shuffle_Onto2Gene"+yeast_append+mix_append+".json", "r", encoding='utf-8')  #本体代号 -> 基因代号
geneDe_file = open(abs_file_path + "/data/processed_data/all_geneDe"+ab_append+yeast_append+mix_append+".json", "r", encoding='utf-8')  #基因代号 -> 基因描述
idName_file = open(abs_file_path + "/data/processed_data/idName"+ab_append+yeast_append+mix_append+".json", "r", encoding='utf-8')      #本体代号 -> 本体名称
term2gene = json.loads(term2Gene_file.read())
geneId2content = json.loads(geneDe_file.read())
termId2tn = json.loads(idName_file.read())


#打乱数据集
dict_key_ls = list(term2gene.keys())
random.shuffle(dict_key_ls)
new_dic = {}
for key in dict_key_ls:
    new_dic[key] = term2gene.get(key)
term2gene = new_dic

#保存文件位置
output_path = abs_file_path + "/data/input_data/gcn_dataset"+ab_append+yeast_append+".txt"


#构造数据集， 将geneName和geneDe末尾添加'<EOS>'并连接
results = {}
print('construct data...')
for termId, geneIds in tqdm(term2gene.items()):
    shuffle_doc_words_list = []  #保存处理好的数据集
    geneContents = [geneId2content[geneId] for geneId in geneIds]
    # termName = termId2tn[termId] + ' <EOS>'
    
    gene_nums = len(geneContents)
    for (geneName, geneDe) in geneContents:
        shuffle_doc_words_list.append(geneName + ' <EOS> ' + geneDe + ' <EOS>')

    # build vocab
    word_freq = {}
    word_set = set()
    for doc_words in shuffle_doc_words_list:
        words = doc_words.split()
        for word in words:
            word_set.add(word)
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1

    vocab = list(word_set)
    vocab_size = len(vocab)
    
    word_doc_list = {}  #word对应出现的doc序号列表

    for i in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        appeared = set() #确保每个word对应的相同doc编号只出现一次
        for word in words:
            if word in appeared:
                continue
            if word in word_doc_list:
                doc_list = word_doc_list[word]
                doc_list.append(i)
                word_doc_list[word] = doc_list
            else:
                word_doc_list[word] = [i]
            appeared.add(word)

    word_doc_freq = {}
    for word, doc_list in word_doc_list.items():
        word_doc_freq[word] = len(doc_list)

    word_id_map = {}
    for i in range(vocab_size):
        word_id_map[vocab[i]] = i

    # word co-occurence with context windows
    windows = []

    for doc_words in shuffle_doc_words_list:
        words = doc_words.split()
        length = len(words)
        if length <= window_size:
            windows.append(words)
        else:
            # print(length, length - window_size + 1)
            for j in range(length - window_size + 1):
                window = words[j: j + window_size]
                windows.append(window)
                # print(window)


    word_window_freq = {}
    for window in windows:
        appeared = set()
        for i in range(len(window)):
            if window[i] in appeared:
                continue
            if window[i] in word_window_freq:
                word_window_freq[window[i]] += 1
            else:
                word_window_freq[window[i]] = 1
            appeared.add(window[i])

    word_pair_count = {}
    for window in windows:
        for i in range(1, len(window)):
            for j in range(0, i):
                word_i = window[i]
                word_i_id = word_id_map[word_i]
                word_j = window[j]
                word_j_id = word_id_map[word_j]
                if word_i_id == word_j_id:
                    continue
                word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1
                # two orders
                word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1

    row = []
    col = []
    weight = []

    # pmi as weights

    num_window = len(windows)

    for key in word_pair_count:
        temp = key.split(',')
        i = int(temp[0])
        j = int(temp[1])
        count = word_pair_count[key]
        word_freq_i = word_window_freq[vocab[i]]
        word_freq_j = word_window_freq[vocab[j]]
        pmi = log((1.0 * count / num_window) /
                (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
        if pmi <= 0:
            continue
        row.append(gene_nums + i)
        col.append(gene_nums + j)
        weight.append(pmi)

    # doc word frequency
    doc_word_freq = {}

    for doc_id in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[doc_id]
        words = doc_words.split()
        for word in words:
            word_id = word_id_map[word]
            doc_word_str = str(doc_id) + ',' + str(word_id)
            if doc_word_str in doc_word_freq:
                doc_word_freq[doc_word_str] += 1
            else:
                doc_word_freq[doc_word_str] = 1

    for i in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        doc_word_set = set()
        for word in words:
            if word in doc_word_set:
                continue
            j = word_id_map[word]
            key = str(i) + ',' + str(j)
            freq = doc_word_freq[key]
            row.append(i)
            col.append(gene_nums + j)
            idf = log(1.0 * len(shuffle_doc_words_list) /
                    word_doc_freq[vocab[j]])
            weight.append(freq * idf)
            doc_word_set.add(word)

    node_size = gene_nums + vocab_size
    adj = sp.csr_matrix((weight, (row, col)), shape=(node_size, node_size))
    results[termId] = (vocab, adj)




#按照训练集，测试集保存两个单独文件
train_output_path = abs_file_path + "/data/input_data/gcn_train_dataset"+ab_append+yeast_append+mix_append+".txt"
test_output_path = abs_file_path + "/data/input_data/gcn_test_dataset"+ab_append+yeast_append+mix_append+".txt"
train_rate = 0.8
data = list(results.items())
train_length = int(len(data)*train_rate)
train_data = data[:train_length]
test_data = data[train_length:]
train_data = dict(train_data)
test_data = dict(test_data)


with open(train_output_path, 'wb') as f:
    pkl.dump(train_data, f)

with open(test_output_path, 'wb') as f:
    pkl.dump(test_data, f)




