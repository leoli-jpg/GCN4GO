import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import torch.nn.functional as F
from models.gcn import GCN
from models.mlp import MLP
from models.copynet_dbg import *
from utils.vocab import Vocab
from utils.functions import *
from utils.calculatebleu import BLEU
from utils.abbre import load_abbre, unpackAbbre
import scipy.sparse as sp
import pickle as pkl
import datetime
import random
import json
import time
import sys
import math
import os

import argparse

parser = argparse.ArgumentParser(description="Demo of argparse")
parser.add_argument('--abbreviation', type=bool, default=False)
parser.add_argument('--yeast', type=bool, default=False)
parser.add_argument('--mix', type=bool, default=False)
parser.add_argument('--attention', type=bool, default=False)
parser.add_argument('--model', type=str, default="")

args = parser.parse_args()

def embed_batch(batch):
    """数据通过整个模型，该函数是模型训练和测试中共享的部分，返回用于解码的特征"""
    batch_size = len(batch)
    #构造adj矩阵
    batch_row = []
    batch_col = []
    batch_data = []
    gene_nums = []
    node_nums = []
    base_num = 0
        
    for (termId, vocab, adj) in batch:
        adj = adj.tocoo()
        row, column, value = adj.row, adj.col, adj.data
        batch_row += (row + base_num).tolist()
        batch_col += (column + base_num).tolist()
        batch_data += value.tolist()
        node_nums.append(adj.shape[0])
        gene_nums.append(node_nums[-1] - len(vocab))
        base_num += node_nums[-1]

    #构造term节点
    base_col = 0
    for i in range(batch_size):
        batch_row += [i+base_num]*gene_nums[i]
        batch_col += [j+base_col for j in range(gene_nums[i])]
        batch_data += [1]*gene_nums[i]
        base_col += node_nums[i]

    #构造adj
    node_size = base_num + batch_size #添加term节点
    adj = sp.csr_matrix((batch_data, (batch_row, batch_col)), shape=(node_size, node_size))
    batch_adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #转化为对称矩阵

    #构造输入输出
    gene_texts = []
    gene_nums = []
    targets = []
    vocabs = []
    for (termId, vocab, adj) in batch:
        geneIds = term2gene[termId]
        geneContents = [geneId2content[geneId] for geneId in geneIds]
        termName = termId2tn[termId] + ' <EOS>'
        gene_texts += [geneName+' <EOS> '+geneDe+' <EOS>' for (geneName, geneDe) in geneContents]
        gene_nums.append(len(geneContents))
        vocabs.append(vocab)
        targets.append(termName)
        

    #对输入输出进行编码
    idx_gene_texts = [all_vocab.word_list_to_idx_list(gene_text.split()) for gene_text in gene_texts]
    idx_gene_texts = [' '.join([str(token) for token in gene_text]) for gene_text in idx_gene_texts]
    idx_targets = [all_vocab.word_list_to_idx_list(target.split()) for target in targets]
    idx_targets = [' '.join([str(token) for token in target]) for target in idx_targets]

    input_out, in_len, x_un_sort_idx = toData(idx_gene_texts) #Pading并且按照之前长度倒序排列
    output_out, out_len, y_un_sort_idx = toData(idx_targets) #Pading并且按照之前长度倒序排列
        

    # input and output in Variable form
    x = numpy_to_var(input_out).to(device)
    y = numpy_to_var(output_out).to(device)


    y = y.index_select(0, torch.tensor(y_un_sort_idx).to(device))
    out_len = out_len[y_un_sort_idx]

    # apply to encoder
    encoded, hidden = encoder(x, in_len)
    encoded = encoded.index_select(0, torch.tensor(x_un_sort_idx).to(device))
    hidden = hidden.transpose(0,1).contiguous().view((x.shape[0],-1))
    hidden = hidden.index_select(0, torch.tensor(x_un_sort_idx).to(device))
    # encoded = mlp1(encoded)
    hidden = mlp2(hidden)
        
    # 准备GCN的输入
    term_hidden_list = []
    encoder_hidden_list = []
    seq_nums = []
    base_num = 0
    for i in range(batch_size):
        gene_hiddens = hidden[base_num:(base_num+gene_nums[i])]

        encoder_hidden_list.append(gene_hiddens)
        seq_nums.append(gene_hiddens.shape[0])


        base_num += gene_nums[i]
        max_pooling = nn.MaxPool2d((gene_hiddens.shape[0], 1), 1)
        term_hidden = max_pooling(gene_hiddens.unsqueeze(0)).squeeze(0)
        term_hidden_list.append(term_hidden)

        if i == 0:
            gcn_input = gene_hiddens
        else:
            gcn_input = torch.cat((gcn_input, gene_hiddens))
            
        idx_vocab = torch.tensor(all_vocab.word_list_to_idx_list(vocabs[i])).to(device)
        embeded_vocab = mlp3(encoder.embed(idx_vocab))
        # embeded_vocab = encoder.embed(idx_vocab)
        gcn_input = torch.cat((gcn_input, embeded_vocab))
        
    gcn_input = torch.cat((gcn_input, mlp1(torch.cat(term_hidden_list))))
    # gcn_input.register_hook(save_grad('gcn_input'))

    #数据通过GCN
    support = [torch.FloatTensor(preprocess_adj(batch_adj)).to(device)]
    features = model(gcn_input, support)
        
    #分类获取经过GCN编码后的高层特征
    term_hiddens = features[-batch_size:]
    # term_hiddens.register_hook(save_grad('term_hiddens'))

    #提取所有vocab的高层特征，这部分特征可以用于decoder端的embedding输入
    node_embeddings = []
    batch_vocabs = []
    index_base = 0

    for vocab, gene_num in zip(vocabs, gene_nums):
        index_base += gene_num   #skip gene_node
        node_embeddings.append(features[index_base:(index_base+len(vocab))])
        batch_vocabs.append(vocab)
        index_base += len(vocab) #next data
    
    x = x.index_select(0, torch.tensor(x_un_sort_idx).to(device))
    in_len = in_len[x_un_sort_idx]

    return term_hiddens, y, out_len, node_embeddings, batch_vocabs, encoder_hidden_list, seq_nums

def decode_batch(term_hiddens, y, with_truth, node_embeddings, batch_vocabs, encoder_hidden_list, seq_nums):
    """数据通过整个模型，该函数是模型训练和测试中共享的部分，返回解码结果"""
    # 经过decoder做生成
    # get initial input of decoder
    decoder_in, s, w = decoder_initial(term_hiddens.shape[0])
    decoder_in = decoder_in.to(device)
    # ini_c = torch.zeros(term_hiddens.shape).unsqueeze(0).to(device) #使用LSTM作为解码器时添加

    for j in range(y.shape[1]): # for all sequences
        # 1st state
        if j==0:
            out, hidden, w = decoder(decoder_in, term_hiddens, w, all_vocab, node_embeddings, batch_vocabs, encoder_hidden_list, seq_nums)
        # remaining states
        else:
            tmp_out, hidden, w = decoder(decoder_in, hidden, w, all_vocab, node_embeddings, batch_vocabs, encoder_hidden_list, seq_nums)
            out = torch.cat([out,tmp_out],dim=1)

        # select next input
        if with_truth and random.random() < teach_force_rate:
            decoder_in = y[:,j] # train with ground truth
        # if with_truth:
        #     decoder_in = y[:,j] # train with ground truth
        else:  #torch.max中输出为元组，第一个表示最大数值，第二个表示最大下标
            decoder_in = out[:,-1].max(1)[1] # train with sequence outputs
            # print('hh', decoder_in.shape)
        
    return out
    
    
# Define model evaluation function
def evaluate(data_set):
    """以文本形式返回传入数据集的预测结果"""
    encoder.eval()
    model.eval()
    mlp1.eval()
    mlp2.eval()
    mlp3.eval()
    # mlp4.eval()
    decoder.eval()

    pred_text = []
    ground_truth = []

    #obtain ground truth
    for (termId, vocab, adj) in data_set:
        termName = termId2tn[termId]
        if abbreviation:
            termName = unpackAbbre(termName)
        ground_truth.append(termName)

    # obtain batch outputs
    samples_read = 0
    while(samples_read<len(data_set)):
        batch = data_set[samples_read:min(samples_read+test_batch_size,len(data_set))]
        cur_batch_size = len(batch)
        samples_read += cur_batch_size

        term_hiddens, y, out_len, node_embeddings, batch_vocabs, encoder_hidden_list, seq_nums = embed_batch(batch)

        # out = decode_batch(term_hiddens, y, False, node_embeddings, batch_vocabs)  #with_truth在这里必须为False，因为现在是测试
        out = decode_batch(term_hiddens, y, False, None, None, encoder_hidden_list, seq_nums)  #with_truth在这里必须为False，因为现在是测试

        #获取文本输出
        pred_len = {}  #标记每个预测的实际长度以"<EOS>"结尾
        pred_out = torch.argmax(out, -1)
        EOS_index = torch.nonzero(pred_out == 3)
        # print('后面的数值越接近batch_size说明预测越准确：',EOS_index.shape)
        for i in range(EOS_index.shape[0]):
            posi = EOS_index[i]
            if posi[0].item() in pred_len.keys():
                continue
            else:
                pred_len[posi[0].item()] = posi[1].item()

        for i in range(pred_out.shape[0]):
            if i in pred_len.keys():
                length = pred_len[i]
            else:
                length = max_length
            pred_idx = [str(idx.item()) for idx in pred_out[i][:length]]
            if len(pred_idx) == 0:
                pred_text.append('<no-words>')
            else:
                pred = ' '.join(all_vocab.idx_list_to_word_list(pred_idx))
                if abbreviation:
                    pred = unpackAbbre(pred)
                pred_text.append(pred)
            # pred_idx = [str(idx.item()) for idx in y[i][:(out_len[i]-1)]] # remove '<EOS>'
            # ground_truth.append(' '.join(all_vocab.idx_list_to_word_list(pred_idx)))

    encoder.train()
    model.train()
    mlp1.train()
    mlp2.train()
    mlp3.train()
    # mlp4.train()
    decoder.train()
    # print(len(pred_text), len(ground_truth))

    return pred_text, ground_truth



def save_results(pred_text, ground_truth, score_format_list, save_path):
    """保存文字结果到指定路径"""
    f = open(save_path, 'w', encoding='utf-8')
    count = 0
    for text, truth, score in zip(pred_text, ground_truth, score_format_list):
        count += 1
        f.write('number_' + str(count) + '\n')
        f.write('groud_truth : \t' + truth + '\n')
        f.write('pred_text : \t' + text + '\n')
        f.write(score + '\n')



# a = ['hh']
# print(len(a))
# a.append('')
# print(len(a))
# exit(0)

#获取当前文件所在的目录
current_path = os.path.abspath(__file__)
abs_file_path = os.path.dirname(current_path)

#设定预处理文件类别
abbreviation = args.abbreviation  #是否使用缩写
yeast = args.yeast          #是否使用酵母数据集
mix = args.mix           #是否使用混合数据集
with_attention = args.attention  #是否使用注意力机制

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

#加载词汇表
all_vocab = Vocab(50000)    #参数为词汇表允许容纳的最大单词数量，没有什么用处，会在超出容量的时候报错提醒
vocab_path = abs_file_path + "/data/vocabulary/" + "sorted_vocab" + ab_append + yeast_append + mix_append + ".txt"
words = []     #保存词汇表单词列表

with open(vocab_path, 'r', encoding='utf-8') as f:
    for line in f:
        word, count = line.split()
        words.append(word)

all_vocab.add_to_vocab(words)
vocab_size = all_vocab.count
print("词汇表数目：", vocab_size)

#设置模型的相关超参数
num_epochs = 100   #epoch for train
eval_epoch = 5    #evaluate per eval_epoch    
teach_force_rate = 1 #rate for train with ground truth
batch_size = 6   #用于训练集的batch_size
test_batch_size = 6 #用于测试集的batch_size
max_length = 20   #decoder生成序列的最大长度
max_annotates = 5 #term注释的最大gene数量
embed_size = 300  #编码的词向量大小
input_dim = 300  #输入GCN的向量大小
hidden_dim = 300  #GCN中间隐藏层向量大小
output_dim = 300  #GCN输出向量大小
train_rate = 0.8
val_rate = 0.1
test_rate = 0.1
lr = 1e-4
weight_decay = 0.99
device = torch.device('cuda')

#获取当前时间
nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')

# Load data
# output_path = abs_file_path + "/data/input_data/gcn_dataset"+ab_append+yeast_append+".txt"
# data_set = load_corpus(output_path)
train_output_path = abs_file_path + "/data/input_data/gcn_train_dataset"+ab_append+yeast_append+mix_append+".txt"
test_output_path = abs_file_path + "/data/input_data/gcn_test_dataset"+ab_append+yeast_append+mix_append+".txt"
train_data_set = load_corpus(train_output_path)
test_data_set = load_corpus(test_output_path)

#划分训练集，验证集和测试集
# train_data = data_set[:int(train_rate*len(data_set))]
# val_data = data_set[int(train_rate*len(data_set)):int((train_rate+val_rate)*len(data_set))]
# test_data = data_set[int((train_rate+val_rate)*len(data_set)):]

# train_data = train_data_set[:1166] + train_data_set[1167:] #1159 - 1168
train_data = train_data_set
val_data = test_data_set[:int(len(test_data_set)/2)]
test_data = test_data_set[int(len(test_data_set)/2):]
print("数据集划分：", len(train_data), len(val_data), len(test_data))


encoder = PackLSTM(vocab_size, embed_size, hidden_dim, bidirectional=True).to(device)
model = GCN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
mlp1 = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=input_dim).to(device)
mlp2 = MLP(input_dim=hidden_dim*2, hidden_dim=hidden_dim, output_dim=input_dim).to(device)
mlp3 = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=input_dim).to(device)
# mlp4 = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=input_dim).to(device)
decoder = AttnDecoderRNN(hidden_size=output_dim, output_size=vocab_size, max_length=max_length, with_attention=with_attention).to(device)

# base_epoch = 29
# model_load_path = abs_file_path + '/models/' + str(base_epoch)
model_load_path = abs_file_path + args.model
encoder = torch.load(model_load_path+'/encoder')
model = torch.load(model_load_path+'/gcn')
mlp1 = torch.load(model_load_path+'/mlp1')
mlp2 = torch.load(model_load_path+'/mlp2')
mlp3 = torch.load(model_load_path+'/mlp3')
decoder = torch.load(model_load_path+'/decoder')

save_path ='temp_result.txt'
with torch.no_grad(): #减少内存占用，pytorch不用存储计算图
    pred_text, ground_truth = evaluate(val_data)
    #输出整个测试集上的平均得分
    f = open(save_path, 'w', encoding='utf-8')
    f.write('此轮结果：\n')
    scores = rouge_score(pred_text, ground_truth, avg=True)
    print('rouge-1, rouge-2, rouge-L : {:.4f}, {:.4f}, {:.4f}'.format(scores['rouge-1']['f'],scores['rouge-2']['f'],scores['rouge-l']['f']))
    f.write('rouge-1, rouge-2, rouge-L : {:.4f}, {:.4f}, {:.4f}'.format(scores['rouge-1']['f'],scores['rouge-2']['f'],scores['rouge-l']['f'])+'\n')
    scores = bleu_score(pred_text, ground_truth, avg=True)
    print('bleu-1, bleu-2, bleu-3 : {:.4f}, {:.4f}, {:.4f}'.format(scores['bleu-1'],scores['bleu-2'],scores['bleu-3']))
    f.write('bleu-1, bleu-2, bleu-3 : {:.4f}, {:.4f}, {:.4f}'.format(scores['bleu-1'],scores['bleu-2'],scores['bleu-3'])+'\n')
    f.close()
    #获取每条结果的得分，并保存在文件中
    score_list = rouge_score(pred_text, ground_truth, avg=False)
    score_format_list = []

    for score in score_list:
        score_format = 'rouge-1, rouge-2, rouge-L : {:.4f}, {:.4f}, {:.4f}'.format(score['rouge-1']['f'],score['rouge-2']['f'],score['rouge-l']['f'])
        score_format_list.append(score_format)

    save_results(pred_text, ground_truth, score_format_list, save_path)
        




