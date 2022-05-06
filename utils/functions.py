import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import pickle as pkl
import scipy.sparse as sp
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from torch.autograd import Variable

# changes a numpy array to a Variable of LongTensor type
def numpy_to_var(x,is_int=True):
    if is_int:
        x = torch.LongTensor(x)
    else:
        x = torch.Tensor(x)
    return Variable(x)

def toData(batch):
    # [input] batch: list of strings
    # [output] input_out, output_out: np array([b x seq]), fixed size, eos & zero padding applied
    # [output] in_idx, out_idx: np.array([b]), length of each line in seq
    inputs_ = []
    in_len = []
    for line in batch:
        inputs_.append([int(num) for num in line.split()])
        in_len.append(len(inputs_[-1]))
    in_len = np.array(in_len)
    max_in = max(in_len)
    batch_size = len(batch)
    input_out = np.zeros([batch_size,max_in],dtype=int)
    for b in range(batch_size):
        input_out[b][:in_len[b]] = np.array(inputs_[b])
    #argsort为正序，[::-1]快速反序
    in_rev = in_len.argsort()[::-1]
    un_sort_idx = in_rev.argsort()
    return input_out[in_rev], in_len[in_rev], un_sort_idx

# def toData(batch):
#     # [input] batch: list of strings
#     # [output] input_out, output_out: np array([b x seq]), fixed size, eos & zero padding applied
#     # [output] in_idx, out_idx: np.array([b]), length of each line in seq
#     batch = [line.replace('\n','') for line in batch]
#     inputs_ = []
#     outputs_ = []
#     in_len = []
#     out_len = []
#     for line in batch:
#         # inputs, outputs, _ = line.split('\t')
#         inputs, outputs = line.split('\t')
#         inputs_.append([int(num) for num in inputs.split(',')])
#         outputs_.append([int(num) for num in outputs.split(',')])
#         in_len.append(len(inputs_[-1]))
#         out_len.append(len(outputs_[-1]))
#     in_len = np.array(in_len)
#     out_len = np.array(out_len)
#     max_in = max(in_len)
#     max_out = max(out_len)
#     batch_size = len(batch)
#     input_out = np.zeros([batch_size,max_in],dtype=int)
#     output_out = np.zeros([batch_size,max_out],dtype=int)
#     for b in range(batch_size):
#         input_out[b][:in_len[b]] = np.array(inputs_[b])
#         output_out[b][:out_len[b]] = np.array(outputs_[b])
#     #argsort为正序，[::-1]快速反序
#     out_rev = out_len.argsort()[::-1]
#     return input_out[out_rev], output_out[out_rev], in_len[out_rev], out_len[out_rev]

def to_np(x):
    return x.data.cpu().numpy()

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def visualize(x):
    plt.pcolor(x.cpu().data.numpy())

def decoder_initial(batch_size):
    decoder_in = torch.LongTensor(np.ones(batch_size,dtype=int))*2
    s = None
    w = None
    decoder_in = Variable(decoder_in)
    return decoder_in, s, w

def update_logger(logger,list_of_models,loss, step):
    # logger: ext. function defined by yunjey
    # list_of_models: [encoder, decoder]
    # step : current step
    info = {
        'loss': loss.data[0]
    }
    encoder, decoder = list_of_models
    for tag, value in info.items():
        logger.scalar_summary(tag,value,step)

    for tag, value in encoder.named_parameters():
        tag = 'encoder/'+tag
        logger.histo_summary(tag, to_np(value), step)
        logger.histo_summary(tag+'/grad', to_np(value.grad), step)

    for tag, value in decoder.named_parameters():
        tag = 'decoder/'+tag
        logger.histo_summary(tag, to_np(value), step)
        logger.histo_summary(tag+'/grad', to_np(value.grad), step)
    return logger

def load_corpus(data_path):
    results = []
    #加载数据
    with open(data_path, 'rb') as f:
            if sys.version_info > (3, 0):
                data = pkl.load(f, encoding='latin1')
            else:
                data = pkl.load(f)

    for termId, (vocab, adj) in data.items():
        #转化为对称矩阵
        # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        results.append((termId, vocab, adj))
    
    return results

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    # return sparse_to_tuple(adj_normalized)
    return adj_normalized.A

def rouge_score(pred_text, ground_truth, avg):
    """返回整个数据的平均rouge得分"""
    rouge = Rouge()
    scores = rouge.get_scores(pred_text, ground_truth, avg=avg)
    return scores

def bleu_score(pred_text, ground_truth, avg):
    results = []
    for text, truth in zip(pred_text, ground_truth):
        references = [truth.split()]
        hypothesis = text.split()
        result = {}
        result['bleu-1'] = sentence_bleu(references, hypothesis, (1.0,))
        result['bleu-2'] = sentence_bleu(references, hypothesis, (1./2.,1./2.))
        result['bleu-3'] = sentence_bleu(references, hypothesis, (1./3.,1./3.,1./3.))
        result['bleu-4'] = sentence_bleu(references, hypothesis, (1./4.,1./4.,1./4.,1./4.))
        results.append(result)

    if avg:
        sum_bleu1 = 0
        sum_bleu2 = 0
        sum_bleu3 = 0
        sum_bleu4 = 0
        count = 0
        for result in results:
            sum_bleu1 += result['bleu-1']
            sum_bleu2 += result['bleu-2']
            sum_bleu3 += result['bleu-3']
            sum_bleu4 += result['bleu-4']
            count += 1
        return{'bleu-1':sum_bleu1/count, 'bleu-2':sum_bleu2/count, 'bleu-3':sum_bleu3/count, 'bleu-4':sum_bleu4/count}
    
    else:
        return results

        

    