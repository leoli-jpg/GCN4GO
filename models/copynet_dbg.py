import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models.model_utils import *
from models.mlp import MLP
import numpy as np
import time

grads = {}

def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook

class RNN(nn.Module):
    def __init__(self, hidden_size, bidirectional=True):
        super(RNN, self).__init__()

        self.gru = nn.GRU(input_size=hidden_size,
            hidden_size=hidden_size, batch_first=True,
            bidirectional=bidirectional)

    def forward(self, x, in_len):
        # input: [b x seq]

        x_pack = pack_padded_sequence(x, in_len, batch_first=True)
        out, h = self.gru(x_pack) # out: [b x seq x hid*2] (biRNN)
        out, _ = pad_packed_sequence(out, batch_first=True)
        return out, h

class PackRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, embedding=None, bidirectional=True):
        super(PackRNN, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.2)
        if embedding is not None:
            self.embed = embedding

        self.gru = nn.GRU(input_size=embed_size,
            hidden_size=hidden_size, batch_first=True,
            bidirectional=bidirectional)

    def forward(self, x, in_len):
        # print(x)
        # input: [b x seq]
        embedded = self.embed(x)
        embedded = self.dropout(embedded)

        x_pack = pack_padded_sequence(embedded, in_len, batch_first=True)
        # print(embedded)
        out, h = self.gru(x_pack) # out: [b x seq x hid*2] (biRNN)
        out, _ = pad_packed_sequence(out, batch_first=True)
        # out.register_hook(save_grad('x_pack'))
        return out, h

class PackLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, embedding=None, bidirectional=True):
        super(PackLSTM, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.)
        # self.mlp = MLP(input_dim=embed_size, hidden_dim=hidden_size, output_dim=hidden_size)
        if embedding is not None:
            self.embed = embedding

        self.gru = nn.LSTM(input_size=embed_size,
            hidden_size=hidden_size, batch_first=True,
            bidirectional=bidirectional)

    def forward(self, x, in_len):
        # print(x)
        # input: [b x seq]
        embedded = self.embed(x)
        # embedded = self.mlp(embedded)
        embedded = self.dropout(embedded)

        x_pack = pack_padded_sequence(embedded, in_len, batch_first=True)
        # print(embedded)
        out, (h,_) = self.gru(x_pack) # out: [b x seq x hid*2] (biRNN)
        out, _ = pad_packed_sequence(out, batch_first=True)
        # out.register_hook(save_grad('x_pack'))
        return out, h




class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, embedding=None, dropout_p=0.1, max_length=10, with_attention=True):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.mlp = MLP(hidden_size, hidden_size, hidden_size)
        self.with_attention = with_attention

        if embedding is not None:
            self.embedding = embedding
        
        else:
            self.embedding = nn.Embedding(self.output_size, self.hidden_size)

        self.dropout = nn.Dropout(self.dropout_p)
        if with_attention: self.gru = nn.GRU(self.hidden_size*2, self.hidden_size, batch_first=True)
        else: self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        # self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.Wo = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, prev_state, weighted, all_vocab, node_embeddings=None, batch_vocabs=None, encoder_hidden_list=None, seq_nums=None):
        embedded = self.embedding(input).unsqueeze(1)
        prev_state = prev_state.unsqueeze(0)
        


        if node_embeddings is not None:
            vocab_lengths = [node_embedding.shape[0] for node_embedding in node_embeddings]
            max_length = max(vocab_lengths)
            pad_embedding = torch.zeros(len(vocab_lengths), max_length, embedded.shape[-1], dtype=torch.float).to(embedded.device)
            mask = torch.zeros(len(vocab_lengths), max_length, dtype=torch.float).to(embedded.device)

            for i in range(len(node_embeddings)):
                node_embedding = node_embeddings[i]
                pad_embedding[i][:node_embedding.shape[0],:] = node_embedding
                mask[i][:node_embedding.shape[0]] = 1

            similarity_matrix = embedded.bmm(pad_embedding.transpose(2, 1).contiguous())/np.sqrt(self.hidden_size)

            atten_score = masked_softmax(similarity_matrix, mask)

            atten_embeded = weighted_sum(pad_embedding, atten_score)

            embedded = torch.cat((embedded, atten_embeded), -1)
            embedded = self.mlp(embedded)

        elif self.with_attention:
            max_length = max(seq_nums)
            pad_embedding = torch.zeros(len(seq_nums), max_length, embedded.shape[-1], dtype=torch.float).to(embedded.device)
            mask = torch.zeros(len(seq_nums), max_length, dtype=torch.float).to(embedded.device)

            for i in range(len(encoder_hidden_list)):
                node_embedding = encoder_hidden_list[i]
                pad_embedding[i][:node_embedding.shape[0],:] = node_embedding
                mask[i][:node_embedding.shape[0]] = 1

            # pad_embedding = self.mlp(pad_embedding)

            # atten_embeded, _ = self.soft_align_attentioin(prev_state.transpose(0,1), torch.LongTensor([1]*prev_state.shape[1]).to(prev_state.device), pad_embedding, torch.LongTensor(seq_nums).to(pad_embedding.device))
            

            similarity_matrix = prev_state.transpose(0,1).bmm(pad_embedding.transpose(2, 1).contiguous())

            atten_score = masked_softmax(similarity_matrix, mask)
            # print(atten_score[3])
            # exit(0)

            atten_embeded = weighted_sum(pad_embedding, atten_score)

            # embedded = embedded + atten_embeded
            embedded = torch.cat((embedded, atten_embeded), -1)
            # embedded = self.mlp(embedded)
 
        embedded = self.dropout(embedded)
        # print(prev_state.shape)
        output, state = self.gru(embedded, prev_state)
        state = state.squeeze(0)
        score_g = self.Wo(state) # [b x vocab_size]
        probs = torch.softmax(score_g, dim=-1)
        out = probs.unsqueeze(1)

        return out, state, weighted

    def soft_align_attentioin(self, x1, x1_lens, x2, x2_lens):
        '''
        :param x1: (batch, seq1_len, hidden_size)
        :param x1_len: (batch,)  每个句子1的长度
        :param x2: (batch, seq2_len, hidden_size)
        :param x2_len: (batch,)  每个句子2的长度
        :return: x1_align (batch, seq1_len, hidden_size), x2_align (batch, seq2_len, hidden_Size)
        '''

        # print(x1.shape)
        # print(x1_lens.shape)
        # print(x2.shape)
        # print(x2_lens.shape)
        # exit(0)
        seq1_len = x1.size(1) # (batch, )
        seq2_len = x2.size(1)  # (batch, )
        batch_size = x1.size(0)
        # (batch, seq1_len, seq2_len)
        attention = torch.matmul(x1, x2.transpose(1, 2))
        # (batch,  seq_len1)
        mask1 = torch.arange(seq1_len).expand(batch_size, seq1_len).to(x1.device) >= x1_lens.unsqueeze(1)
        # (batch, seq_len2)
        mask2 = torch.arange(seq2_len).expand(batch_size, seq2_len).to(x2.device) >= x2_lens.unsqueeze(1)
        mask1 = mask1.float().masked_fill_(mask1, float('inf'))
        mask2 = mask2.float().masked_fill_(mask2, float('inf'))
        # dim=-1: 对某一维度的行进行softmax运算
        weight1 = F.softmax(attention.transpose(1,2) + mask1.unsqueeze(1), dim = -1) # (batch, seq2_len, seq1_len)
        weight2 = F.softmax(attention + mask2.unsqueeze(1), dim=-1)  # (batch, seq1_len, seq2_len)
        x1_align = torch.matmul(weight2, x2)  # (batch, seq1_len, hidden_size)
        x2_align = torch.matmul(weight1, x1)  # (batch, seq2_len, hidden_size)
        print(x1_align.shape)
        print(x1_align)
        exit(0)

        return x1_align, x2_align
    
    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        return result

class AttnDecoderLSTM(nn.Module):
    def __init__(self, hidden_size, output_size, embedding=None, dropout_p=0.1, max_length=10):
        super(AttnDecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.mlp = MLP(hidden_size*2, hidden_size, hidden_size)

        if embedding is not None:
            self.embedding = embedding
        
        else:
            self.embedding = nn.Embedding(self.output_size, self.hidden_size)

        # self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        # self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        # self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.gru = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        self.Wo = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, prev_state, weighted, all_vocab, node_embeddings=None, batch_vocabs=None, encoder_hidden_list=None, seq_nums=None):
        embedded = self.embedding(input).unsqueeze(1)
        prev_state, prev_c = prev_state
        prev_state = prev_state.unsqueeze(0)
        


        if node_embeddings is not None:
            vocab_lengths = [node_embedding.shape[0] for node_embedding in node_embeddings]
            max_length = max(vocab_lengths)
            pad_embedding = torch.zeros(len(vocab_lengths), max_length, embedded.shape[-1], dtype=torch.float).to(embedded.device)
            mask = torch.zeros(len(vocab_lengths), max_length, dtype=torch.float).to(embedded.device)

            for i in range(len(node_embeddings)):
                node_embedding = node_embeddings[i]
                pad_embedding[i][:node_embedding.shape[0],:] = node_embedding
                mask[i][:node_embedding.shape[0]] = 1

            similarity_matrix = embedded.bmm(pad_embedding.transpose(2, 1).contiguous())

            atten_score = masked_softmax(similarity_matrix, mask)

            atten_embeded = weighted_sum(pad_embedding, atten_score)

            embedded = torch.cat((embedded, atten_embeded), -1)
            embedded = self.mlp(embedded)

        elif encoder_hidden_list is not None:
            max_length = max(seq_nums)
            pad_embedding = torch.zeros(len(seq_nums), max_length, embedded.shape[-1], dtype=torch.float).to(embedded.device)
            mask = torch.zeros(len(seq_nums), max_length, dtype=torch.float).to(embedded.device)

            for i in range(len(encoder_hidden_list)):
                node_embedding = encoder_hidden_list[i]
                pad_embedding[i][:node_embedding.shape[0],:] = node_embedding
                mask[i][:node_embedding.shape[0]] = 1

            similarity_matrix = hidden.bmm(pad_embedding.transpose(2, 1).contiguous())

            atten_score = masked_softmax(similarity_matrix, mask)

            atten_embeded = weighted_sum(pad_embedding, atten_score)

            embedded = torch.cat((embedded, atten_embeded), -1)
            embedded = self.mlp(embedded)
 
        embedded = self.dropout(embedded)
        # print(prev_state.shape)
        output, (state, c) = self.gru(embedded, (prev_state,prev_c))
        state = state.squeeze(0)
        score_g = self.Wo(state) # [b x vocab_size]
        probs = torch.softmax(score_g, dim=-1)
        out = probs.unsqueeze(1)

        return out, (state,c), weighted

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        return result

class CopyEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(CopyEncoder, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)

        self.gru = nn.GRU(input_size=embed_size,
            hidden_size=hidden_size, batch_first=True,
            bidirectional=True)

    def forward(self, x):
        # print(x)
        # input: [b x seq]
        embedded = self.embed(x)
        # print(embedded)
        out, h = self.gru(embedded) # out: [b x seq x hid*2] (biRNN)
        return out, h

class CopyDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, max_oovs=12, embedding=None):
        super(CopyDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.time = time.time()
        self.embed = nn.Embedding(vocab_size, embed_size)
        if embedding is not None:
            self.embed = embedding
        self.gru = nn.GRU(input_size=embed_size+hidden_size,
            hidden_size=hidden_size, batch_first=True)
        self.max_oovs = max_oovs # largest number of OOVs available per sample

        # weights
        self.Wo = nn.Linear(hidden_size, vocab_size) # generate mode
        self.Wc = nn.Linear(hidden_size, hidden_size) # copy mode
        self.nonlinear = nn.Tanh()
        self.mlp = MLP(hidden_size*2, hidden_size*2, hidden_size)

    def forward(self, input_idx, encoded, encoded_idx, prev_state, weighted, order):
        # input_idx(y_(t-1)): [b]			<- idx of next input to the decoder (Variable)
        # encoded: [b x seq x hidden]		<- hidden states created at encoder (Variable)
        # encoded_idx: [b x seq]			<- idx of inputs used at encoder (numpy)
        # prev_state(s_(t-1)): [b x hidden]		<- hidden states to be used at decoder (Variable)
        # weighted: [b x 1 x hidden]		<- weighted attention of previous state, init with all zeros (Variable)

        # hyperparameters
        start = time.time()
        time_check = False
        b = encoded.size(0) # batch size
        seq = encoded.size(1) # input sequence length
        vocab_size = self.vocab_size
        hidden_size = self.hidden_size

        # 0. set initial state s0 and initial attention (blank)
        if order==0:
            weighted = torch.Tensor(b,1,hidden_size).zero_()
            weighted = self.to_cuda(weighted)
            weighted = Variable(weighted)

        prev_state = prev_state.unsqueeze(0) # [1 x b x hidden]
        if time_check:
            self.elapsed_time('state 0')

        # 1. update states
        gru_input = torch.cat([self.embed(input_idx).unsqueeze(1), weighted],2) # [b x 1 x (h*2+emb)]
        # gru_input = self.mlp(gru_input)
        _, state = self.gru(gru_input, prev_state)
        state = state.squeeze(0) # [b x h]

        if time_check:
            self.elapsed_time('state 1')

        # 2. predict next word y_t
        # 2-1) get scores score_g for generation- mode
        score_g = self.Wo(state) # [b x vocab_size]

        if time_check:
            self.elapsed_time('state 2-1')

        # 2-2) get scores score_c for copy mode, remove possibility of giving attention to padded values
        score_c = torch.tanh(self.Wc(encoded.contiguous().view(-1,hidden_size))) # [b*seq x hidden_size]
        score_c = score_c.view(b,-1,hidden_size) # [b x seq x hidden_size]
        score_c = torch.bmm(score_c, state.unsqueeze(2)).squeeze() # [b x seq]

        score_c = torch.tanh(score_c) # purely optional....

        encoded_mask = torch.Tensor(np.array(encoded_idx==0, dtype=float)*(-1000)) # [b x seq]
        encoded_mask = self.to_cuda(encoded_mask)
        encoded_mask = Variable(encoded_mask)
        score_c = score_c + encoded_mask # padded parts will get close to 0 when applying softmax

        if time_check:
            self.elapsed_time('state 2-2')

        # 2-3) get softmax-ed probabilities
        score = torch.cat([score_g,score_c],1) # [b x (vocab+seq)]
        probs = torch.softmax(score, dim=-1)
        # print(probs)
        # exit(0)
        # probs.register_hook(save_grad('probs'))
        if(torch.sum(probs!=probs)>0):
            print('probs')
            print(probs)
            exit(0)
        prob_g = probs[:,:vocab_size] # [b x vocab]
        prob_c = probs[:,vocab_size:] # [b x seq]
        # prob_c.register_hook(save_grad('prob_c'))

        if time_check:
            self.elapsed_time('state 2-3')

        # # 2-4) add empty sizes to prob_g which correspond to the probability of obtaining OOV words
        # oovs = Variable(torch.Tensor(b,self.max_oovs).zero_())+1e-4
        # oovs = self.to_cuda(oovs)
        # prob_g = torch.cat([prob_g,oovs],1)

        # if time_check:
        #     self.elapsed_time('state 2-4')

        # 2-5) add prob_c to prob_g
        # prob_c_to_g = self.to_cuda(torch.Tensor(prob_g.size()).zero_())
        # prob_c_to_g = Variable(prob_c_to_g)
        # for b_idx in range(b): # for each sequence in batch
        # 	for s_idx in range(seq):
        # 		prob_c_to_g[b_idx,encoded_idx[b_idx,s_idx]]=prob_c_to_g[b_idx,encoded_idx[b_idx,s_idx]]+prob_c[b_idx,s_idx]


        # prob_c_to_g = Variable
        en = torch.LongTensor(encoded_idx) # [b x in_seq]
        en.unsqueeze_(2) # [b x in_seq x 1]
        #得到one_hot编码的经典代码
        one_hot = torch.FloatTensor(en.size(0),en.size(1),prob_g.size(1)).zero_() # [b x in_seq x vocab]
        one_hot.scatter_(2,en,1) # one hot tensor: [b x seq x vocab]
        # print(one_hot.shape)
        # print(one_hot[:,:,0].shape)
        # exit(0)
        # one_hot[:,:,0] = 0
        one_hot = self.to_cuda(one_hot)
        prob_c_to_g = torch.bmm(prob_c.unsqueeze(1),Variable(one_hot, requires_grad=False)) # [b x 1 x vocab]
        prob_c_to_g = prob_c_to_g.squeeze(1) # [b x vocab]
        # prob_c_to_g.register_hook(save_grad('prob_c_to_g'))

        # #prob_c_to_g = Variable
        # en = torch.LongTensor(encoded_idx) # [b x in_seq]
        # prob_c_to_g = []
        # for i in range(b):
        #     true_len = np.sum(np.array(encoded_idx[i]>0, dtype=int))
        #     col_index = en[i][:true_len]
        #     row_index = torch.LongTensor(range(0, true_len))
        #     value = torch.ones(col_index.shape)
        #     index = torch.cat((col_index.unsqueeze(0), row_index.unsqueeze(0)))
        #     one_hot = torch.sparse.FloatTensor(index, value, (vocab_size, col_index.shape[0])) # [vocab x seq]
        #     one_hot = self.to_cuda(one_hot)
        #     prob_c_to_g.append(torch.mm(Variable(one_hot, requires_grad=False), prob_c[i][:true_len].unsqueeze(1)).unsqueeze(0)) #[1 x vocab x 1]

        # prob_c_to_g = torch.cat(prob_c_to_g) # [b x vocab x 1]
        # prob_c_to_g = prob_c_to_g.squeeze(2) # [b x vocab]

        # if(torch.sum(prob_g!=prob_g)>0):
        #     print('prob_g')
        #     print(prob_g)
        #     exit(0)
        # if(torch.sum(prob_c_to_g!=prob_c_to_g)>0):
        #     print('prob_c_to_g')
        #     print(prob_c_to_g)
        #     exit(0)
        # prob_g.register_hook(save_grad('prob_g'))
        out = prob_g + prob_c_to_g
        out = out.unsqueeze(1) # [b x 1 x vocab]
        # out.register_hook(save_grad('out'))
        # if(torch.sum(out!=out)>0):
        #     print('out')
        #     print(out)
        #     exit(0)

        if time_check:
            self.elapsed_time('state 2-5')

        # 3. get weighted attention to use for predicting next word
        # 3-1) get tensor that shows whether each decoder input has previously appeared in the encoder
        idx_from_input = []
        
        for i,j in enumerate(encoded_idx):
            idx_from_input.append([int(k==input_idx[i].item()) for k in j])
        idx_from_input = torch.Tensor(np.array(idx_from_input, dtype=float))
        # idx_from_input : np.array of [b x seq]
        idx_from_input = self.to_cuda(idx_from_input)
        idx_from_input = Variable(idx_from_input)
        for i in range(b):
            if idx_from_input[i].sum().item()>1:
                idx_from_input[i] = idx_from_input[i]/idx_from_input[i].sum().item()

        if time_check:
            self.elapsed_time('state 3-1')

        # 3-2) multiply with prob_c to get final weighted representation
        attn = prob_c * idx_from_input
        # for i in range(b):
        # 	tmp_sum = attn[i].sum()
        # 	if (tmp_sum.data[0]>1e-6):
        # 		attn[i] = attn[i] / tmp_sum.data[0]
        attn = attn.unsqueeze(1) # [b x 1 x seq]
        weighted = torch.bmm(attn, encoded) # weighted: [b x 1 x hidden*2]

        if time_check:
            self.elapsed_time('state 3-2')


        return out, state, weighted

    def to_cuda(self, tensor):
        # turns to cuda
        if torch.cuda.is_available():
            return tensor.cuda()
        else:
            return tensor

    def elapsed_time(self, state):
        elapsed = time.time()
        print("Time difference from %s: %1.4f"%(state,elapsed-self.time))
        self.time = elapsed
        return
