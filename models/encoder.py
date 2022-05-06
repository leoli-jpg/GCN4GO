import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models.model_utils import *
from models.mlp import MLP
import numpy as np
import time
from seed import *

grads = {}

def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook

class PackRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, embedding=None, bidirectional=True):
        super(PackRNN, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.2)
        self.mlp = MLP(input_dim=embed_size, hidden_dim=hidden_size, output_dim=hidden_size)
        if embedding is not None:
            self.embed = embedding

        self.gru = nn.GRU(input_size=embed_size,
            hidden_size=hidden_size, batch_first=True,
            bidirectional=bidirectional)

    def forward(self, x, in_len):
        # print(x)
        # input: [b x seq]
        embedded = self.embed(x)
        embedded = self.mlp(embedded)
        embedded = self.dropout(embedded)

        x_pack = pack_padded_sequence(embedded, in_len, batch_first=True)
        # print(embedded)
        out, h = self.gru(x_pack) # out: [b x seq x hid*2] (biRNN)
        out, _ = pad_packed_sequence(out, batch_first=True)
        # out.register_hook(save_grad('x_pack'))
        return out, h

class PackRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, embedding=None, bidirectional=True):
        super(PackRNN, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.2)
        self.mlp = MLP(input_dim=embed_size, hidden_dim=hidden_size, output_dim=hidden_size)
        if embedding is not None:
            self.embed = embedding

        self.gru = nn.GRU(input_size=embed_size,
            hidden_size=hidden_size, batch_first=True,
            bidirectional=bidirectional)

    def forward(self, x, in_len):
        # print(x)
        # input: [b x seq]
        embedded = self.embed(x)
        embedded = self.mlp(embedded)
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
        self.dropout = nn.Dropout(0.2)
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

