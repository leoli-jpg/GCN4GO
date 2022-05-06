#!/usr/bin/env python
import torch
import torch.nn as nn
# from seed import *



class GraphConvolution(nn.Module):
    def __init__( self, input_dim, \
                        output_dim, \
                        support_num, \
                        act_func = None, \
                        featureless = False, \
                        dropout_rate = 0., \
                        bias=False):
        super(GraphConvolution, self).__init__()
        self.support_num = support_num
        self.featureless = featureless

        for i in range(self.support_num):
            setattr(self, 'W{}'.format(i), nn.Parameter(torch.randn(input_dim, output_dim)))

        if bias:
            self.b = nn.Parameter(torch.zeros(1, output_dim))

        self.act_func = act_func
        self.bn = nn.BatchNorm1d(output_dim)

        
    def forward(self, x, support):
        

        for i in range(self.support_num):
            if self.featureless:
                pre_sup = getattr(self, 'W{}'.format(i))
            else:
                pre_sup = x.mm(getattr(self, 'W{}'.format(i)))
            
            if i == 0:
                out = support[i].mm(pre_sup)
            else:
                out += support[i].mm(pre_sup)

        if self.act_func is not None:
            # out = self.bn(out)
            out = self.act_func(out)

        self.embedding = out
        return out


class GCN(nn.Module):
    def __init__( self, input_dim, \
                        hidden_dim,\
                        output_dim=10, \
                        support_num=1, \
                        dropout_rate=0.3):
        super(GCN, self).__init__()
        
        # GraphConvolution
        self.layer1 = GraphConvolution(input_dim, hidden_dim, support_num=support_num, act_func=nn.ReLU(), dropout_rate=dropout_rate)
        self.layer2 = GraphConvolution(hidden_dim, output_dim, support_num=support_num, dropout_rate=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        
    
    def forward(self, x, support):
        x = self.dropout(x)
        out = self.layer1(x, support)
        out = self.layer2(out, support)
        # out = self.dropout(out)
        return out
