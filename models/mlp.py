#!/usr/bin/env python
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5, is_relu=True):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        if is_relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = nn.Tanh()
        # self.relu = nn.Tanh()
        self.dropout = nn.Dropout(dropout_rate)
        self.bn = nn.BatchNorm1d(hidden_dim)
    
    def forward(self, x):
        out = self.fc1(x)
        # out = self.bn(out)
        out = self.relu(out)
        # out = self.dropout(out)
        # out = self.fc2(out)

        return out