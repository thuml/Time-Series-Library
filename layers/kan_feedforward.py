import torch
import torch.nn as nn
from layers.efficient_kan import KANLinear

class FeedForward(nn.Module):
    def __init__(self, hidden_size, d_ff, activation='gelu'):
        super(FeedForward, self).__init__()
        self.linear1 = KANLinear(hidden_size, d_ff)
        self.linear2 = KANLinear(d_ff, hidden_size)
        if activation == 'gelu':
            self.act = torch.nn.GELU()
        elif activation == 'relu':
            self.act = torch.nn.ReLU()
        else:
            self.act = torch.nn.Identity()
    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        return self.linear2(x)