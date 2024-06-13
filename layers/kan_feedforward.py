import torch
import torch.nn as nn
from layers.efficient_kan import KANLinear

class FeedForward(nn.Module):
    def __init__(self, hidden_size, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = KANLinear(hidden_size, d_ff)
        self.linear2 = KANLinear(d_ff, hidden_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        return self.linear2(x)