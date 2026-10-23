"""Official TimeXer inverted embedding: one token per historical variate."""
import torch
from torch import nn


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        if x_mark is not None:
            x = torch.cat([x, x_mark.permute(0, 2, 1)], dim=1)
        return self.dropout(self.value_embedding(x))
