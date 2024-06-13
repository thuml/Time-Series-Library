
import torch
import torch.nn as nn
from layers.MultiheadKANAttention import MultiheadKANAttention
from layers.moe_kan_layer import MoeKANLayer, RMSNorm
from layers.kan_feedforward import FeedForward


class KANBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, window_size, d_ff, num_experts, n_experts_per_token, rotation_matrix):
        super(KANBlock, self).__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.attention = MultiheadKANAttention(hidden_size, num_heads, window_size, num_experts, n_experts_per_token, rotation_matrix)
        self.moe = MoeKANLayer(hidden_size, d_ff, num_experts, n_experts_per_token)

    def forward(self, x):
        x1 = self.attention(self.norm1(x))
        x += x1
        x2 = self.moe(self.norm2(x))
        return x + x2

class KANFormer(nn.Module):
    def __init__(self, config):
        super(KANFormer, self).__init__()
        self.embedding = nn.Embedding(config['vocabulary_size'], config['hidden_size'])
        self.blocks = nn.ModuleList([
            KANBlock(config) for _ in range(config['num_layers'])
        ])
        self.out = nn.Linear(config['hidden_size'], config['vocabulary_size'])

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        return self.out(x)

if __name__ == "__main__":
    # Example configuration
    config = {
        'vocabulary_size': 10000,
        'hidden_size': 512,
        'num_layers': 6,
        'num_heads': 8,
        'dropout': 0.1,
        'max_length': 512,
        'num_experts': 10,
        'n_experts_per_token': 2,
        'd_ff': 2048
    }
    model = KANFormer(config)
    print(model)
