import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.kan_feedforward import FeedForward

class MoeKANLayer(nn.Module):
    def __init__(self, hidden_size, d_ff, num_experts, n_experts_per_token):
        super(MoeKANLayer, self).__init__()
        self.num_experts = num_experts
        self.n_experts_per_token = n_experts_per_token
        self.experts = nn.ModuleList([FeedForward(hidden_size, d_ff) for _ in range(num_experts)])
        self.gate = nn.Linear(hidden_size, num_experts)

    def forward(self, x):
        # Gating mechanism: determine which part of the input should be processed by which expert
        gate_scores = self.gate(x)
        top_k_weights, top_k_indices = torch.topk(F.softmax(gate_scores, dim=-1), self.n_experts_per_token, dim=-1)

        # Combine the outputs from the selected experts
        output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            # Create a mask that selects the appropriate elements for this expert
            expert_mask = top_k_indices == i
            expert_contrib = expert(x * expert_mask.float()) * top_k_weights[expert_mask]
            output += expert_contrib

        return output

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        mean_square = torch.mean(x ** 2, dim=-1, keepdim=True)
        normalized_x = x / torch.sqrt(mean_square + self.eps)
        return self.scale * normalized_x

if __name__ == "__main__":
    # Example to test MoeKANLayer
    moe_layer = MoeKANLayer(hidden_size=512, d_ff=2048, num_experts=10, n_experts_per_token=2)
    dummy_input = torch.rand(1, 10, 512)  # (batch_size, seq_length, feature_size)
    print(moe_layer(dummy_input))