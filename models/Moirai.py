import numpy as np
import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
from transformers import AutoConfig, AutoModel


class Model(nn.Module):
    def __init__(self, configs):
        """
        Moirai-2 is a decoder-only transformer for time series.
        Initialize with random weights using a compatible config.
        """
        super().__init__()
        # Use a standard transformer config for random initialization
        config = AutoConfig.from_pretrained("Salesforce/moirai-2.0-R-small", trust_remote_code=True)
        self.model = AutoModel.from_config(config, trust_remote_code=True)
        self.pred_head = nn.Linear(config.hidden_size if hasattr(config, 'hidden_size') else 512, 1)

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B, L, C = x_enc.shape
        device = x_enc.device

        # Normalize
        means = x_enc.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = (x_enc - means) / stdev

        outputs = []
        for i in range(C):
            channel_data = x_enc[:, :, i]
            input_ids = ((channel_data + 3) * 100).long().clamp(0, 4095)
            hidden = self.model(input_ids=input_ids).last_hidden_state
            pred = self.pred_head(hidden[:, -self.pred_len:, :]).squeeze(-1)
            outputs.append(pred)

        dec_out = torch.stack(outputs, dim=-1)
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'zero_shot_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        return None
