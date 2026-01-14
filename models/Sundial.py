import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
from transformers import GPT2Config, GPT2LMHeadModel


class Model(nn.Module):
    def __init__(self, configs):
        """
        Sundial is a flow-based time series generative model.
        Initialize with random weights using GPT2 architecture.
        """
        super().__init__()
        # Use GPT2 config sized similar to Sundial-128M
        config = GPT2Config(
            vocab_size=4096,
            n_positions=1024,
            n_embd=512,
            n_layer=12,
            n_head=8,
        )
        self.model = GPT2LMHeadModel(config)
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
            generated = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=self.pred_len,
                do_sample=False,
                pad_token_id=0,
            )
            pred_tokens = generated[:, -self.pred_len:]
            pred_values = (pred_tokens.float() / 100) - 3
            outputs.append(pred_values)

        dec_out = torch.stack(outputs, dim=-1)
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'zero_shot_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        return None
