import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
from transformers import T5Config, T5ForConditionalGeneration


class Model(nn.Module):
    def __init__(self, configs):
        """
        Chronos-Bolt is based on T5 architecture.
        Initialize with random weights using T5Config.
        """
        super().__init__()
        # Load config from Chronos-Bolt and create T5 model with random weights
        config = T5Config.from_pretrained("amazon/chronos-bolt-base")
        self.model = T5ForConditionalGeneration(config)
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B, L, C = x_enc.shape
        device = x_enc.device

        # Normalize input
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # Process each channel and generate predictions
        outputs = []
        for i in range(C):
            # Prepare input for T5 (quantize to token ids)
            channel_data = x_enc[:, :, i]  # [B, L]
            # Simple quantization to vocab range
            input_ids = ((channel_data + 3) * 100).long().clamp(0, 4095)  # [B, L]

            # Generate future tokens
            generated = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=self.pred_len,
                do_sample=False,
            )
            # Dequantize back to values
            pred_tokens = generated[:, -self.pred_len:]  # [B, pred_len]
            pred_values = (pred_tokens.float() / 100) - 3
            outputs.append(pred_values)

        dec_out = torch.stack(outputs, dim=-1)  # [B, pred_len, C]

        # Denormalize
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'zero_shot_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        return None
