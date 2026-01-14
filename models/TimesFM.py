import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
from transformers import TimesFmConfig, TimesFmModelForPrediction


class Model(nn.Module):
    def __init__(self, configs):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()

        # Load config and create model with random weights
        config = TimesFmConfig.from_pretrained("google/timesfm-2.0-500m-pytorch")
        self.model = TimesFmModelForPrediction(config)

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        B, L, C = x_enc.shape
        device = x_enc.device

        # Process each channel separately
        outputs = []
        for i in range(C):
            channel_input = x_enc[:, :, i].unsqueeze(-1)  # [B, L, 1]
            output = self.model(
                past_values=channel_input,
                future_values=None,
                freq=[0] * B,  # frequency indicator
            )
            # Get mean prediction
            pred = output.prediction_outputs[:, :self.pred_len, 0]  # [B, pred_len]
            outputs.append(pred)

        dec_out = torch.stack(outputs, dim=-1)  # [B, pred_len, C]
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'zero_shot_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        return None
