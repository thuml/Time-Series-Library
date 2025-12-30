import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding

try:
    from chronos import BaseChronosPipeline
except ImportError:
    print("Warning: chronos not installed. Chronos model may not work properly.")


class Model(nn.Module):
    def __init__(self, configs):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.model = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-base",
            device_map="cuda",  # use "cpu" for CPU inference and "mps" for Apple Silicon
            torch_dtype=torch.bfloat16,
        )
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        outputs = []
        for i in range(x_enc.shape[-1]):
            output = self.model.predict(x_enc[..., i], prediction_length=self.pred_len)
            output = output.mean(dim=1)
            outputs.append(output)
        dec_out = torch.stack(outputs, dim=-1)

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == "zero_shot_forecast":
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        return None
