import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from einops import rearrange


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2406.16964
    """
    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_size = patch_len 
        self.stride = stride
        
        self.d_model = configs.d_model
       
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 2
        self.padding_patch_layer = nn.ReplicationPad1d((0,  self.stride)) 
        self.in_layer = nn.Linear(self.patch_size, self.d_model)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(1)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )
        self.out_layer = nn.Linear(self.d_model * self.patch_num, configs.pred_len)
            
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        B, _, C = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1)
        x_enc = self.padding_patch_layer(x_enc)
        x_enc = x_enc.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        enc_out = self.in_layer(x_enc)
        enc_out =  rearrange(enc_out, 'b c m l -> (b c) m l')
        dec_out, _ = self.encoder(enc_out)
        dec_out =  rearrange(dec_out, '(b c) m l -> b c (m l)' , b=B , c=C)
        dec_out = self.out_layer(dec_out)
        dec_out = dec_out.permute(0, 2, 1)
        
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out