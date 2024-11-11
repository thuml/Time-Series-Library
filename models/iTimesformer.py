import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from layers.iTimesformer_Periodicity import PeriodicityReshape, PositionalEncoding


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.main_cycle = configs.main_cycle
        if configs.n_cycles == -1:
            self.n_cycles = configs.seq_len // self.main_cycle
        self.n_cycles = configs.n_cycles # Number of historic cycles 
        self.n_features = configs.c_out
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(self.main_cycle, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.positional_encoding = PositionalEncoding(configs.d_model)
        self.periodicity_reshape = PeriodicityReshape(self.main_cycle)
        # Encoder
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
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.seq_len//self.n_cycles, bias=True) # divide by n_cycles to match the shaping strategy by main_cycle
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.seq_len//self.n_cycles, bias=True) # divide by n_cycles to match the shaping strategy by main_cycle
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)

    def _apply_positional_encoding(self, enc_out):
        enc_out_parts = torch.chunk(enc_out, self.n_features, dim=1)  # Split along the second dimension
        # Apply positional encoding to each part
        encoded_parts = [self.positional_encoding(part) for part in enc_out_parts]
        # Combine the parts back together
        enc_out = torch.cat(encoded_parts, dim=1)
        return enc_out

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        # Reshape by periodicity
        x_enc = self.periodicity_reshape(x_enc, self.n_features, 'apply')
        x_mark_enc = self.periodicity_reshape(x_mark_enc, x_mark_enc.shape[-1], 'apply')

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self._apply_positional_encoding(enc_out)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :self.n_features]
        
        #* No need to restore the original shape for forecast because it 
        #* is already projected and subset to the correct shape
        
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        # Reshape by periodicity
        x_enc = self.periodicity_reshape(x_enc, self.n_features, 'apply')
        x_mark_enc = self.periodicity_reshape(x_mark_enc, x_mark_enc.shape[-1], 'apply')

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self._apply_positional_encoding(enc_out)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
 
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :self.n_features*self.n_cycles] # multiply by n_cycles to match the shaping strategy by main_cycle
        
        # Restore the original shape
        dec_out = self.periodicity_reshape(dec_out, self.n_features, 'revert')
        
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        # Reshape by periodicity
        x_enc = self.periodicity_reshape(x_enc, self.n_features, 'apply')
        x_mark_enc = self.periodicity_reshape(x_mark_enc, x_mark_enc.shape[-1], 'apply')

        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out = self._apply_positional_encoding(enc_out)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :self.n_features*self.n_cycles] # multiply by n_cycles to match the shaping strategy by main_cycle
        
        # Restore the original shape
        dec_out = self.periodicity_reshape(dec_out, self.n_features, 'revert')

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Reshape by periodicity
        x_enc = self.periodicity_reshape(x_enc, self.n_features, 'apply')
        x_mark_enc = self.periodicity_reshape(x_mark_enc, x_mark_enc.shape[-1], 'apply')
        
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out = self._apply_positional_encoding(enc_out)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, c_in * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None