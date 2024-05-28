import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, TemporalEmbedding
from torch import Tensor
from typing import Optional
from collections import namedtuple

# static: time-independent features
# observed: time features of the past(e.g. predicted targets)
# known: known information about the past and future(i.e. time stamp)
TypePos = namedtuple('TypePos', ['static', 'observed'])

# When you want to use new dataset, please add the index of 'static, observed' columns here.
# 'known' columns needn't be added, because 'known' inputs are automatically judged and provided by the program.
datatype_dict = {'ETTh1': TypePos([], [x for x in range(7)]),
                 'ETTm1': TypePos([], [x for x in range(7)])}


def get_known_len(embed_type, freq):
    if embed_type != 'timeF':
        if freq == 't':
            return 5
        else:
            return 4
    else:
        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        return freq_map[freq]


class TFTTemporalEmbedding(TemporalEmbedding):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TFTTemporalEmbedding, self).__init__(d_model, embed_type, freq)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        embedding_x = torch.stack([month_x, day_x, weekday_x, hour_x, minute_x], dim=-2) if hasattr(
            self, 'minute_embed') else torch.stack([month_x, day_x, weekday_x, hour_x], dim=-2)
        return embedding_x


class TFTTimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TFTTimeFeatureEmbedding, self).__init__()
        d_inp = get_known_len(embed_type, freq)
        self.embed = nn.ModuleList([nn.Linear(1, d_model, bias=False) for _ in range(d_inp)])

    def forward(self, x):
        return torch.stack([embed(x[:,:,i].unsqueeze(-1)) for i, embed in enumerate(self.embed)], dim=-2)


class TFTEmbedding(nn.Module):
    def __init__(self, configs):
        super(TFTEmbedding, self).__init__()
        self.pred_len = configs.pred_len
        self.static_pos = datatype_dict[configs.data].static
        self.observed_pos = datatype_dict[configs.data].observed
        self.static_len = len(self.static_pos)
        self.observed_len = len(self.observed_pos)

        self.static_embedding = nn.ModuleList([DataEmbedding(1,configs.d_model,dropout=configs.dropout) for _ in range(self.static_len)]) \
            if self.static_len else None
        self.observed_embedding = nn.ModuleList([DataEmbedding(1,configs.d_model,dropout=configs.dropout) for _ in range(self.observed_len)])
        self.known_embedding = TFTTemporalEmbedding(configs.d_model, configs.embed, configs.freq) \
            if configs.embed != 'timeF' else TFTTimeFeatureEmbedding(configs.d_model, configs.embed, configs.freq)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.static_len:
            # static_input: [B,C,d_model]
            static_input = torch.stack([embed(x_enc[:,:1,self.static_pos[i]].unsqueeze(-1), None).squeeze(1) for i, embed in enumerate(self.static_embedding)], dim=-2)
        else:
            static_input = None

        # observed_input: [B,T,C,d_model]
        observed_input = torch.stack([embed(x_enc[:,:,self.observed_pos[i]].unsqueeze(-1), None) for i, embed in enumerate(self.observed_embedding)], dim=-2)

        x_mark = torch.cat([x_mark_enc, x_mark_dec[:,-self.pred_len:,:]], dim=-2)
        # known_input: [B,T,C,d_model]
        known_input = self.known_embedding(x_mark)

        return static_input, observed_input, known_input


class GLU(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.fc2 = nn.Linear(input_size, output_size)
        self.glu = nn.GLU()

    def forward(self, x):
        a = self.fc1(x)
        b = self.fc2(x)
        return self.glu(torch.cat([a, b], dim=-1))


class GateAddNorm(nn.Module):
    def __init__(self, input_size, output_size):
        super(GateAddNorm, self).__init__()
        self.glu = GLU(input_size, input_size)
        self.projection = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()
        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, x, skip_a):
        x = self.glu(x)
        x = x + skip_a
        return self.layer_norm(self.projection(x))


class GRN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=None, context_size=None, dropout=0.0):
        super(GRN, self).__init__()
        hidden_size = input_size if hidden_size is None else hidden_size
        self.lin_a = nn.Linear(input_size, hidden_size)
        self.lin_c = nn.Linear(context_size, hidden_size) if context_size is not None else None
        self.lin_i = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.project_a = nn.Linear(input_size, hidden_size) if hidden_size != input_size else nn.Identity()
        self.gate = GateAddNorm(hidden_size, output_size)

    def forward(self, a: Tensor, c: Optional[Tensor] = None):
        # a: [B,T,d], c: [B,d]
        x = self.lin_a(a)
        if c is not None:
            x = x + self.lin_c(c).unsqueeze(1)
        x = F.elu(x)
        x = self.lin_i(x)
        x = self.dropout(x)
        return self.gate(x, self.project_a(a))


class VariableSelectionNetwork(nn.Module):
    def __init__(self, d_model, variable_num, dropout=0.0):
        super(VariableSelectionNetwork, self).__init__()
        self.joint_grn = GRN(d_model * variable_num, variable_num, hidden_size=d_model, context_size=d_model, dropout=dropout)
        self.variable_grns = nn.ModuleList([GRN(d_model, d_model, dropout=dropout) for _ in range(variable_num)])

    def forward(self, x: Tensor, context: Optional[Tensor] = None):
        # x: [B,T,C,d] or [B,C,d]
        # selection_weights: [B,T,C] or [B,C]
        # x_processed: [B,T,d,C] or [B,d,C]
        # selection_result: [B,T,d] or [B,d]
        x_flattened = torch.flatten(x, start_dim=-2)
        selection_weights = self.joint_grn(x_flattened, context)
        selection_weights = F.softmax(selection_weights, dim=-1)

        x_processed = torch.stack([grn(x[...,i,:]) for i, grn in enumerate(self.variable_grns)], dim=-1)

        selection_result = torch.matmul(x_processed, selection_weights.unsqueeze(-1)).squeeze(-1)
        return selection_result


class StaticCovariateEncoder(nn.Module):
    def __init__(self, d_model, static_len, dropout=0.0):
        super(StaticCovariateEncoder, self).__init__()
        self.static_vsn = VariableSelectionNetwork(d_model, static_len) if static_len else None
        self.grns = nn.ModuleList([GRN(d_model, d_model, dropout=dropout) for _ in range(4)])

    def forward(self, static_input):
        # static_input: [B,C,d]
        if static_input is not None:
            static_features = self.static_vsn(static_input)
            return [grn(static_features) for grn in self.grns]
        else:
            return [None] * 4


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, configs):
        super(InterpretableMultiHeadAttention, self).__init__()
        self.n_heads = configs.n_heads
        assert configs.d_model % configs.n_heads == 0
        self.d_head = configs.d_model // configs.n_heads
        self.qkv_linears = nn.Linear(configs.d_model, (2 * self.n_heads + 1) * self.d_head, bias=False)
        self.out_projection = nn.Linear(self.d_head, configs.d_model, bias=False)
        self.out_dropout = nn.Dropout(configs.dropout)
        self.scale = self.d_head ** -0.5
        example_len = configs.seq_len + configs.pred_len
        self.register_buffer("mask", torch.triu(torch.full((example_len, example_len), float('-inf')), 1))

    def forward(self, x):
        # Q,K,V are all from x
        B, T, d_model = x.shape
        qkv = self.qkv_linears(x)
        q, k, v = qkv.split((self.n_heads * self.d_head, self.n_heads * self.d_head, self.d_head), dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head)
        k = k.view(B, T, self.n_heads, self.d_head)
        v = v.view(B, T, self.d_head)

        attention_score = torch.matmul(q.permute((0, 2, 1, 3)), k.permute((0, 2, 3, 1)))  # [B,n,T,T]
        attention_score.mul_(self.scale)
        attention_score = attention_score + self.mask
        attention_prob = F.softmax(attention_score, dim=3)  # [B,n,T,T]

        attention_out = torch.matmul(attention_prob, v.unsqueeze(1))  # [B,n,T,d]
        attention_out = torch.mean(attention_out, dim=1)  # [B,T,d]
        out = self.out_projection(attention_out)
        out = self.out_dropout(out)  # [B,T,d]
        return out


class TemporalFusionDecoder(nn.Module):
    def __init__(self, configs):
        super(TemporalFusionDecoder, self).__init__()
        self.pred_len = configs.pred_len

        self.history_encoder = nn.LSTM(configs.d_model, configs.d_model, batch_first=True)
        self.future_encoder = nn.LSTM(configs.d_model, configs.d_model, batch_first=True)
        self.gate_after_lstm = GateAddNorm(configs.d_model, configs.d_model)
        self.enrichment_grn = GRN(configs.d_model, configs.d_model, context_size=configs.d_model, dropout=configs.dropout)
        self.attention = InterpretableMultiHeadAttention(configs)
        self.gate_after_attention = GateAddNorm(configs.d_model, configs.d_model)
        self.position_wise_grn = GRN(configs.d_model, configs.d_model, dropout=configs.dropout)
        self.gate_final = GateAddNorm(configs.d_model, configs.d_model)
        self.out_projection = nn.Linear(configs.d_model, configs.c_out)

    def forward(self, history_input, future_input, c_c, c_h, c_e):
        # history_input, future_input: [B,T,d]
        # c_c, c_h, c_e: [B,d]
        # LSTM
        c = (c_c.unsqueeze(0), c_h.unsqueeze(0)) if c_c is not None and c_h is not None else None
        historical_features, state = self.history_encoder(history_input, c)
        future_features, _ = self.future_encoder(future_input, state)

        # Skip connection
        temporal_input = torch.cat([history_input, future_input], dim=1)
        temporal_features = torch.cat([historical_features, future_features], dim=1)
        temporal_features = self.gate_after_lstm(temporal_features, temporal_input)  # [B,T,d]

        # Static enrichment
        enriched_features = self.enrichment_grn(temporal_features, c_e)  # [B,T,d]

        # Temporal self-attention
        attention_out = self.attention(enriched_features)  # [B,T,d]
        # Don't compute historical loss
        attention_out = self.gate_after_attention(attention_out[:,-self.pred_len:], enriched_features[:,-self.pred_len:])

        # Position-wise feed-forward
        out = self.position_wise_grn(attention_out)  # [B,T,d]

        # Final skip connection
        out = self.gate_final(out, temporal_features[:,-self.pred_len:])
        return self.out_projection(out)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        # Number of variables
        self.static_len = len(datatype_dict[configs.data].static)
        self.observed_len = len(datatype_dict[configs.data].observed)
        self.known_len = get_known_len(configs.embed, configs.freq)

        self.embedding = TFTEmbedding(configs)
        self.static_encoder = StaticCovariateEncoder(configs.d_model, self.static_len)
        self.history_vsn = VariableSelectionNetwork(configs.d_model, self.observed_len + self.known_len)
        self.future_vsn = VariableSelectionNetwork(configs.d_model, self.known_len)
        self.temporal_fusion_decoder = TemporalFusionDecoder(configs)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Data embedding
        # static_input: [B,C,d], observed_input:[B,T,C,d], known_input: [B,T,C,d]
        static_input, observed_input, known_input = self.embedding(x_enc, x_mark_enc, x_dec, x_mark_dec)

        # Static context
        # c_s,...,c_e: [B,d]
        c_s, c_c, c_h, c_e = self.static_encoder(static_input)

        # Temporal input Selection
        history_input = torch.cat([observed_input, known_input[:,:self.seq_len]], dim=-2)
        future_input = known_input[:,self.seq_len:]
        history_input = self.history_vsn(history_input, c_s)
        future_input = self.future_vsn(future_input, c_s)

        # TFT main procedure after variable selection
        # history_input: [B,T,d], future_input: [B,T,d]
        dec_out = self.temporal_fusion_decoder(history_input, future_input, c_c, c_h, c_e)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)  # [B,pred_len,C]
            dec_out = torch.cat([torch.zeros_like(x_enc), dec_out], dim=1)
            return dec_out  # [B, T, D]
        return None