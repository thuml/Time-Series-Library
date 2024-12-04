import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, CyclicEncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from layers.iTimesformer_Periodicity import PeriodicityReshape, PositionalEncoding
from .iTimesformer import Model as iTimesformerModel


class Model(iTimesformerModel):

    def __init__(self, configs):
        super(Model, self).__init__(configs)
        self.main_cycle = configs.main_cycle
        self.n_cycles = configs.seq_len // self.main_cycle
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
                CyclicEncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_temp, configs.n_heads),
                    configs.d_model,
                    configs.d_temp,
                    self.n_cycles,
                    configs.c_out,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    full_mlp=configs.full_mlp
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(configs.d_model*self.n_cycles, configs.pred_len, bias=True)
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.seq_len//self.n_cycles, bias=True) # divide by n_cycles to match the shaping strategy by main_cycle
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.seq_len//self.n_cycles, bias=True) # divide by n_cycles to match the shaping strategy by main_cycle
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)

