"""
iTransformer 模型实现
论文: iTransformer: Inverted Transformers Are Effective for Time Series Forecasting
链接: https://arxiv.org/abs/2310.06625

核心创新 - 倒置Transformer架构:
传统Transformer: 在时间维度上做注意力（不同时间步之间）
iTransformer: 在变量维度上做注意力（不同变量之间）

关键设计:
1. 倒置嵌入(Inverted Embedding): 将整个时间序列嵌入为单个token
   - 输入 [B, L, N] -> 嵌入后 [B, N, D]，每个变量成为一个token
2. 变量间注意力: Attention在N个变量之间计算，捕获多变量相关性
3. 独立的时序建模: 每个变量的时序模式由MLP独立学习

优势:
- 更好地捕获多变量之间的相互依赖
- 避免时间步之间的注意力稀释问题
- 在长序列预测上表现优异
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np


class Model(nn.Module):
    """
    iTransformer 模型
    - 倒置的Transformer，在变量维度而非时间维度上应用注意力
    - 每个变量的整个时间序列作为一个token
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # ==================== 倒置嵌入层 ====================
        # 输入: [B, L, N] -> 输出: [B, N, D]
        # 将每个变量的整个时间序列映射到d_model维空间
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)

        # ==================== 编码器（在变量维度上做注意力）====================
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        # 标准全注意力（在N个变量之间计算）
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

        # ==================== 预测头（投影到目标长度）====================
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # d_model -> pred_len: 从嵌入空间映射到预测长度
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            # N个变量，每个d_model维
            self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        预测任务

        数据流:
        1. 输入 x_enc: [B, L, N]
        2. 实例归一化: 减均值除标准差
        3. 倒置嵌入: [B, L, N] -> [B, N, D]
        4. Transformer编码: 变量间注意力
        5. 投影: [B, N, D] -> [B, N, pred_len]
        6. 转置: [B, pred_len, N]
        7. 反归一化
        """
        # ===== 实例归一化（Non-stationary Transformer风格）=====
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape  # N = 变量数量

        # ===== 倒置嵌入 =====
        # [B, L, N] -> [B, N, D]
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # ===== 编码器（变量间注意力）=====
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # ===== 投影到预测长度 =====
        # [B, N, D] -> [B, N, pred_len] -> [B, pred_len, N]
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

        # ===== 反归一化 =====
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        """缺失值填充任务"""
        # 归一化
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # 嵌入和编码
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # 投影
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

        # 反归一化
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        """异常检测任务"""
        # 归一化
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # 嵌入和编码
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # 投影
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

        # 反归一化
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        """分类任务"""
        # 嵌入和编码
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # 输出处理
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # [B, N * d_model]
        output = self.projection(output)  # [B, num_classes]
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """统一前向传播接口"""
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
