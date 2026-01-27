"""
Informer 模型实现
论文: Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting
链接: https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132

核心创新:
1. ProbSparse 自注意力机制: 复杂度从 O(L^2) 降至 O(L log L)
   - 通过 KL 散度选择最重要的 Query，只计算关键的注意力
2. 自注意力蒸馏(Self-attention Distilling): 逐层减半序列长度
   - 使用卷积和最大池化压缩特征
3. 生成式解码器: 一次性生成整个预测序列
   - 避免自回归解码的累积误差

获奖: AAAI 2021 Best Paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding


class Model(nn.Module):
    """
    Informer 模型
    - 使用 ProbSparse 注意力实现 O(L log L) 复杂度
    - 适用于长序列时间序列预测
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len  # 解码器输入的已知标签长度

        # ==================== 嵌入层 ====================
        # 编码器嵌入: 值嵌入 + 位置编码 + 时间特征
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # 解码器嵌入
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        # ==================== 编码器 ====================
        self.encoder = Encoder(
            # 编码器层列表
            [
                EncoderLayer(
                    AttentionLayer(
                        # ProbSparse 注意力: Informer 的核心创新
                        # factor 参数控制采样的稀疏程度
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            # 蒸馏层: 使用卷积逐层压缩序列长度（仅在预测任务中使用）
            [
                ConvLayer(
                    configs.d_model
                ) for l in range(configs.e_layers - 1)
            ] if configs.distil and ('forecast' in configs.task_name) else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # ==================== 解码器 ====================
        self.decoder = Decoder(
            [
                DecoderLayer(
                    # 解码器自注意力（带掩码）
                    AttentionLayer(
                        ProbAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    # 编码器-解码器交叉注意力
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

        # ==================== 其他任务的输出层 ====================
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        长期预测

        Args:
            x_enc: 编码器输入 [B, seq_len, enc_in]
            x_mark_enc: 编码器时间戳 [B, seq_len, time_features]
            x_dec: 解码器输入 [B, label_len + pred_len, dec_in]
            x_mark_dec: 解码器时间戳 [B, label_len + pred_len, time_features]
        """
        # 嵌入
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)

        # 编码
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # 解码（生成式，一次性输出整个预测序列）
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        return dec_out  # [B, L, D]

    def short_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        短期预测（带实例归一化）

        与长期预测的区别:
        - 使用 RevIN (Reversible Instance Normalization) 处理分布偏移
        """
        # ===== 实例归一化 =====
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # [B, 1, E]
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        # 嵌入和编码
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # 解码
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        # ===== 反归一化 =====
        dec_out = dec_out * std_enc + mean_enc
        return dec_out  # [B, L, D]

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        """缺失值填充任务"""
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out)
        return dec_out

    def anomaly_detection(self, x_enc):
        """异常检测任务"""
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        """分类任务"""
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # 输出处理
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)  # 掩码填充位置
        output = output.reshape(output.shape[0], -1)  # [B, seq_len * d_model]
        output = self.projection(output)  # [B, num_classes]
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """统一前向传播接口"""
        if self.task_name == 'long_term_forecast':
            dec_out = self.long_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'short_term_forecast':
            dec_out = self.short_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
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
