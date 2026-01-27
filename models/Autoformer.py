"""
Autoformer 模型实现
论文: Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting
链接: https://openreview.net/pdf?id=I55UqU-M11y

核心创新:
1. 序列分解(Series Decomposition): 将时间序列分解为趋势项和季节项
2. 自相关机制(Auto-Correlation): 替代传统自注意力，利用序列的周期性进行信息聚合
3. 复杂度为 O(L log L)，比标准Transformer更高效
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import math
import numpy as np


class Model(nn.Module):
    """
    Autoformer 模型
    - 首个实现序列级连接的方法
    - 内在复杂度为 O(L log L)

    支持任务:
    - long_term_forecast: 长期预测
    - short_term_forecast: 短期预测
    - imputation: 缺失值填充
    - anomaly_detection: 异常检测
    - classification: 分类
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len          # 输入序列长度
        self.label_len = configs.label_len      # 标签长度（解码器输入的已知部分）
        self.pred_len = configs.pred_len        # 预测长度

        # ==================== 序列分解模块 ====================
        # 使用移动平均进行趋势-季节分解
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # ==================== 嵌入层 ====================
        # 使用不带位置编码的嵌入（因为自相关机制本身具有位置感知能力）
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)

        # ==================== 编码器 ====================
        self.encoder = Encoder(
            [
                EncoderLayer(
                    # 自相关层替代传统自注意力
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,  # 分解模块的移动平均窗口
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )

        # ==================== 解码器（仅预测任务需要）====================
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        # 解码器自注意力（使用掩码）
                        AutoCorrelationLayer(
                            AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False),
                            configs.d_model, configs.n_heads),
                        # 编码器-解码器交叉注意力
                        AutoCorrelationLayer(
                            AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.c_out,
                        configs.d_ff,
                        moving_avg=configs.moving_avg,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=my_Layernorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
            )

        # ==================== 其他任务的输出层 ====================
        if self.task_name == 'imputation':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        预测任务的前向传播

        Args:
            x_enc: 编码器输入 [B, seq_len, enc_in]
            x_mark_enc: 编码器时间戳特征 [B, seq_len, time_features]
            x_dec: 解码器输入 [B, label_len + pred_len, dec_in]
            x_mark_dec: 解码器时间戳特征 [B, label_len + pred_len, time_features]

        Returns:
            预测结果 [B, pred_len, c_out]
        """
        # ===== 分解初始化 =====
        # 用输入序列的均值初始化预测部分的趋势项
        mean = torch.mean(x_enc, dim=1).unsqueeze(
            1).repeat(1, self.pred_len, 1)
        # 用零初始化预测部分的季节项
        zeros = torch.zeros([x_dec.shape[0], self.pred_len,
                             x_dec.shape[2]], device=x_enc.device)
        # 对输入进行趋势-季节分解
        seasonal_init, trend_init = self.decomp(x_enc)

        # ===== 构造解码器输入 =====
        # 趋势项: 已知部分的趋势 + 均值填充的预测部分
        trend_init = torch.cat(
            [trend_init[:, -self.label_len:, :], mean], dim=1)
        # 季节项: 已知部分的季节 + 零填充的预测部分
        seasonal_init = torch.cat(
            [seasonal_init[:, -self.label_len:, :], zeros], dim=1)

        # ===== 编码器 =====
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # ===== 解码器 =====
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None,
                                                 trend=trend_init)

        # ===== 最终输出 = 趋势项 + 季节项 =====
        dec_out = trend_part + seasonal_part
        return dec_out

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
        # 使用掩码将填充位置置零
        output = output * x_mark_enc.unsqueeze(-1)
        # 展平为 [batch_size, seq_length * d_model]
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # [batch_size, num_classes]
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        统一前向传播接口

        根据 task_name 自动路由到对应的任务方法
        """
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
