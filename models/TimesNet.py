"""
TimesNet 模型实现
论文: TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis
链接: https://openreview.net/pdf?id=ju_Uqw384Oq

核心创新 - 时序2D变化建模:
1. 将1D时间序列转换为2D张量
   - 通过FFT发现时间序列中的主要周期
   - 按周期将1D序列折叠成2D张量
   - 例如：周期为24的序列 [1, 168] -> [7, 24]

2. 使用2D卷积（Inception块）捕获:
   - 周期内变化（Intraperiod）: 一个周期内的模式
   - 周期间变化（Interperiod）: 不同周期之间的趋势

3. 自适应聚合:
   - 同时考虑多个周期（top_k个）
   - 根据FFT幅度加权聚合不同周期的结果

优势:
- 统一框架：支持预测、填充、异常检测、分类5大任务
- 参数高效：2D卷积比Transformer更紧凑
- 在多个任务上达到SOTA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    """
    使用FFT发现时间序列中的主要周期

    Args:
        x: 输入序列 [B, T, C]
        k: 返回的周期数量

    Returns:
        period: 周期列表，长度为k
        period_weight: 每个周期的FFT幅度权重 [B, k]
    """
    # 对时间维度做FFT
    xf = torch.fft.rfft(x, dim=1)  # [B, T//2+1, C]

    # 计算幅度谱（平均所有batch和channel）
    frequency_list = abs(xf).mean(0).mean(-1)  # [T//2+1]
    frequency_list[0] = 0  # 去除直流分量

    # 选择幅度最大的k个频率
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()

    # 将频率转换为周期: period = T / frequency
    period = x.shape[1] // top_list

    # 返回周期和对应的权重
    return period, abs(xf).mean(-1)[:, top_list]  # [B, k]


class TimesBlock(nn.Module):
    """
    TimesNet的核心模块 - 时序2D变化建模块

    将1D时间序列按周期折叠成2D张量，用2D卷积提取特征
    """
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k  # 考虑的周期数量

        # ===== 2D卷积模块 =====
        # 使用Inception块：多尺度卷积并行
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        """
        Args:
            x: [B, T, N] 其中 T = seq_len + pred_len

        Returns:
            输出: [B, T, N]
        """
        B, T, N = x.size()

        # Step 1: 发现主要周期
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]

            # Step 2: 填充使得序列长度能被周期整除
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x

            # Step 3: 1D -> 2D 重塑
            # [B, T, N] -> [B, length//period, period, N] -> [B, N, length//period, period]
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()

            # Step 4: 2D卷积捕获周期内和周期间的变化
            out = self.conv(out)

            # Step 5: 2D -> 1D 重塑回来
            # [B, N, length//period, period] -> [B, T, N]
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])

        # Step 6: 多周期自适应聚合
        res = torch.stack(res, dim=-1)  # [B, T, N, k]
        # 使用FFT幅度作为权重
        period_weight = F.softmax(period_weight, dim=1)  # [B, k]
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)  # [B, T, N, k]
        res = torch.sum(res * period_weight, -1)  # [B, T, N]

        # Step 7: 残差连接
        res = res + x
        return res


class Model(nn.Module):
    """
    TimesNet 模型
    - 通用时间序列分析模型
    - 支持5大任务：长期预测、短期预测、填充、异常检测、分类
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        # ==================== TimesBlock堆叠 ====================
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])

        # ==================== 嵌入层 ====================
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)

        # ==================== 预测头（根据任务不同）====================
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # 预测任务：先扩展序列长度，再投影到输出维度
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        预测任务

        数据流:
        1. 实例归一化
        2. 嵌入: [B, L, C] -> [B, L, d_model]
        3. 序列扩展: [B, L, d_model] -> [B, L+pred_len, d_model]
        4. 多层TimesBlock: 2D变化建模
        5. 投影: [B, L+pred_len, d_model] -> [B, L+pred_len, c_out]
        6. 反归一化
        """
        # ===== 实例归一化 =====
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        # ===== 嵌入 =====
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, T, d_model]

        # ===== 序列扩展（对齐预测长度）=====
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # [B, T+pred_len, d_model]

        # ===== TimesBlock堆叠 =====
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # ===== 投影到输出维度 =====
        dec_out = self.projection(enc_out)

        # ===== 反归一化 =====
        dec_out = dec_out.mul(
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        dec_out = dec_out.add(
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        """缺失值填充任务（使用mask感知的归一化）"""
        # 归一化（仅使用非缺失值）
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc.sub(means)
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc = x_enc.div(stdev)

        # 嵌入和TimesBlock
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        dec_out = self.projection(enc_out)

        # 反归一化
        dec_out = dec_out.mul(
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        dec_out = dec_out.add(
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        return dec_out

    def anomaly_detection(self, x_enc):
        """异常检测任务"""
        # 归一化
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        # 嵌入和TimesBlock
        enc_out = self.enc_embedding(x_enc, None)
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        dec_out = self.projection(enc_out)

        # 反归一化
        dec_out = dec_out.mul(
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        dec_out = dec_out.add(
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        """分类任务"""
        # 嵌入和TimesBlock
        enc_out = self.enc_embedding(x_enc, None)
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # 输出处理
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)  # 掩码填充位置
        output = output.reshape(output.shape[0], -1)  # [B, seq_len * d_model]
        output = self.projection(output)  # [B, num_classes]
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """统一前向传播接口"""
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
