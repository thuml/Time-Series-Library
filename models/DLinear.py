"""
DLinear 模型实现
论文: Are Transformers Effective for Time Series Forecasting?
链接: https://arxiv.org/pdf/2205.13504.pdf

核心思想:
1. 质疑Transformer在时序预测��的有效性
2. 提出极简的线性模型，仅使用一层线性层
3. 结合序列分解，将输入分解为趋势和季节两部分分别预测

模型特点:
- 参数量极少，训练速度快
- 在多个基准上表现优异
- 可选择 individual 模式：为每个变量单独训练线性层
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp


class Model(nn.Module):
    """
    DLinear 模型
    - 使用分解的线性模型进行时序预测
    - 将序列分解为趋势和季节，分别用线性层预测

    Args:
        configs: 配置对象
        individual: 是否为每个变量单独使用线性层（默认False共享参数）
    """

    def __init__(self, configs, individual=False):
        """
        初始化 DLinear 模型

        参数:
            individual: Bool, 是否在不同变量间共享模型参数
                - False: 所有变量共享一个线性层（参数高效）
                - True: 每个变量有独立的线性层（容量更大）
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len

        # 对于非预测任务，输出长度等于输入长度
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len

        # ==================== 序列分解模块 ====================
        # 使用移动平均进行趋势-季节分解（来自Autoformer）
        self.decompsition = series_decomp(configs.moving_avg)
        self.individual = individual
        self.channels = configs.enc_in  # 输入变量数量

        # ==================== 线性层定义 ====================
        if self.individual:
            # 每个变量单独的线性层
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                # 季节项线性层: seq_len -> pred_len
                self.Linear_Seasonal.append(
                    nn.Linear(self.seq_len, self.pred_len))
                # 趋势项线性层: seq_len -> pred_len
                self.Linear_Trend.append(
                    nn.Linear(self.seq_len, self.pred_len))

                # 初始化权重为均匀分布（类似移动平均）
                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        else:
            # 所有变量共享的线性层
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            # 初始化权重
            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        # ==================== 分类任务的输出层 ====================
        if self.task_name == 'classification':
            self.projection = nn.Linear(
                configs.enc_in * configs.seq_len, configs.num_class)

    def encoder(self, x):
        """
        编码器：分解 + 线性预测

        Args:
            x: 输入序列 [B, seq_len, channels]

        Returns:
            预测序列 [B, pred_len, channels]
        """
        # Step 1: 序列分解
        seasonal_init, trend_init = self.decompsition(x)

        # Step 2: 转置以在时间维度上应用线性层 [B, C, L]
        seasonal_init, trend_init = seasonal_init.permute(
            0, 2, 1), trend_init.permute(0, 2, 1)

        # Step 3: 线性预测
        if self.individual:
            # 为每个变量单独预测
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](
                    trend_init[:, i, :])
        else:
            # 共享参数预测
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        # Step 4: 趋势 + 季节 = 最终预测
        x = seasonal_output + trend_output

        # Step 5: 转置回 [B, L, C]
        return x.permute(0, 2, 1)

    def forecast(self, x_enc):
        """预测任务"""
        return self.encoder(x_enc)

    def imputation(self, x_enc):
        """缺失值填充任务"""
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        """异常检测任务"""
        return self.encoder(x_enc)

    def classification(self, x_enc):
        """分类任务"""
        # 编码
        enc_out = self.encoder(x_enc)
        # 展平 [batch_size, seq_length * d_model]
        output = enc_out.reshape(enc_out.shape[0], -1)
        # 分类投影 [batch_size, num_classes]
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        统一前向传播接口

        注意: DLinear 不使用时间戳特征（x_mark_enc, x_mark_dec）
        """
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        return None
