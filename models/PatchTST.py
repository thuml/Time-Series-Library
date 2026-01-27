"""
PatchTST 模型实现
论文: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers
链接: https://arxiv.org/pdf/2211.14730.pdf

核心创新:
1. Patching (分块): 将时间序列分割成固定长度的子序列(patch)
   - 类似ViT对图像的处理方式
   - 减少序列长度，降低计算复杂度
   - 保留局部语义信息

2. Channel Independence (通道独立):
   - 每个变量独立处理，共享Transformer参数
   - 避免变量间的虚假相关性
   - 增强泛化能力

关键参数:
- patch_len: patch长度（默认16）
- stride: 滑动步长（默认8，有50%重叠）

优势:
- 显著降低内存消耗
- 更好的长程依赖建模
- 在多个基准上达到SOTA
"""

import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding


class Transpose(nn.Module):
    """转置辅助模块，用于在BatchNorm前后调整维度"""
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    """
    预测头: 将编码器输出展平并投影到目标长度

    输入: [B, nvars, d_model, patch_num]
    输出: [B, nvars, target_window]
    """
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)  # 展平最后两维
        self.linear = nn.Linear(nf, target_window)  # 投影到目标长度
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs, nvars, d_model, patch_num]
        x = self.flatten(x)  # [bs, nvars, d_model * patch_num]
        x = self.linear(x)   # [bs, nvars, target_window]
        x = self.dropout(x)
        return x


class Model(nn.Module):
    """
    PatchTST 模型
    - 基于Patch的时间序列Transformer
    - 通道独立设计，每个变量单独处理
    """

    def __init__(self, configs, patch_len=16, stride=8):
        """
        Args:
            configs: 配置对象
            patch_len: patch长度，将连续patch_len个时间步作为一个token
            stride: 滑动步长，控制patch之间的重叠程度
        """
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride  # 填充以确保序列能被完整分割

        # ==================== Patch嵌入层 ====================
        # 将时间序列分割成patches并嵌入
        # 输入: [B, N, L] -> 输出: [B*N, patch_num, d_model]
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        # ==================== Transformer编码器 ====================
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
            # 使用BatchNorm替代LayerNorm（经验上更稳定）
            norm_layer=nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2))
        )

        # ==================== 预测头 ====================
        # 计算patch数量: (seq_len - patch_len) / stride + 2（含padding）
        self.head_nf = configs.d_model * \
                       int((configs.seq_len - patch_len) / stride + 2)

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                self.head_nf * configs.enc_in, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        预测任务

        数据流:
        1. 实例归一化
        2. 转置: [B, L, N] -> [B, N, L]（通道独立处理）
        3. Patch嵌入: [B, N, L] -> [B*N, patch_num, d_model]
        4. Transformer编码
        5. 重塑: [B*N, patch_num, d_model] -> [B, N, d_model, patch_num]
        6. 预测头: [B, N, pred_len]
        7. 转置: [B, pred_len, N]
        8. 反归一化
        """
        # ===== 实例归一化 =====
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # ===== Patch嵌入（通道独立）=====
        x_enc = x_enc.permute(0, 2, 1)  # [B, L, N] -> [B, N, L]
        # enc_out: [B*N, patch_num, d_model], n_vars: N
        enc_out, n_vars = self.patch_embedding(x_enc)

        # ===== Transformer编码 =====
        enc_out, attns = self.encoder(enc_out)  # [B*N, patch_num, d_model]

        # ===== 重塑回多变量形式 =====
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))  # [B, N, patch_num, d_model]
        enc_out = enc_out.permute(0, 1, 3, 2)  # [B, N, d_model, patch_num]

        # ===== 预测头 =====
        dec_out = self.head(enc_out)  # [B, N, pred_len]
        dec_out = dec_out.permute(0, 2, 1)  # [B, pred_len, N]

        # ===== 反归一化 =====
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        """缺失值填充任务（考虑mask的归一化）"""
        # 归一化（仅使用非缺失值）
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # Patch嵌入和编码
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        # 预测头
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        # 反归一化
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        """异常检测任务"""
        # 归一化
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Patch嵌入和编码
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        # 预测头
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        # 反归一化
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        """分类任务"""
        # 归一化
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Patch嵌入和编码
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        # 分类输出
        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
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
