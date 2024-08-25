import torch
from math import ceil
from torch import nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer

# forecast task head
class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super(Flatten_Head, self).__init__()

        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x d_model * patch_num]
                z = self.linears[i](z)  # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, large_size, small_size, dmodel, dff, nvars, small_kernel_merged=False, drop=0.1):
        super(Block, self).__init__()
        # self.dw = ReparamLargeKernelConv(in_channels=nvars * dmodel, out_channels=nvars * dmodel,
        #                                  kernel_size=large_size, stride=1, groups=nvars * dmodel,
        #                                  small_kernel=small_size, small_kernel_merged=small_kernel_merged, nvars=nvars)
        self.dw_large = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dmodel, kernel_size=large_size, stride=1,
                                 padding=large_size // 2, dilation=1, groups=nvars * dmodel)
        self.dw_small = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dmodel, kernel_size=small_size, stride=1,
                                    padding=small_size // 2, dilation=1, groups=nvars * dmodel)

        self.norm = nn.BatchNorm1d(dmodel)

        # #convffn1
        # self.ffn1pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1,
        #                          padding=0, dilation=1, groups=nvars)
        # self.ffn1act = nn.GELU()
        # self.ffn1pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
        #                          padding=0, dilation=1, groups=nvars)
        # self.ffn1drop1 = nn.Dropout(drop)
        # self.ffn1drop2 = nn.Dropout(drop)
        # 
        # #convffn2
        # self.ffn2pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1,
        #                          padding=0, dilation=1, groups=dmodel)
        # self.ffn2act = nn.GELU()
        # self.ffn2pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
        #                          padding=0, dilation=1, groups=dmodel)
        # self.ffn2drop1 = nn.Dropout(drop)
        # self.ffn2drop2 = nn.Dropout(drop)
        # 
        # self.ffn_ratio = dff//dmodel
        
        # Encoder
        self.encoder = EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, 3, attention_dropout=drop,
                                      output_attention=False), dmodel, 8),
                    dmodel,
                    dff,
                    dropout=drop,
                    activation='gelu'
                )

    def forward(self, x):
        input = x
        B, M, D, N = x.shape
        x = x.reshape(B,M*D,N)
        # x = self.dw(x)
        x = self.dw_large(x) + self.dw_small(x)
        x = x.reshape(B,M,D,N)
        x = x.reshape(B*M,D,N)
        x = self.norm(x)
        x = x.reshape(B, M, D, N)

        # x = x.permute(0, 2, 1, 3)
        # x = x.reshape(B, D * M, N)
        # x = self.ffn2drop1(self.ffn2pw1(x))
        # x = self.ffn2act(x)
        # x = self.ffn2drop2(self.ffn2pw2(x))
        # x = x.reshape(B, D, M, N)
        # x = x.permute(0, 2, 1, 3)
        # x = x.reshape(B, M * D, N)
        # 
        # x = self.ffn1drop1(self.ffn1pw1(x))
        # x = self.ffn1act(x)
        # x = self.ffn1drop2(self.ffn1pw2(x))
        # x = x.reshape(B, M, D, N)

        x = x.permute(0, 3, 1, 2).reshape(B*N, M, D)
        x, attns = self.encoder(x)
        x = x.reshape(B, N, M, D).permute(0, 2, 3, 1)

        x = input + x
        # print("jb")
        return x

# 定义一个简单的卷积层
class FlexStrideConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(FlexStrideConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)

    def forward(self, x, stride):
        # 在每次前向传播时动态地指定stride
        return F.conv1d(x, self.conv.weight, self.conv.bias, stride=stride)


class Model(nn.Module):
    def __init__(self, configs):
        """
        v2.3 用iTransformer替换ModernTCN中的两个FFN，现在问题是速度略慢且提升不明显，但相当于iTransformer有提升
        """
        super(Model, self).__init__()

        ffn_ratio = configs.ffn_ratio
        large_size = configs.large_size
        small_size = configs.small_size
        dims = configs.dims

        nvars = configs.enc_in
        small_kernel_merged = configs.small_kernel_merged

        seq_len = configs.seq_len
        c_in = nvars,
        individual = 0
        target_window = configs.pred_len

        patch_stride = configs.patch_stride
        self.pred_len = configs.pred_len
        self.period_list = configs.period_list
        self.seq_len = configs.seq_len
        self.n_vars = c_in
        self.individual = individual
        # stem layer & down sampling layers(if needed)
        # self.downsample_layer = FlexStrideConv1d(1, dims[0], kernel_size=patch_size)
        # nn.BatchNorm1d(dims[0])
        self.downsample_layers = nn.ModuleList(
            [nn.Conv1d(1, dims[0], kernel_size=period, stride=period) for period in self.period_list]
        )
        self.patch_sizes = [period for period in self.period_list]
        self.patch_strides = [period for period in self.period_list]
        self.k = len(self.patch_sizes)

        self.upsampel_layers = nn.ModuleList(
            [Flatten_Head(individual, self.n_vars, dims[0] * ceil(self.seq_len / period), target_window,
                                 head_dropout=0) for period in self.period_list]
        )


        d_ffn = dims[0] * ffn_ratio
        blks = []
        for i in range(configs.e_layers):
            blk = Block(large_size=large_size[0], small_size=small_size[0], dmodel=dims[0], dff=d_ffn, nvars=nvars,
                        small_kernel_merged=small_kernel_merged, drop=configs.dropout)
            blks.append(blk)
        self.blocks = nn.ModuleList(blks)

        # Multi scale fusing (if needed)

        # head
        patch_num = seq_len // patch_stride


        d_model = dims[-1]
        self.head_nf = d_model * patch_num
        # self.head = nn.Linear(configs.seq_len, target_window)

    def forward_feature(self, x):

        x = x.permute(0, 2, 1).unsqueeze(-2)
        B, M, L, N = x.shape
        x = x.reshape(B * M, L, N)
        res = []
        for i in range(self.k):
            if N % self.patch_sizes[i] != 0:
                # stem layer padding
                pad_len = self.patch_sizes[i] - N % self.patch_sizes[i]
                pad = x[:, :, -1:].repeat(1, 1, pad_len)
                out = torch.cat([x, pad], dim=-1)
            else:
                out = x
            # 等价于overlap的Patch Embedding
            out = self.downsample_layers[i](out)
            _, D_, N_ = out.shape
            out = out.reshape(B, M, D_, N_)
            for blk in self.blocks:
                out = blk(out)
            out = self.upsampel_layers[i](out.reshape(B * M, D_, N_)).reshape(B, M, -1)

            res.append(out)
        res = torch.stack(res, dim=-1).mean(-1).permute(0, 2, 1)
        return res

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):

        # Normalization from Non-stationary Transformer
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev

        x = self.forward_feature(x)
        # x = self.head(x).permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        x = x * \
            (stdev[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len, 1))
        x = x + \
            (means[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len, 1))
        return x
