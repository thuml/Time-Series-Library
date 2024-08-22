import torch
from torch import nn


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

        #convffn1
        self.ffn1pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars)
        self.ffn1act = nn.GELU()
        self.ffn1pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars)
        self.ffn1drop1 = nn.Dropout(drop)
        self.ffn1drop2 = nn.Dropout(drop)

        #convffn2
        self.ffn2pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=dmodel)
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=dmodel)
        self.ffn2drop1 = nn.Dropout(drop)
        self.ffn2drop2 = nn.Dropout(drop)

        self.ffn_ratio = dff//dmodel

    def forward(self, x):
        input = x
        B, M, D, N = x.shape
        x = x.reshape(B,M*D,N)
        # x = self.dw(x)
        # x = self.dw_large(x) + self.dw_small(x)
        x = x.reshape(B,M,D,N)
        x = x.reshape(B*M,D,N)
        x = self.norm(x)
        x = x.reshape(B, M, D, N)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, D * M, N)
        x = self.ffn2drop1(self.ffn2pw1(x))
        x = self.ffn2act(x)
        x = self.ffn2drop2(self.ffn2pw2(x))
        x = x.reshape(B, D, M, N)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, M * D, N)

        x = self.ffn1drop1(self.ffn1pw1(x))
        x = self.ffn1act(x)
        x = self.ffn1drop2(self.ffn1pw2(x))
        x = x.reshape(B, M, D, N)

        x = input + x
        # print("jb")
        return x

class Model(nn.Module):
    def __init__(self, configs):

        super(Model, self).__init__()

        ffn_ratio = configs.ffn_ratio
        num_blocks = configs.num_blocks
        large_size = configs.large_size
        small_size = configs.small_size
        dims = configs.dims

        nvars = configs.enc_in
        small_kernel_merged = configs.small_kernel_merged

        seq_len = configs.seq_len
        c_in = nvars,
        individual = 0
        target_window = configs.pred_len

        patch_size = configs.patch_size
        patch_stride = configs.patch_stride
        self.pred_len = configs.pred_len

        # stem layer & down sampling layers(if needed)
        self.downsample_layer = nn.Sequential(

            nn.Conv1d(1, dims[0], kernel_size=patch_size, stride=patch_stride),
            nn.BatchNorm1d(dims[0])
        )
        self.patch_size = patch_size
        self.patch_stride = patch_stride


        # backbone
        self.num_stage = len(num_blocks)

        d_ffn = dims[0] * ffn_ratio
        blks = []
        for i in range(num_blocks[0]):
            blk = Block(large_size=large_size[0], small_size=small_size[0], dmodel=dims[0], dff=d_ffn, nvars=nvars,
                        small_kernel_merged=small_kernel_merged, drop=configs.dropout)
            blks.append(blk)
        self.blocks = nn.ModuleList(blks)

        # Multi scale fusing (if needed)

        # head
        patch_num = seq_len // patch_stride

        self.n_vars = c_in
        self.individual = individual
        d_model = dims[-1]
        self.head_nf = d_model * patch_num
        self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window,
                                 head_dropout=0)

    def forward_feature(self, x):
        x = x.unsqueeze(-2)
        B, M, D, N = x.shape
        x = x.reshape(B * M, D, N)
        if self.patch_size != self.patch_stride:
            # stem layer padding
            pad_len = self.patch_size - self.patch_stride
            pad = x[:, :, -1:].repeat(1, 1, pad_len)
            x = torch.cat([x, pad], dim=-1)
        # 等价于overlap的Patch Embedding
        x = self.downsample_layer(x)
        _, D_, N_ = x.shape
        x = x.reshape(B, M, D_, N_)
        for blk in self.blocks:
            x = blk(x)
        return x

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):

        # Normalization from Non-stationary Transformer
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev

        x = self.forward_feature(x.permute(0, 2, 1))
        x = self.head(x).permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        x = x * \
            (stdev[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len, 1))
        x = x + \
            (means[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len, 1))
        return x
