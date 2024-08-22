import torch
from torch import nn
import torch.nn.functional as F


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
    def forward(self,x):

        input = x
        B, M, D, N = x.shape
        x = x.reshape(B,M*D,N)
        # x = self.dw(x)
        x = self.dw_large(x) + self.dw_small(x)
        x = x.reshape(B,M,D,N)
        x = x.reshape(B*M,D,N)
        x = self.norm(x)
        x = x.reshape(B, M, D, N)
        x = x.reshape(B, M * D, N)

        x = self.ffn1drop1(self.ffn1pw1(x))
        x = self.ffn1act(x)
        x = self.ffn1drop2(self.ffn1pw2(x))
        x = x.reshape(B, M, D, N)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, D * M, N)
        x = self.ffn2drop1(self.ffn2pw1(x))
        x = self.ffn2act(x)
        x = self.ffn2drop2(self.ffn2pw2(x))
        x = x.reshape(B, D, M, N)
        x = x.permute(0, 2, 1, 3)

        x = input + x
        return x


class Stage(nn.Module):
    def __init__(self, ffn_ratio, num_blocks, large_size, small_size, dmodel, dw_model, nvars,
                 small_kernel_merged=False, drop=0.1):

        super(Stage, self).__init__()
        d_ffn = dmodel * ffn_ratio
        blks = []
        for i in range(num_blocks):
            blk = Block(large_size=large_size, small_size=small_size, dmodel=dmodel, dff=d_ffn, nvars=nvars, small_kernel_merged=small_kernel_merged, drop=drop)
            blks.append(blk)

        self.blocks = nn.ModuleList(blks)

    def forward(self, x):

        for blk in self.blocks:
            x = blk(x)

        return x


class Model(nn.Module):
    def __init__(self, configs):

        super(Model, self).__init__()

        stem_ratio = configs.stem_ratio
        downsample_ratio = configs.downsample_ratio
        ffn_ratio = configs.ffn_ratio
        num_blocks = configs.num_blocks
        large_size = configs.large_size
        small_size = configs.small_size
        dims = configs.dims
        dw_dims = configs.dw_dims

        nvars = configs.enc_in
        small_kernel_merged = configs.small_kernel_merged
        drop_backbone = configs.dropout
        drop_head = 0
        use_multi_scale = configs.use_multi_scale
        affine = 0
        subtract_last = 0

        freq = configs.freq
        seq_len = configs.seq_len
        c_in = nvars,
        individual = 0
        target_window = configs.pred_len

        kernel_size = configs.kernel_size
        patch_size = configs.patch_size
        patch_stride = configs.patch_stride
        self.pred_len = configs.pred_len


        # stem layer & down sampling layers(if needed)
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(

            nn.Conv1d(1, dims[0], kernel_size=patch_size, stride=patch_stride),
            nn.BatchNorm1d(dims[0])
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.BatchNorm1d(dims[i]),
                nn.Conv1d(dims[i], dims[i + 1], kernel_size=downsample_ratio, stride=downsample_ratio),
            )
            self.downsample_layers.append(downsample_layer)
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.downsample_ratio = downsample_ratio

        if freq == 'h':
            time_feature_num = 4
        elif freq == 't':
            time_feature_num = 5
        else:
            raise NotImplementedError("time_feature_num should be 4 or 5")

        self.te_patch = nn.Sequential(

            nn.Conv1d(time_feature_num, time_feature_num, kernel_size=patch_size, stride=patch_stride,groups=time_feature_num),
            nn.Conv1d(time_feature_num, dims[0], kernel_size=1, stride=1, groups=1),
            nn.BatchNorm1d(dims[0]))

        # backbone
        self.num_stage = len(num_blocks)
        self.stages = nn.ModuleList()
        for stage_idx in range(self.num_stage):
            layer = Stage(ffn_ratio, num_blocks[stage_idx], large_size[stage_idx], small_size[stage_idx], dmodel=dims[stage_idx],
                          dw_model=dw_dims[stage_idx], nvars=nvars, small_kernel_merged=small_kernel_merged, drop=configs.dropout)
            self.stages.append(layer)

        # Multi scale fusing (if needed)
        self.use_multi_scale = use_multi_scale
        self.up_sample_ratio = downsample_ratio

        self.lat_layer = nn.ModuleList()
        self.smooth_layer = nn.ModuleList()
        self.up_sample_conv = nn.ModuleList()
        for i in range(self.num_stage):
            align_dim = dims[-1]
            lat = nn.Conv1d(dims[i], align_dim, kernel_size=1,
                            stride=1)
            self.lat_layer.append(lat)
            smooth = nn.Conv1d(align_dim, align_dim, kernel_size=3, stride=1, padding=1)
            self.smooth_layer.append(smooth)

            up_conv = nn.Sequential(
                nn.ConvTranspose1d(align_dim, align_dim, kernel_size=self.up_sample_ratio, stride=self.up_sample_ratio),
                nn.BatchNorm1d(align_dim))
            self.up_sample_conv.append(up_conv)

        # head
        patch_num = seq_len // patch_stride

        self.n_vars = c_in
        self.individual = individual
        d_model = dims[-1]
        if patch_num % pow(downsample_ratio, (self.num_stage - 1)) == 0:
            self.head_nf = d_model * patch_num // pow(downsample_ratio, (self.num_stage - 1))
        else:
            self.head_nf = d_model * (patch_num // pow(downsample_ratio, (self.num_stage - 1)) + 1)
        self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window,
                                 head_dropout=0)


    def forward_feature(self, x):

        B,M,L=x.shape

        x = x.unsqueeze(-2)
        for i in range(self.num_stage):
            B, M, D, N = x.shape
            x = x.reshape(B * M, D, N)
            if i==0:
                if self.patch_size != self.patch_stride:
                    # stem layer padding
                    pad_len = self.patch_size - self.patch_stride
                    pad = x[:,:,-1:].repeat(1,1,pad_len)
                    x = torch.cat([x,pad],dim=-1)
            else:
                if N % self.downsample_ratio != 0:
                    pad_len = self.downsample_ratio - (N % self.downsample_ratio)
                    x = torch.cat([x, x[:, :, -pad_len:]],dim=-1)
            # 等价于overlap的Patch Embedding
            x = self.downsample_layers[i](x)
            _, D_, N_ = x.shape
            x = x.reshape(B, M, D_, N_)
            x = self.stages[i](x)
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
