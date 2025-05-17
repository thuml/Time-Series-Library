import torch
import torch.nn as nn
import math
from einops import rearrange

from layers.SelfAttention_Family import AttentionLayer, FullAttention


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_hidden: int = 512):
        super(FeedForward, self).__init__()

        self.linear_1 = torch.nn.Linear(d_model, d_hidden)
        self.linear_2 = torch.nn.Linear(d_hidden, d_model)
        self.activation = torch.nn.GELU()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        mha: AttentionLayer,
        d_hidden: int,
        dropout: float = 0,
        channel_wise=False,
    ):
        super(Encoder, self).__init__()

        self.channel_wise = channel_wise
        if self.channel_wise:
            self.conv = torch.nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=1,
                stride=1,
                padding=0,
                padding_mode="reflect",
            )
        self.MHA = mha
        self.feedforward = FeedForward(d_model=d_model, d_hidden=d_hidden)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.layerNormal_1 = torch.nn.LayerNorm(d_model)
        self.layerNormal_2 = torch.nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        q = residual
        if self.channel_wise:
            x_r = self.conv(x.permute(0, 2, 1)).transpose(1, 2)
            k = x_r
            v = x_r
        else:
            k = residual
            v = residual
        x, score = self.MHA(q, k, v, attn_mask=None)
        x = self.dropout(x)
        x = self.layerNormal_1(x + residual)

        residual = x
        x = self.feedforward(residual)
        x = self.dropout(x)
        x = self.layerNormal_2(x + residual)

        return x, score


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_channel = configs.enc_in
        self.N = configs.e_layers
        # Embedding
        self.d_model = configs.d_model
        self.d_hidden = configs.d_ff
        self.n_heads = configs.n_heads
        self.mask = True
        self.dropout = configs.dropout

        self.stride1 = 8
        self.patch_len1 = 8
        self.stride2 = 8
        self.patch_len2 = 16
        self.stride3 = 7
        self.patch_len3 = 24
        self.stride4 = 6
        self.patch_len4 = 32
        self.patch_num1 = int((self.seq_len - self.patch_len2) // self.stride2) + 2
        self.padding_patch_layer1 = nn.ReplicationPad1d((0, self.stride1))
        self.padding_patch_layer2 = nn.ReplicationPad1d((0, self.stride2))
        self.padding_patch_layer3 = nn.ReplicationPad1d((0, self.stride3))
        self.padding_patch_layer4 = nn.ReplicationPad1d((0, self.stride4))

        self.shared_MHA = nn.ModuleList(
            [
                AttentionLayer(
                    FullAttention(mask_flag=self.mask),
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                )
                for _ in range(self.N)
            ]
        )

        self.shared_MHA_ch = nn.ModuleList(
            [
                AttentionLayer(
                    FullAttention(mask_flag=self.mask),
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                )
                for _ in range(self.N)
            ]
        )

        self.encoder_list = nn.ModuleList(
            [
                Encoder(
                    d_model=self.d_model,
                    mha=self.shared_MHA[ll],
                    d_hidden=self.d_hidden,
                    dropout=self.dropout,
                    channel_wise=False,
                )
                for ll in range(self.N)
            ]
        )

        self.encoder_list_ch = nn.ModuleList(
            [
                Encoder(
                    d_model=self.d_model,
                    mha=self.shared_MHA_ch[0],
                    d_hidden=self.d_hidden,
                    dropout=self.dropout,
                    channel_wise=True,
                )
                for ll in range(self.N)
            ]
        )

        pe = torch.zeros(self.patch_num1, self.d_model)
        for pos in range(self.patch_num1):
            for i in range(0, self.d_model, 2):
                wavelength = 10000 ** ((2 * i) / self.d_model)
                pe[pos, i] = math.sin(pos / wavelength)
                pe[pos, i + 1] = math.cos(pos / wavelength)
        pe = pe.unsqueeze(0)  # add a batch dimention to your pe matrix
        self.register_buffer("pe", pe)

        self.embedding_channel = nn.Conv1d(
            in_channels=self.d_model * self.patch_num1,
            out_channels=self.d_model,
            kernel_size=1,
        )

        self.embedding_patch_1 = torch.nn.Conv1d(
            in_channels=1,
            out_channels=self.d_model // 4,
            kernel_size=self.patch_len1,
            stride=self.stride1,
        )
        self.embedding_patch_2 = torch.nn.Conv1d(
            in_channels=1,
            out_channels=self.d_model // 4,
            kernel_size=self.patch_len2,
            stride=self.stride2,
        )
        self.embedding_patch_3 = torch.nn.Conv1d(
            in_channels=1,
            out_channels=self.d_model // 4,
            kernel_size=self.patch_len3,
            stride=self.stride3,
        )
        self.embedding_patch_4 = torch.nn.Conv1d(
            in_channels=1,
            out_channels=self.d_model // 4,
            kernel_size=self.patch_len4,
            stride=self.stride4,
        )

        self.out_linear_1 = torch.nn.Linear(self.d_model, self.pred_len // 8)
        self.out_linear_2 = torch.nn.Linear(
            self.d_model + self.pred_len // 8, self.pred_len // 8
        )
        self.out_linear_3 = torch.nn.Linear(
            self.d_model + 2 * self.pred_len // 8, self.pred_len // 8
        )
        self.out_linear_4 = torch.nn.Linear(
            self.d_model + 3 * self.pred_len // 8, self.pred_len // 8
        )
        self.out_linear_5 = torch.nn.Linear(
            self.d_model + self.pred_len // 2, self.pred_len // 8
        )
        self.out_linear_6 = torch.nn.Linear(
            self.d_model + 5 * self.pred_len // 8, self.pred_len // 8
        )
        self.out_linear_7 = torch.nn.Linear(
            self.d_model + 6 * self.pred_len // 8, self.pred_len // 8
        )
        self.out_linear_8 = torch.nn.Linear(
            self.d_model + 7 * self.pred_len // 8,
            self.pred_len - 7 * (self.pred_len // 8),
        )

        self.remap = torch.nn.Linear(self.d_model, self.seq_len)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Multi-scale embedding
        x_i = x_enc.permute(0, 2, 1)

        x_i_p1 = x_i
        x_i_p2 = self.padding_patch_layer2(x_i)
        x_i_p3 = self.padding_patch_layer3(x_i)
        x_i_p4 = self.padding_patch_layer4(x_i)
        encoding_patch1 = self.embedding_patch_1(
            rearrange(x_i_p1, "b c l -> (b c) l").unsqueeze(-1).permute(0, 2, 1)
        ).permute(0, 2, 1)
        encoding_patch2 = self.embedding_patch_2(
            rearrange(x_i_p2, "b c l -> (b c) l").unsqueeze(-1).permute(0, 2, 1)
        ).permute(0, 2, 1)
        encoding_patch3 = self.embedding_patch_3(
            rearrange(x_i_p3, "b c l -> (b c) l").unsqueeze(-1).permute(0, 2, 1)
        ).permute(0, 2, 1)
        encoding_patch4 = self.embedding_patch_4(
            rearrange(x_i_p4, "b c l -> (b c) l").unsqueeze(-1).permute(0, 2, 1)
        ).permute(0, 2, 1)

        encoding_patch = (
            torch.cat(
                (encoding_patch1, encoding_patch2, encoding_patch3, encoding_patch4),
                dim=-1,
            )
            + self.pe
        )
        # Temporal encoding
        for i in range(self.N):
            encoding_patch = self.encoder_list[i](encoding_patch)[0]

        # Channel-wise encoding
        x_patch_c = rearrange(
            encoding_patch, "(b c) p d -> b c (p d)", b=x_enc.shape[0], c=self.d_channel
        )
        x_ch = self.embedding_channel(x_patch_c.permute(0, 2, 1)).transpose(
            1, 2
        )  # [b c d]

        encoding_1_ch = self.encoder_list_ch[0](x_ch)[0]

        # Semi Auto-regressive
        forecast_ch1 = self.out_linear_1(encoding_1_ch)
        forecast_ch2 = self.out_linear_2(
            torch.cat((encoding_1_ch, forecast_ch1), dim=-1)
        )
        forecast_ch3 = self.out_linear_3(
            torch.cat((encoding_1_ch, forecast_ch1, forecast_ch2), dim=-1)
        )
        forecast_ch4 = self.out_linear_4(
            torch.cat((encoding_1_ch, forecast_ch1, forecast_ch2, forecast_ch3), dim=-1)
        )
        forecast_ch5 = self.out_linear_5(
            torch.cat(
                (encoding_1_ch, forecast_ch1, forecast_ch2, forecast_ch3, forecast_ch4),
                dim=-1,
            )
        )
        forecast_ch6 = self.out_linear_6(
            torch.cat(
                (
                    encoding_1_ch,
                    forecast_ch1,
                    forecast_ch2,
                    forecast_ch3,
                    forecast_ch4,
                    forecast_ch5,
                ),
                dim=-1,
            )
        )
        forecast_ch7 = self.out_linear_7(
            torch.cat(
                (
                    encoding_1_ch,
                    forecast_ch1,
                    forecast_ch2,
                    forecast_ch3,
                    forecast_ch4,
                    forecast_ch5,
                    forecast_ch6,
                ),
                dim=-1,
            )
        )
        forecast_ch8 = self.out_linear_8(
            torch.cat(
                (
                    encoding_1_ch,
                    forecast_ch1,
                    forecast_ch2,
                    forecast_ch3,
                    forecast_ch4,
                    forecast_ch5,
                    forecast_ch6,
                    forecast_ch7,
                ),
                dim=-1,
            )
        )

        final_forecast = torch.cat(
            (
                forecast_ch1,
                forecast_ch2,
                forecast_ch3,
                forecast_ch4,
                forecast_ch5,
                forecast_ch6,
                forecast_ch7,
                forecast_ch8,
            ),
            dim=-1,
        ).permute(0, 2, 1)

        # De-Normalization
        dec_out = final_forecast * (
            stdev[:, 0].unsqueeze(1).repeat(1, self.pred_len, 1)
        )
        dec_out = dec_out + (means[:, 0].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]
        if self.task_name == "imputation":
            raise NotImplementedError(
                "Task imputation for WPMixer is temporarily not supported"
            )
        if self.task_name == "anomaly_detection":
            raise NotImplementedError(
                "Task anomaly_detection for WPMixer is temporarily not supported"
            )
        if self.task_name == "classification":
            raise NotImplementedError(
                "Task classification for WPMixer is temporarily not supported"
            )
        return None
