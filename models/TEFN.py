import torch
import torch.nn as nn


class NormLayer(nn.Module):
    def __init__(self):
        super(NormLayer, self).__init__()
        self.means = None
        self.stds = None

    def norm(self, x):
        self.means = x.mean(1, keepdim=True).detach()  # B x 1 x E
        x = x - self.means
        self.stds = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x = x / self.stds
        return x

    def denorm(self, x):
        x = x * self.stds + self.means
        return x


class EvidenceMachineKernel(nn.Module):
    def __init__(self, C, F):
        super(EvidenceMachineKernel, self).__init__()
        self.C = C
        self.F = 2 ** F
        self.C_weight = nn.Parameter(torch.randn(self.C, self.F))
        self.C_bias = nn.Parameter(torch.randn(self.C, self.F))

    def forward(self, x):
        x = torch.einsum('btc,cf->btcf', x, self.C_weight) + self.C_bias
        return x


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        if self.task_name.startswith('long_term_forecast') or self.task_name == 'short_term_forecast':
            self.nl = NormLayer()
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
            self.T_model = EvidenceMachineKernel(self.pred_len + self.seq_len, self.configs.e_layers)
            self.C_model = EvidenceMachineKernel(self.configs.enc_in, self.configs.e_layers)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc [B, T, C]
        x = self.nl.norm(x_enc)
        # x [B, T, C]
        x = self.predict_linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.T_model(x.permute(0, 2, 1)).permute(0, 2, 1, 3) + self.C_model(x)
        x = torch.einsum('btcf->btc', x)
        x = self.nl.denorm(x)
        return x

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name.startswith('long_term_forecast'):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
