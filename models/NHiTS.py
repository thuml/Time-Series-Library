# Cell
import math
import random
import numpy as np

import torch as t
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, Tuple
from functools import partial

from layers.nhits_utils import RepeatVector

# Cell
# class _StaticFeaturesEncoder(nn.Module):
#     def __init__(self, in_features, out_features):
#         super(_StaticFeaturesEncoder, self).__init__()
#         layers = [nn.Dropout(p=0.5),
#                   nn.Linear(in_features=in_features, out_features=out_features),
#                   nn.ReLU()]
#         self.encoder = nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.encoder(x)
#         return x

# class _sEncoder(nn.Module):
#     def __init__(self, in_features, out_features, n_time_in):
#         super(_sEncoder, self).__init__()
#         layers = [nn.Dropout(p=0.5),
#                   nn.Linear(in_features=in_features, out_features=out_features),
#                   nn.ReLU()]
#         self.encoder = nn.Sequential(*layers)
#         self.repeat = RepeatVector(repeats=n_time_in)

#     def forward(self, x):
#         # Encode and repeat values to match time
#         x = self.encoder(x)
#         x = self.repeat(x) # [N,S_out] -> [N,S_out,T]
#         return x

# Cell
class IdentityBasis(nn.Module):
    def __init__(self, backcast_size: int, forecast_size: int, interpolation_mode: str):
        super().__init__()
        assert (interpolation_mode in ['linear','nearest']) or ('cubic' in interpolation_mode)
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size
        self.interpolation_mode = interpolation_mode

    def forward(self, theta: t.Tensor, insample_x_t: t.Tensor, outsample_x_t: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:

        backcast = theta[:, :self.backcast_size]
        knots = theta[:, self.backcast_size:]

        if self.interpolation_mode=='nearest':
            knots = knots[:,None,:]
            forecast = F.interpolate(knots, size=self.forecast_size, mode=self.interpolation_mode)
            forecast = forecast[:,0,:]
        elif self.interpolation_mode=='linear':
            knots = knots[:,None,:]
            forecast = F.interpolate(knots, size=self.forecast_size, mode=self.interpolation_mode) #, align_corners=True)
            forecast = forecast[:,0,:]
        elif 'cubic' in self.interpolation_mode:
            batch_size = int(self.interpolation_mode.split('-')[-1])
            knots = knots[:,None,None,:]
            forecast = t.zeros((len(knots), self.forecast_size)).to(knots.device)
            n_batches = int(np.ceil(len(knots)/batch_size))
            for i in range(n_batches):
                forecast_i = F.interpolate(knots[i*batch_size:(i+1)*batch_size], size=self.forecast_size, mode='bicubic') #, align_corners=True)
                forecast[i*batch_size:(i+1)*batch_size] += forecast_i[:,0,0,:]

        return backcast, forecast

# Cell
def init_weights(module, initialization):
    if type(module) == t.nn.Linear:
        if initialization == 'orthogonal':
            t.nn.init.orthogonal_(module.weight)
        elif initialization == 'he_uniform':
            t.nn.init.kaiming_uniform_(module.weight)
        elif initialization == 'he_normal':
            t.nn.init.kaiming_normal_(module.weight)
        elif initialization == 'glorot_uniform':
            t.nn.init.xavier_uniform_(module.weight)
        elif initialization == 'glorot_normal':
            t.nn.init.xavier_normal_(module.weight)
        elif initialization == 'lecun_normal':
            pass #t.nn.init.normal_(module.weight, 0.0, std=1/np.sqrt(module.weight.numel()))
        else:
            assert 1<0, f'Initialization {initialization} not found'

# Cell
ACTIVATIONS = ['ReLU',
               'Softplus',
               'Tanh',
               'SELU',
               'LeakyReLU',
               'PReLU',
               'Sigmoid']

class _NHiTSBlock(nn.Module):
    """
    N-HiTS block which takes a basis function as an argument.
    """
    def __init__(self, backcast_size, forecast_size,
                 n_pool_kernel_size, n_layers, n_theta_hidden, n_theta,
                 pooling_mode, activation, batch_norm, dropout, interpolation_mode):
        """
        """
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        self.n_pool_kernel_size = n_pool_kernel_size
        self.n_layers = n_layers
        self.n_theta_hidden = n_theta_hidden
        self.n_theta = n_theta
        self.pooling_mode = pooling_mode
        self.activation = activation
        self.batch_norm = batch_norm

        assert (self.pooling_mode in ['max','average'])       

        assert self.activation in ACTIVATIONS, f'{self.activation} is not in {ACTIVATIONS}'
        activ = getattr(nn, self.activation)()

        if self.pooling_mode == 'max':
            self.pooling_layer = nn.MaxPool1d(kernel_size=self.n_pool_kernel_size,
                                              stride=self.n_pool_kernel_size, ceil_mode=True)
        elif self.pooling_mode == 'average':
            self.pooling_layer = nn.AvgPool1d(kernel_size=self.n_pool_kernel_size,
                                              stride=self.n_pool_kernel_size, ceil_mode=True)

        hidden_layers = []
        for i in range(n_layers):
            hidden_layers += [nn.Linear(n_theta_hidden[i], n_theta_hidden[i+1]), activ]
            if batch_norm:
                hidden_layers += [nn.BatchNorm1d(n_theta_hidden[i+1])]
            if dropout > 0:
                hidden_layers += [nn.Dropout(p=dropout)]
        output_layer = [nn.Linear(n_theta_hidden[-1], n_theta)]
        self.layers = nn.Sequential(*(hidden_layers + output_layer))

        self.basis = IdentityBasis(backcast_size=backcast_size,
                                   forecast_size=forecast_size,
                                   interpolation_mode=interpolation_mode)

    def forward(self, insample_y: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        # insample_y: (B*, L)  ← B* = B*C
        x = self.pooling_layer(insample_y.unsqueeze(1)).squeeze(1)  # (B*, ceil(L/k))
        theta = self.layers(x)                                      # (B*, n_theta)
        backcast, forecast = self.basis(theta, None, None)          # (B*, L), (B*, H)
        return backcast, forecast

# Cell
class Model(nn.Module):
    """
    N-HiTS Model.
    """
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len

        blocks = self.create_stack(configs)
        self.blocks = t.nn.ModuleList(blocks)

    def create_stack(self, cfg):
        blocks = []
        for i, stack_type in enumerate(cfg.stack_types):
            for b in range(cfg.n_blocks[i]):
                bn_block = (len(blocks) == 0) and cfg.batch_normalization

                k = cfg.n_pool_kernel_size[i]            # pooling kernel for this stack
                d = cfg.n_freq_downsample[i]             # downsample factor for knots

                pooled_len = int(np.ceil(cfg.seq_len / k))
                n_knots = max(cfg.pred_len // d, 1)
                backcast_size = cfg.seq_len
                n_theta = backcast_size + n_knots

                n_theta_hidden = [pooled_len] + list(cfg.n_theta_hidden)

                blk = _NHiTSBlock(
                    backcast_size=backcast_size,
                    forecast_size=cfg.pred_len,
                    n_pool_kernel_size=k,
                    n_layers=cfg.n_layers,
                    n_theta_hidden=n_theta_hidden,
                    n_theta=n_theta,
                    pooling_mode=cfg.pooling_mode,
                    activation=cfg.activation,
                    batch_norm=bn_block,
                    dropout=cfg.dropout,
                    interpolation_mode=cfg.interpolation_mode,
                )
                # 초기화
                blk.layers.apply(partial(init_weights, initialization=cfg.initialization))
                blocks.append(blk)
        return blocks



    def forecast(self, insample_y: t.Tensor, insample_mask: t.Tensor):
        residuals     = insample_y.flip(-1)
        insample_mask = insample_mask.flip(-1)
        forecast = insample_y[:, -1:]  # (B*, 1)
        for block in self.blocks:
            backcast, block_forecast = block(residuals)
            residuals = (residuals - backcast) * insample_mask
            forecast  = forecast + block_forecast
        return forecast  # (B*, H)

    def forecast_decomposition(self, insample_y: t.Tensor, insample_mask: t.Tensor,
                               outsample_y: t.Tensor):

        residuals = insample_y.flip(dims=(-1,))
        insample_mask = insample_mask.flip(dims=(-1,))

        n_batch, n_channels, n_t = outsample_y.size(0), outsample_y.size(1), outsample_y.size(2)

        level = insample_y[:, -1:] # Level with Naive1
        block_forecasts = [ level.repeat(1, n_t) ]

        forecast = level
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(residuals)
            residuals = (residuals - backcast) * insample_mask
            forecast = forecast + block_forecast
            block_forecasts.append(block_forecast)

        # (n_batch, n_blocks, n_t)
        block_forecasts = t.stack(block_forecasts)
        block_forecasts = block_forecasts.permute(1,0,2)

        return forecast, block_forecasts


    def forward(self, batch_x: t.Tensor, batch_x_mark: t.Tensor,
                dec_inp: t.Tensor, batch_y_mark: t.Tensor,
                return_decomposition: bool=False):
        # batch_x: (B, L, C)
        B, L, C = batch_x.shape
        assert L == self.seq_len, f"seq_len mismatch: got {L}, expected {self.seq_len}"

        # 각 채널을 독립 univariate로 처리: (B,C,L) -> (B*C, L)
        insample_y = batch_x.transpose(1, 2).reshape(B * C, L)
        insample_mask = t.ones_like(insample_y)

        if return_decomposition:
            forecast_bc, block_forecasts_bc = self.forecast_decomposition(
                insample_y=insample_y, insample_mask=insample_mask
            )  # (B*C, H), (B*C, #blocks, H)
            # (B*C, H) -> (B, H, C)
            forecast = forecast_bc.view(B, C, self.pred_len).transpose(1, 2)
            # block_decomp도 필요하면 같은 방식으로 reshape
            return forecast

        forecast_bc = self.forecast(insample_y=insample_y, insample_mask=insample_mask)  # (B*C, H)
        forecast = forecast_bc.view(B, C, self.pred_len).transpose(1, 2)               # (B, H, C)
        return forecast
