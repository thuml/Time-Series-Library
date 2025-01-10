# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 16:10:01 2025
@author: Murad
SISLab, USF
mmurad@usf.edu
https://github.com/Secure-and-Intelligent-Systems-Lab/WPMixer
"""

import torch.nn as nn
import torch
from layers.DWT_Decomposition import Decomposition


class TokenMixer(nn.Module):
    def __init__(self, input_seq=[], batch_size=[], channel=[], pred_seq=[], dropout=[], factor=[], d_model=[]):
        super(TokenMixer, self).__init__()
        self.input_seq = input_seq
        self.batch_size = batch_size
        self.channel = channel
        self.pred_seq = pred_seq
        self.dropout = dropout
        self.factor = factor
        self.d_model = d_model

        self.dropoutLayer = nn.Dropout(self.dropout)
        self.layers = nn.Sequential(nn.Linear(self.input_seq, self.pred_seq * self.factor),
                                    nn.GELU(),
                                    nn.Dropout(self.dropout),
                                    nn.Linear(self.pred_seq * self.factor, self.pred_seq)
                                    )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.layers(x)
        x = x.transpose(1, 2)
        return x


class Mixer(nn.Module):
    def __init__(self,
                 input_seq=[],
                 out_seq=[],
                 batch_size=[],
                 channel=[],
                 d_model=[],
                 dropout=[],
                 tfactor=[],
                 dfactor=[]):
        super(Mixer, self).__init__()
        self.input_seq = input_seq
        self.pred_seq = out_seq
        self.batch_size = batch_size
        self.channel = channel
        self.d_model = d_model
        self.dropout = dropout
        self.tfactor = tfactor  # expansion factor for patch mixer
        self.dfactor = dfactor  # expansion factor for embedding mixer

        self.tMixer = TokenMixer(input_seq=self.input_seq, batch_size=self.batch_size, channel=self.channel,
                                 pred_seq=self.pred_seq, dropout=self.dropout, factor=self.tfactor,
                                 d_model=self.d_model)
        self.dropoutLayer = nn.Dropout(self.dropout)
        self.norm1 = nn.BatchNorm2d(self.channel)
        self.norm2 = nn.BatchNorm2d(self.channel)

        self.embeddingMixer = nn.Sequential(nn.Linear(self.d_model, self.d_model * self.dfactor),
                                            nn.GELU(),
                                            nn.Dropout(self.dropout),
                                            nn.Linear(self.d_model * self.dfactor, self.d_model))

    def forward(self, x):
        '''
        Parameters
        ----------
        x : input: [Batch, Channel, Patch_number, d_model]

        Returns
        -------
        x: output: [Batch, Channel, Patch_number, d_model]

        '''
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.dropoutLayer(self.tMixer(x))
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = x + self.dropoutLayer(self.embeddingMixer(x))
        return x


class ResolutionBranch(nn.Module):
    def __init__(self,
                 input_seq=[],
                 pred_seq=[],
                 batch_size=[],
                 channel=[],
                 d_model=[],
                 dropout=[],
                 embedding_dropout=[],
                 tfactor=[],
                 dfactor=[],
                 patch_len=[],
                 patch_stride=[]):
        super(ResolutionBranch, self).__init__()
        self.input_seq = input_seq
        self.pred_seq = pred_seq
        self.batch_size = batch_size
        self.channel = channel
        self.d_model = d_model
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.tfactor = tfactor
        self.dfactor = dfactor
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.patch_num = int((self.input_seq - self.patch_len) / self.patch_stride + 2)

        self.patch_norm = nn.BatchNorm2d(self.channel)
        self.patch_embedding_layer = nn.Linear(self.patch_len, self.d_model)  # shared among all channels
        self.mixer1 = Mixer(input_seq=self.patch_num,
                            out_seq=self.patch_num,
                            batch_size=self.batch_size,
                            channel=self.channel,
                            d_model=self.d_model,
                            dropout=self.dropout,
                            tfactor=self.tfactor,
                            dfactor=self.dfactor)
        self.mixer2 = Mixer(input_seq=self.patch_num,
                            out_seq=self.patch_num,
                            batch_size=self.batch_size,
                            channel=self.channel,
                            d_model=self.d_model,
                            dropout=self.dropout,
                            tfactor=self.tfactor,
                            dfactor=self.dfactor)
        self.norm = nn.BatchNorm2d(self.channel)
        self.dropoutLayer = nn.Dropout(self.embedding_dropout)
        self.head = nn.Sequential(nn.Flatten(start_dim=-2, end_dim=-1),
                                  nn.Linear(self.patch_num * self.d_model, self.pred_seq))

    def forward(self, x):
        '''
        Parameters
        ----------
        x : input coefficient series: [Batch, channel, length_of_coefficient_series]

        Returns
        -------
        out : predicted coefficient series: [Batch, channel, length_of_pred_coeff_series]
        '''

        x_patch = self.do_patching(x)
        x_patch = self.patch_norm(x_patch)
        x_emb = self.dropoutLayer(self.patch_embedding_layer(x_patch))

        out = self.mixer1(x_emb)
        res = out
        out = res + self.mixer2(out)
        out = self.norm(out)

        out = self.head(out)
        return out

    def do_patching(self, x):
        x_end = x[:, :, -1:]
        x_padding = x_end.repeat(1, 1, self.patch_stride)
        x_new = torch.cat((x, x_padding), dim=-1)
        x_patch = x_new.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        return x_patch


class WPMixerCore(nn.Module):
    def __init__(self,
                 input_length=[],
                 pred_length=[],
                 wavelet_name=[],
                 level=[],
                 batch_size=[],
                 channel=[],
                 d_model=[],
                 dropout=[],
                 embedding_dropout=[],
                 tfactor=[],
                 dfactor=[],
                 device=[],
                 patch_len=[],
                 patch_stride=[],
                 no_decomposition=[],
                 use_amp=[]):
        super(WPMixerCore, self).__init__()
        self.input_length = input_length
        self.pred_length = pred_length
        self.wavelet_name = wavelet_name
        self.level = level
        self.batch_size = batch_size
        self.channel = channel
        self.d_model = d_model
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.device = device
        self.no_decomposition = no_decomposition
        self.tfactor = tfactor
        self.dfactor = dfactor
        self.use_amp = use_amp

        self.Decomposition_model = Decomposition(input_length=self.input_length,
                                                 pred_length=self.pred_length,
                                                 wavelet_name=self.wavelet_name,
                                                 level=self.level,
                                                 batch_size=self.batch_size,
                                                 channel=self.channel,
                                                 d_model=self.d_model,
                                                 tfactor=self.tfactor,
                                                 dfactor=self.dfactor,
                                                 device=self.device,
                                                 no_decomposition=self.no_decomposition,
                                                 use_amp=self.use_amp)

        self.input_w_dim = self.Decomposition_model.input_w_dim  # list of the length of the input coefficient series
        self.pred_w_dim = self.Decomposition_model.pred_w_dim  # list of the length of the predicted coefficient series

        self.patch_len = patch_len
        self.patch_stride = patch_stride

        # (m+1) number of resolutionBranch
        self.resolutionBranch = nn.ModuleList([ResolutionBranch(input_seq=self.input_w_dim[i],
                                                                pred_seq=self.pred_w_dim[i],
                                                                batch_size=self.batch_size,
                                                                channel=self.channel,
                                                                d_model=self.d_model,
                                                                dropout=self.dropout,
                                                                embedding_dropout=self.embedding_dropout,
                                                                tfactor=self.tfactor,
                                                                dfactor=self.dfactor,
                                                                patch_len=self.patch_len,
                                                                patch_stride=self.patch_stride) for i in
                                               range(len(self.input_w_dim))])

    def forward(self, xL):
        '''
        Parameters
        ----------
        xL : Look back window: [Batch, look_back_length, channel]

        Returns
        -------
        xT : Prediction time series: [Batch, prediction_length, output_channel]
        '''
        x = xL.transpose(1, 2)  # [batch, channel, look_back_length]

        # xA: approximation coefficient series,
        # xD: detail coefficient series
        # yA: predicted approximation coefficient series
        # yD: predicted detail coefficient series

        xA, xD = self.Decomposition_model.transform(x)

        yA = self.resolutionBranch[0](xA)
        yD = []
        for i in range(len(xD)):
            yD_i = self.resolutionBranch[i + 1](xD[i])
            yD.append(yD_i)

        y = self.Decomposition_model.inv_transform(yA, yD)
        y = y.transpose(1, 2)
        xT = y[:, -self.pred_length:, :]  # decomposition output is always even, but pred length can be odd

        return xT


class Model(nn.Module):
    def __init__(self, args, tfactor=5, dfactor=5, wavelet='db2', level=1, stride=8, no_decomposition=False):
        super(Model, self).__init__()
        self.args = args
        self.task_name = args.task_name
        self.wpmixerCore = WPMixerCore(input_length=self.args.seq_len,
                                       pred_length=self.args.pred_len,
                                       wavelet_name=wavelet,
                                       level=level,
                                       batch_size=self.args.batch_size,
                                       channel=self.args.c_out,
                                       d_model=self.args.d_model,
                                       dropout=self.args.dropout,
                                       embedding_dropout=self.args.dropout,
                                       tfactor=tfactor,
                                       dfactor=dfactor,
                                       device=self.args.device,
                                       patch_len=self.args.patch_len,
                                       patch_stride=stride,
                                       no_decomposition=no_decomposition,
                                       use_amp=self.args.use_amp)

    def forecast(self, x_enc, x_mark_enc, x_dec, batch_y_mark):
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        pred = self.wpmixerCore(x_enc)
        pred = pred[:, :, -self.args.c_out:]

        # De-Normalization
        dec_out = pred * (stdev[:, 0].unsqueeze(1).repeat(1, self.args.pred_len, 1))
        dec_out = dec_out + (means[:, 0].unsqueeze(1).repeat(1, self.args.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out  # [B, L, D]
        if self.task_name == 'imputation':
            raise NotImplementedError("Task imputation for WPMixer is temporarily not supported")
        if self.task_name == 'anomaly_detection':
            raise NotImplementedError("Task anomaly_detection for WPMixer is temporarily not supported")
        if self.task_name == 'classification':
            raise NotImplementedError("Task classification for WPMixer is temporarily not supported")
        return None
