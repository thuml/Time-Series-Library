# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 16:10:01 2025
@author: Murad
SISLab, USF
mmurad@usf.edu
"""

import torch.nn as nn
import torch
from layers.WPMixer_misc import Decomposition, RevIN

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.wpmixerCore = WPMixerCore(input_length = self.args.seq_len,
                                                      pred_length = self.args.pred_len,
                                                      wavelet_name = self.args.wavelet,
                                                      level = self.args.level,
                                                      batch_size = self.args.batch_size,
                                                      channel = self.args.c_in, 
                                                      d_model = self.args.d_model, 
                                                      dropout = self.args.dropout, 
                                                      embedding_dropout = self.args.embedding_dropout,
                                                      tfactor = self.args.tfactor, 
                                                      dfactor = self.args.dfactor, 
                                                      device = self.args.device,
                                                      patch_len = self.args.patch_len, 
                                                      patch_stride = self.args.stride,
                                                      no_decomposition = self.args.no_decomposition,
                                                      use_amp = self.args.use_amp)
        
        
    def forward(self, x, x_mark, xd, xd_mark):
        pred = self.wpmixerCore(x)
        pred = pred[:, :, -self.args.c_out:]
        return pred 


class WPMixerCore(nn.Module):
    def __init__(self, 
                 input_length = [], 
                 pred_length = [],
                 wavelet_name = [],
                 level = [],
                 batch_size = [],
                 channel = [],
                 d_model = [],
                 dropout = [],
                 embedding_dropout = [],
                 tfactor = [],
                 dfactor = [],
                 device = [],
                 patch_len = [],
                 patch_stride = [],
                 no_decomposition = [],
                 use_amp = []):
        
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
        
        self.Decomposition_model = Decomposition(input_length = self.input_length, 
                                        pred_length = self.pred_length,
                                        wavelet_name = self.wavelet_name,
                                        level = self.level,
                                        batch_size = self.batch_size,
                                        channel = self.channel,
                                        d_model = self.d_model,
                                        tfactor = self.tfactor,
                                        dfactor = self.dfactor,
                                        device = self.device,
                                        no_decomposition = self.no_decomposition,
                                        use_amp = self.use_amp)
        
        self.input_w_dim = self.Decomposition_model.input_w_dim # list of the length of the input coefficient series
        self.pred_w_dim = self.Decomposition_model.pred_w_dim # list of the length of the predicted coefficient series

        self.patch_len = patch_len
        self.patch_stride = patch_stride
        
        # (m+1) number of resolutionBranch
        self.resolutionBranch = nn.ModuleList([ResolutionBranch(input_seq = self.input_w_dim[i],
                                                           pred_seq = self.pred_w_dim[i],
                                                           batch_size = self.batch_size,
                                                           channel = self.channel,
                                                           d_model = self.d_model,
                                                           dropout = self.dropout,
                                                           embedding_dropout = self.embedding_dropout,
                                                           tfactor = self.tfactor,
                                                           dfactor = self.dfactor,
                                                           patch_len = self.patch_len,
                                                           patch_stride = self.patch_stride) for i in range(len(self.input_w_dim))])
        
        self.revin = RevIN(self.channel, eps=1e-5, affine = True, subtract_last = False)
        
    def forward(self, xL):
        '''
        Parameters
        ----------
        xL : Look back window: [Batch, look_back_length, channel]

        Returns
        -------
        xT : Prediction time series: [Batch, prediction_length, output_channel]
        '''
        
        x = self.revin(xL, 'norm')
        x = x.transpose(1, 2) # [batch, channel, look_back_length]
        
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
        y = y[:, -self.pred_length:, :] # decomposition output is always even, but pred length can be odd
        xT = self.revin(y, 'denorm')
        
        return xT


class ResolutionBranch(nn.Module):
    def __init__(self, 
                 input_seq = [],
                 pred_seq = [],
                 batch_size = [],
                 channel = [],
                 d_model = [],
                 dropout = [],
                 embedding_dropout = [],
                 tfactor = [], 
                 dfactor = [],
                 patch_len = [],
                 patch_stride = []):
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
        self.patch_embedding_layer = nn.Linear(self.patch_len, self.d_model) # shared among all channels
        self.mixer1 = Mixer(input_seq = self.patch_num, 
                            out_seq = self.patch_num,
                            batch_size = self.batch_size,
                            channel = self.channel,
                            d_model = self.d_model,
                            dropout = self.dropout,
                            tfactor = self.tfactor, 
                            dfactor = self.dfactor)
        self.mixer2 = Mixer(input_seq = self.patch_num, 
                            out_seq = self.patch_num, 
                            batch_size = self.batch_size, 
                            channel = self.channel,
                            d_model = self.d_model,
                            dropout = self.dropout,
                            tfactor = self.tfactor, 
                            dfactor = self.dfactor)
        self.norm = nn.BatchNorm2d(self.channel)
        self.dropoutLayer = nn.Dropout(self.embedding_dropout) 
        self.head = nn.Sequential(nn.Flatten(start_dim = -2 , end_dim = -1),
                                  nn.Linear(self.patch_num * self.d_model, self.pred_seq))
        self.revin = RevIN(self.channel)
        
    def forward(self, x):
        '''
        Parameters
        ----------
        x : input coefficient series: [Batch, channel, length_of_coefficient_series]
        
        Returns
        -------
        out : predicted coefficient series: [Batch, channel, length_of_pred_coeff_series]
        '''
        
        x = x.transpose(1, 2)
        x = self.revin(x, 'norm')
        x = x.transpose(1, 2)
        
        x_patch = self.do_patching(x) 
        x_patch  = self.patch_norm(x_patch)
        x_emb = self.dropoutLayer(self.patch_embedding_layer(x_patch)) 
        
        out =  self.mixer1(x_emb) 
        res = out
        out = res + self.mixer2(out)
        out = self.norm(out) 
        
        out = self.head(out) 
        out = out.transpose(1, 2)
        out = self.revin(out, 'denorm')
        out = out.transpose(1, 2)
        return out
    
    def do_patching(self, x):
        x_end = x[:, :, -1:]
        x_padding = x_end.repeat(1, 1, self.patch_stride)
        x_new = torch.cat((x, x_padding), dim = -1)
        x_patch = x_new.unfold(dimension = -1, size = self.patch_len, step = self.patch_stride) 
        return x_patch 
        
        
class Mixer(nn.Module):
    def __init__(self, 
                 input_seq = [],
                 out_seq = [], 
                 batch_size = [], 
                 channel = [], 
                 d_model = [],
                 dropout = [],
                 tfactor = [],
                 dfactor = []):
        super(Mixer, self).__init__()
        self.input_seq = input_seq
        self.pred_seq = out_seq
        self.batch_size = batch_size
        self.channel = channel
        self.d_model = d_model
        self.dropout = dropout
        self.tfactor = tfactor # expansion factor for patch mixer
        self.dfactor = dfactor # expansion factor for embedding mixer
        
        self.tMixer = TokenMixer(input_seq = self.input_seq, batch_size = self.batch_size, channel = self.channel, pred_seq = self.pred_seq, dropout = self.dropout, factor = self.tfactor, d_model = self.d_model)
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
    
    
class TokenMixer(nn.Module):
    def __init__(self, input_seq = [], batch_size = [], channel = [], pred_seq = [], dropout = [], factor = [], d_model = []):
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
    


