import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np

from scipy.signal import find_peaks
from scipy.signal import periodogram
import pywt
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class Preprocessor(nn.Module):
    def __init__(self, window_size=5):
        super().__init__()
        self.window_size = window_size

    def moving_average(self, signal):
        pad_signal = torch.nn.functional.pad(signal, (0, 0, self.window_size - 1, 0), mode='reflect')
        cumsum_signal = torch.cumsum(pad_signal, dim=1)
        ma_signal = (cumsum_signal[:, self.window_size:] - cumsum_signal[:, :-self.window_size]) / self.window_size
        return ma_signal
    
    def apply_fft_denoising(self, signal, threshold_factor=0.1):
        fft_result = torch.fft.fft(signal, dim=1)
        threshold = threshold_factor * torch.max(torch.abs(fft_result), dim=1, keepdim=True)[0]
        fft_result[torch.abs(fft_result) < threshold] = 0
        denoised_signal = torch.fft.ifft(fft_result, dim=1)
        return denoised_signal.real

    def forward(self, original_signal):
        ma_signal = self.apply_fft_denoising(original_signal)
        reference_noise = original_signal - ma_signal
        return reference_noise


class AdaptiveNoiseCanceller(nn.Module):
    def __init__(self, filter_order, mu, batch, c_in):
        super().__init__()
        self.filter_order = filter_order
        self.mu = mu
        self.weights = torch.randn((batch, filter_order, c_in)) * 0.1

    def forward(self, noisy_signal, reference_signal):
        B, L, C = noisy_signal.size()
        denoised_signal = torch.zeros_like(noisy_signal)

        for n in range(self.filter_order, L):
            if n < self.filter_order:
                continue
            x = reference_signal[:, n - self.filter_order:n, :].flip(dims=[1]).to(noisy_signal.device)
            weights = self.weights.to(x.device)
            y = torch.sum(weights * x, dim=1)
            e = noisy_signal[:, n, :] - y
            probabilities = 2 * self.mu * e
            probabilities = probabilities.unsqueeze(1).expand(-1, 10, -1).to(x.device)
            weights.data += probabilities * x
            denoised_signal[:, n, :] = e

        denoised_signal[:, :self.filter_order, :] = noisy_signal[:, :self.filter_order, :]
        return denoised_signal


class NoiseCancellation(nn.Module):
    def __init__(self, filter_order=5, mu=0.01, window_size=5, batch_size = 32, c_in = 21, 
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.device = device
        self.filter_order = filter_order
        self.window_size = window_size
        self.preprocessor = Preprocessor(window_size)
        self.anc = AdaptiveNoiseCanceller(filter_order, mu, batch_size, c_in)

    def forward(self, x):
        reference_noise = self.preprocessor(x)
        denoised_data = self.anc(x, reference_noise)
        return denoised_data

def optimized_linear_interpolation_padding(x, seq_len, period,mode='linear'):

    B,C,T = x.shape
    if seq_len % period != 0:
        length = ((seq_len // period) + 1) * period
    else:
        length = seq_len
        return x,length 
    last_period_start = seq_len - (seq_len % period)
    remaining_length = seq_len - last_period_start
    last_period_data = x
    target = length
    if mode =='linear':
        interpolated_data = F.interpolate(last_period_data, size=target, mode='linear', align_corners=False)
    elif mode =='nearest':
        interpolated_data = F.interpolate(last_period_data, size=target, mode='nearest')
    else:
        last_period_data_reshaped = last_period_data.unsqueeze(1) 
        if mode =='bicubic':
            interpolated_data = F.interpolate(last_period_data_reshaped, size=(C, target), mode='bicubic', align_corners=True)
        elif mode =='bilinear':
            interpolated_data = F.interpolate(last_period_data_reshaped, size=(C, target), mode='bilinear', align_corners=True)
        elif mode =='nearest2':
            interpolated_data = F.interpolate(last_period_data_reshaped, size=(C, target), mode='nearest')        
        interpolated_data = interpolated_data.squeeze(1)
    return interpolated_data,length


class ComplexWeightedSumModel(nn.Module):
    def __init__(self, channels, d_model, normal, dropout):
        super(ComplexWeightedSumModel, self).__init__()

        self.weights_inner = nn.Parameter(torch.randn(1, channels, 1))
        self.weights_whole = nn.Parameter(torch.randn(1, channels, 1))

        self.norm_inner = nn.LayerNorm([channels, d_model])
        self.norm_whole = nn.LayerNorm([channels, d_model])
        
        self.linear1 = nn.Linear(d_model, d_model*2)
        self.linear2 = nn.Linear(d_model*2, d_model*2)
        self.linear3 = nn.Linear(d_model*2, d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.normal = normal
        self.drop = dropout

    def forward(self, inner, whole):
        if self.normal:
            inner = self.norm_inner(inner)
            whole = self.norm_whole(whole)

        weighted_inner = self.weights_inner * inner
        weighted_whole = self.weights_whole * whole
        
        sum_result = weighted_inner + weighted_whole

        mlp_output = F.relu(self.linear1(sum_result))
        mlp_output = self.dropout1(mlp_output)
        mlp_output = F.relu(self.linear2(mlp_output))
        mlp_output = self.dropout2(mlp_output)
        mlp_output = F.gelu(self.linear3(mlp_output))
        
        return mlp_output
      

class ComplexTensorProcessor(nn.Module):
    def __init__(self, d_model,configs):
        super().__init__()
        self.encoder1 = Encoder(
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
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.conv1 = nn.Conv1d(d_model, d_model *2, kernel_size=3, padding=0)
        self.point_conv = nn.Conv1d(d_model*2, d_model*2, kernel_size=1)
        self.point = nn.Conv1d(d_model*2, d_model, kernel_size=1)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)

        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.attention = configs.attention

    def forward(self, x):
  
        B,C,N,P = x.shape
        x = x.reshape(B*C,N,P)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.point_conv(x)
        x = self.point(x)

        if self.attention :
          x = x.transpose(1, 2) 
          x = self.layer_norm1(x) 
          x, attns = self.encoder1(x, attn_mask=None)
          x = x.transpose(1, 2)
            
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.squeeze(2)
        x = self.mlp(x)
        x = x.reshape(B,C,P)
        return x
def STFT_for_Period(x, k=3, n_fft=16, hop_length=8, win_length=16, window='hann'):
    if window == 'hann':
        window_tensor = torch.hann_window(win_length, periodic=True).to(x.device)
    elif window == 'hamming':
        window_tensor = torch.hamming_window(win_length, periodic=True).to(x.device)
    else:
        raise ValueError(f"Unsupported window type: {window}")

    B, C, T = x.shape
    stft_results = []

    for c in range(C):
        single_channel_data = x[:, c, :]
        stft_result = torch.stft(single_channel_data, n_fft=n_fft, hop_length=hop_length,
                                 win_length=win_length, window=window_tensor, return_complex=True)
        stft_results.append(stft_result)

    stft_results = torch.stack(stft_results, dim = 1)
    xf_magnitude = torch.abs(stft_results).mean(dim=0)
    frequency_list = xf_magnitude.permute(0,2,1)
    frequency_list[:,:, 0] = 0 
    k_amplitude_all = []
    k_index_all = []
    for c in range(C):
        top_values = []
        for t in range(frequency_list.shape[1]):
            _, top_list = torch.topk(frequency_list[c, t, :], k)
            if top_list[0] == 1:
              if top_list[1] < 1:
                chosen_index = top_list[2]
              else:
                chosen_index = top_list[1]
            else:
              chosen_index = top_list[0]       
            top_values.append(chosen_index)
        top_values = torch.tensor(top_values)
        k_amplitude, k_index = torch.topk(top_values.flatten(), 1) 
        k_amplitude_all.append(k_amplitude)
        k_index_all.append(k_index)

    k_amplitude_all = torch.stack(k_amplitude_all)
    k_index_all = torch.stack(k_index_all)
    period_list = T // k_amplitude_all
    return period_list

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.patchH =configs.patchH
        self.patchW =configs.patchW
        self.strideH = configs.strideH
        self.strideW = configs.strideW
        self.B = configs.batch_size
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
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
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.value_embedding = nn.Linear(self.patchH*self.patchW, configs.d_model)
    
        self.CWSM = ComplexWeightedSumModel(configs.dec_in,configs.d_model,configs.normal,configs.dropout)
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        self.CTP = ComplexTensorProcessor(configs.d_model,configs) 
        self.noise_cancellation = NoiseCancellation(filter_order=10, mu=0.01, window_size=5, batch_size = self.B, c_in =configs.dec_in)
    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        B, T, C = x_enc.shape
        
        whole_out = self.enc_embedding(x_enc, x_mark_enc)
        whole_out, attns = self.encoder(whole_out, attn_mask=None)
        whole_out = whole_out[:, :C, :]

        x_enc = self.noise_cancellation(x_enc)
        x = x_enc.permute(0,2,1)
        period_list = STFT_for_Period(x)  
        patch_size = [self.patchH,self.patchW ]
        stride = (self.strideH,self.strideW)
        period_to_channels = defaultdict(list)
        for i in range(C):
            period = period_list[i].item()
            period_to_channels[period].append(i)
        
        CI = []

        channel_order = []
        for period, channels in period_to_channels.items():
            x_selected = x[:, channels, :]
            if period < self.patchW:
              period = self.patchW
            x_padded, length = optimized_linear_interpolation_padding(x_selected, self.seq_len, period)
            N_per = length // period
            out = x_padded.reshape(B, len(channels), N_per, period).contiguous()
            patches = out.unfold(2, patch_size[0], stride[0]).unfold(3, patch_size[1], stride[1])
            patches_reshape = patches.contiguous().view(B, len(channels), -1, patch_size[0] * patch_size[1])
            embedfor = torch.reshape(patches_reshape, (patches_reshape.shape[0]*len(channels) , patches_reshape.shape[2], patches_reshape.shape[3]))
            enc_out = self.value_embedding(embedfor)
            enc_out = torch.reshape(enc_out, (B, len(channels), enc_out.shape[-2], enc_out.shape[-1]))
            CI_out = self.CTP(enc_out)
            CI.append(CI_out)
            channel_order.extend(channels)

        dec_out = torch.cat(CI, dim=1) 
        original_order_index = torch.argsort(torch.tensor(channel_order))
        dec_out = dec_out[:, original_order_index, :] 
        dec_out = self.CWSM(dec_out,whole_out)
        dec_out = self.projection(dec_out).permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]

        return None
