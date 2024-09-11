import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)



class ResBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, bias=True): 
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias) 
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.fc3 = nn.Linear(input_dim, output_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.ln = LayerNorm(output_dim, bias=bias)
        
    def forward(self, x):

        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + self.fc3(x)
        out = self.ln(out)
        return out


#TiDE
class Model(nn.Module):  
    """
    paper: https://arxiv.org/pdf/2304.08424.pdf 
    """
    def __init__(self, configs, bias=True, feature_encode_dim=2): 
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len  #L 
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len  #H 
        self.hidden_dim=configs.d_model
        self.res_hidden=configs.d_model 
        self.encoder_num=configs.e_layers
        self.decoder_num=configs.d_layers
        self.freq=configs.freq
        self.feature_encode_dim=feature_encode_dim
        self.decode_dim = configs.c_out
        self.temporalDecoderHidden=configs.d_ff
        dropout=configs.dropout

        
        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        
        self.feature_dim=freq_map[self.freq]


        flatten_dim = self.seq_len + (self.seq_len + self.pred_len) * self.feature_encode_dim

        self.feature_encoder = ResBlock(self.feature_dim, self.res_hidden, self.feature_encode_dim, dropout, bias)
        self.encoders = nn.Sequential(ResBlock(flatten_dim, self.res_hidden, self.hidden_dim, dropout, bias),*([ ResBlock(self.hidden_dim, self.res_hidden, self.hidden_dim, dropout, bias)]*(self.encoder_num-1)))
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.decoders = nn.Sequential(*([ ResBlock(self.hidden_dim, self.res_hidden, self.hidden_dim, dropout, bias)]*(self.decoder_num-1)),ResBlock(self.hidden_dim, self.res_hidden, self.decode_dim * self.pred_len, dropout, bias))
            self.temporalDecoder = ResBlock(self.decode_dim + self.feature_encode_dim, self.temporalDecoderHidden, 1, dropout, bias)
            self.residual_proj = nn.Linear(self.seq_len, self.pred_len, bias=bias)
        if self.task_name == 'imputation':
            self.decoders = nn.Sequential(*([ ResBlock(self.hidden_dim, self.res_hidden, self.hidden_dim, dropout, bias)]*(self.decoder_num-1)),ResBlock(self.hidden_dim, self.res_hidden, self.decode_dim * self.seq_len, dropout, bias))
            self.temporalDecoder = ResBlock(self.decode_dim + self.feature_encode_dim, self.temporalDecoderHidden, 1, dropout, bias)
            self.residual_proj = nn.Linear(self.seq_len, self.seq_len, bias=bias)
        if self.task_name == 'anomaly_detection':
            self.decoders = nn.Sequential(*([ ResBlock(self.hidden_dim, self.res_hidden, self.hidden_dim, dropout, bias)]*(self.decoder_num-1)),ResBlock(self.hidden_dim, self.res_hidden, self.decode_dim * self.seq_len, dropout, bias))
            self.temporalDecoder = ResBlock(self.decode_dim + self.feature_encode_dim, self.temporalDecoderHidden, 1, dropout, bias)
            self.residual_proj = nn.Linear(self.seq_len, self.seq_len, bias=bias)
            
        
    def forecast(self, x_enc, x_mark_enc, x_dec, batch_y_mark):
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        feature = self.feature_encoder(batch_y_mark)
        hidden = self.encoders(torch.cat([x_enc, feature.reshape(feature.shape[0], -1)], dim=-1))
        decoded = self.decoders(hidden).reshape(hidden.shape[0], self.pred_len, self.decode_dim)
        dec_out = self.temporalDecoder(torch.cat([feature[:,self.seq_len:], decoded], dim=-1)).squeeze(-1) + self.residual_proj(x_enc)
        
        
        # De-Normalization 
        dec_out = dec_out * (stdev[:, 0].unsqueeze(1).repeat(1, self.pred_len))
        dec_out = dec_out + (means[:, 0].unsqueeze(1).repeat(1, self.pred_len))
        return dec_out
    
    def imputation(self, x_enc, x_mark_enc, x_dec, batch_y_mark, mask):
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        feature = self.feature_encoder(x_mark_enc)
        hidden = self.encoders(torch.cat([x_enc, feature.reshape(feature.shape[0], -1)], dim=-1))
        decoded = self.decoders(hidden).reshape(hidden.shape[0], self.seq_len, self.decode_dim)
        dec_out = self.temporalDecoder(torch.cat([feature[:,:self.seq_len], decoded], dim=-1)).squeeze(-1) + self.residual_proj(x_enc)
    
        # De-Normalization 
        dec_out = dec_out * (stdev[:, 0].unsqueeze(1).repeat(1, self.seq_len))
        dec_out = dec_out + (means[:, 0].unsqueeze(1).repeat(1, self.seq_len))
        return dec_out
    
    
    def forward(self, x_enc, x_mark_enc, x_dec, batch_y_mark, mask=None):
        '''x_mark_enc is the exogenous dynamic feature described in the original paper'''
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            if batch_y_mark is None:
                batch_y_mark = torch.zeros((x_enc.shape[0], self.seq_len+self.pred_len, self.feature_dim)).to(x_enc.device).detach()
            else:
                batch_y_mark = torch.concat([x_mark_enc, batch_y_mark[:, -self.pred_len:, :]],dim=1)
            dec_out = torch.stack([self.forecast(x_enc[:, :, feature], x_mark_enc, x_dec, batch_y_mark) for feature in range(x_enc.shape[-1])],dim=-1)
            return dec_out # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = torch.stack([self.imputation(x_enc[:, :, feature], x_mark_enc, x_dec, batch_y_mark, mask) for feature in range(x_enc.shape[-1])],dim=-1)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            raise NotImplementedError("Task anomaly_detection for Tide is temporarily not supported")
        if self.task_name == 'classification':
            raise NotImplementedError("Task classification for Tide is temporarily not supported")
        return None
    
    



