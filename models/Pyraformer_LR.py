import torch
import torch.nn as nn
from layers.Pyra_Layers import EncoderLayer, Decoder, Predictor
from layers.Pyra_Layers import Bottleneck_Construct, Conv_Construct, MaxPooling_Construct, AvgPooling_Construct
from layers.Pyra_Layers import get_mask, get_subsequent_mask, refer_points, get_k_q, get_q_k
from layers.Embed import DataEmbedding, TokenEmbedding
import pdb


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        self.d_model = opt.d_model
        if opt.task_name == 'classification':
            self.window_size = [4,4]
        else:
            self.window_size = [2]
        self.truncate = False
        self.mask, self.all_size = get_mask(opt.seq_len, self.window_size, 5, torch.device("cuda"))
        self.indexes = refer_points(self.all_size, self.window_size, torch.device("cuda"))

        self.layers = nn.ModuleList([
                EncoderLayer(opt.d_model, opt.d_model, opt.n_heads, d_k=64, d_v=64, dropout=opt.dropout, \
                    normalize_before=False) for i in range(opt.e_layers)
                ])

        if opt.data == 'm4':
            self.enc_embedding = TokenEmbedding(opt.enc_in, opt.d_model)
        else:
            self.enc_embedding = DataEmbedding(opt.enc_in, opt.d_model, opt.dropout)

        self.conv_layers = eval('Bottleneck_Construct')(opt.d_model, self.window_size, 64)

    def forward(self, x_enc, x_mark_enc):

        if self.opt.data == 'm4':
            seq_enc = self.enc_embedding(x_enc)
        else:
            seq_enc = self.enc_embedding(x_enc, x_mark_enc)

        mask = self.mask.repeat(len(seq_enc), 1, 1).to(x_enc.device)
        seq_enc = self.conv_layers(seq_enc)

        for i in range(len(self.layers)):
            seq_enc, _ = self.layers[i](seq_enc, mask)

        indexes = self.indexes.repeat(seq_enc.size(0), 1, 1, seq_enc.size(2)).to(seq_enc.device)
        indexes = indexes.view(seq_enc.size(0), -1, seq_enc.size(2))
        all_enc = torch.gather(seq_enc, 1, indexes)
        seq_enc = all_enc.view(seq_enc.size(0), self.all_size[0], -1)

        return seq_enc


class Model(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(self, opt):
        super().__init__()

        self.pred_len = opt.pred_len
        self.d_model = opt.d_model
        self.seq_len = opt.seq_len
        self.enc_in = opt.enc_in
        self.task_name = opt.task_name

        self.encoder = Encoder(opt)
        
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = Predictor(2*self.d_model, self.pred_len * opt.enc_in)
        elif self.task_name == 'imputation':
            self.projection = nn.Linear(2*self.d_model, self.enc_in, bias=True)
        elif self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(2*self.d_model, self.enc_in, bias=True)
        elif self.task_name == 'classification':
            self.act = torch.nn.functional.gelu
            self.dropout = nn.Dropout(opt.dropout)
            self.projection = nn.Linear(3*self.d_model * self.seq_len, opt.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """

        enc_output = self.encoder(x_enc, x_mark_enc)[:, -1, :]
        pred = self.projection(enc_output).view(enc_output.size(0), self.pred_len, -1)

        return pred

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        enc_output = self.encoder(x_enc, x_mark_enc)
        pred = self.projection(enc_output)

        return pred

    def anomaly_detection(self, x_enc, x_mark_enc):
        # enc
        enc_out = self.encoder(x_enc, x_mark_enc)
        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # enc
        enc_out = self.encoder(x_enc, x_mark_enc=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)

        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc, x_mark_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None