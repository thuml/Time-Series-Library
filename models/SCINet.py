import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from layers.SCINet_EncDec import SCINet


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len

        if 'ETT' in configs.data:
            hid_size = 1
        else:
            hid_size = 0.0625

        self.model = SCINet(
                output_len=configs.pred_len,
                input_len=configs.seq_len,
                input_dim=configs.enc_in,
                hid_size = hid_size,
                num_stacks=1,
                num_levels=2,
                num_decoder_layer=1,
                concat_len = 0,
                groups = 1,
                kernel = 5,
                dropout = 0.5,
                single_step_output_One = 0,
                positionalE = True,
                modified = True,
                RIN=False
            )

    def forecast(self, x_enc):
        return self.model(x_enc)

    def imputation(self, x_enc):
        return self.model(x_enc)

    def anomaly_detection(self, x_enc):
        return self.model(x_enc)

    def classification(self, x_enc):
        enc_out = self.model(x_enc)
        # Output
        # (batch_size, seq_length * d_model)
        output = enc_out.reshape(enc_out.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        return None
