import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        return lstm_out


class SelfAttention(nn.Module):
    # 增加 num_heads 參數
    def __init__(self, in_dim, hidden_dim, num_heads, out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = hidden_dim

        self.embedding = nn.Linear(in_dim, hidden_dim)

        # 這裡把 1 換成傳進來的 num_heads
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.embedding(x)

        attn_out, attn_wights = self.attention(
            query=x, key=x, value=x, need_weights=True
        )

        out = self.out_proj(attn_out)

        return out, attn_wights


class TimeSeriesPredictionHead(nn.Module):
    def __init__(self, in_dim, out_dim, pred_len=1):
        super().__init__()

        self.out_dim = out_dim
        self.pred_len = pred_len

        self.fc = nn.Linear(in_dim, out_dim * pred_len)

    def forward(self, x):
        last_step_forward = x[:, -1, :]

        pred = self.fc(last_step_forward)  # (Batch, pred_len * out_dim)

        pred = pred.view(-1, self.pred_len, self.out_dim)  # (Batch, pred_len, out_dim)

        return pred


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        in_features = configs.enc_in  
        lstm_hidden = configs.d_model  
        att_hidden = configs.d_model  
        att_out = configs.d_model  
        final_out = configs.c_out  
        
        # 🌟 把腳本裡的層數 (e_layers) 和頭數 (n_heads) 抓出來
        lstm_layers = configs.e_layers
        attention_heads = configs.n_heads

        # 把參數傳給底層模塊
        self.backbone = LSTM(
            in_dim=in_features, 
            hidden_dim=lstm_hidden, 
            num_layers=lstm_layers      # <-- LSTM 變成 2 層了！
        )
        
        self.neck = SelfAttention(
            in_dim=lstm_hidden, 
            hidden_dim=att_hidden, 
            num_heads=attention_heads,  # <-- Attention 變成 4 頭了！
            out_dim=att_out
        )
        
        self.head = TimeSeriesPredictionHead(
            in_dim=att_out, out_dim=final_out, pred_len=self.pred_len
        )

    def forecast(self, x_enc):
        features = self.backbone(x_enc)
        attendend_features, weights = self.neck(features)
        predictions = self.head(attendend_features)
        
        return predictions, weights

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dex, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out, attns = self.forecast(x_enc)
                
        return dec_out[:, -self.pred_len:, :]
