import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Linear(in_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.embedding(x)
        
        memory = self.encoder(x)
        
        return memory


class LSTMDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Linear(in_dim, hidden_dim)
        
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, x_dec, memory):
        dec_emb = self.embedding(x_dec)
        
        # 把 Transformer 的最後一步的記憶，當作 LSTM 的初始狀態
        last_memory = memory[:, -1, :]  # (Batch, hidden_dim)
        
        # 展開成符合 LSTM Layer 的形狀
        h_0 = last_memory.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        
        c_0 = torch.zeros_like(h_0).contiguous()
        
        # LSTM 逐步解碼
        dec_out, _ = self.lstm(dec_emb, (h_0, c_0))  # (Batch, label_len + pred_len, hidden_dim)
        
        return dec_out

class Seq2SeqPrdictionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        
        self.fc = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        pred = self.fc(x)
        
        return pred

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        
        # 抓取參數
        enc_in = configs.enc_in
        dec_in = configs.dec_in
        c_out = configs.c_out
        d_model = configs.d_model
        n_heads = configs.n_heads
        e_layers = configs.e_layers
        d_layers = configs.d_layers
        d_ff = configs.d_ff
        dropout = configs.dropout
        
        self.encoder = TransformerEncoder(
            in_dim=enc_in,
            hidden_dim=d_model,
            num_heads=n_heads,
            d_ff=d_ff,
            num_layers=e_layers,
            dropout=dropout
        )

        self.decoder = LSTMDecoder(
            in_dim=dec_in,
            hidden_dim=d_model,
            num_layers=d_layers,
            dropout=dropout
        )

        self.head = Seq2SeqPredictionHead(
            in_dim=d_model, 
            out_dim=c_out
        )

    def forecast(self, x_enc, x_dec):
        # 1. 抽取歷史記憶
        memory = self.encoder(x_enc)
        
        # 2. 傳遞記憶並解碼未來
        dec_out = self.decoder(x_dec, memory)
        
        # 3. 輸出最終預測值
        predictions = self.head(dec_out)
        
        return predictions

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # 這裡必須把 x_enc 和 x_dec 都傳進去
            dec_out = self.forecast(x_enc, x_dec)
            
            # TSLib 只需要最後 pred_len 這段未來預測區間
            return dec_out[:, -self.pred_len:, :]
            
        return None
        