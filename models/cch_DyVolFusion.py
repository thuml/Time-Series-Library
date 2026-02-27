import torch
import torch.nn as nn
import math

class moving_avg(nn.Module):
    """
    DLinear 中的移動平均區塊，用於提取趨勢
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    時間序列分解模塊 (季節性與趨勢)
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class PositionalEncoding(nn.Module):
    """
    Transformer 必備的位置編碼
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class Model(nn.Module):
    """
    Paper Name: Hybrid DLinear with Transformer-LSTM Residual Learning
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.task_name = getattr(configs, 'task_name', 'long_term_forecast')
        
        # ==========================================
        # 1. DLinear 模塊
        # ==========================================
        kernel_size = getattr(configs, 'moving_avg', 25)
        self.decomp = series_decomp(kernel_size)
        
        self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.linear_trend = nn.Linear(self.seq_len, self.pred_len)
        
        # ==========================================
        # 2. Transformer Encoder 模塊
        # ==========================================
        self.d_model = configs.d_model
        self.enc_embedding = nn.Linear(configs.enc_in, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=configs.n_heads,
            dim_feedforward=getattr(configs, 'd_ff', 2048),
            dropout=configs.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=configs.e_layers
        )
        
        # ==========================================
        # 3. LSTM Decoder 模塊
        # ==========================================
        self.lstm_hidden_size = self.d_model
        self.lstm_layers = getattr(configs, 'd_layers', 1)
        
        # TSlib 的 x_dec 長度通常為 label_len + pred_len
        self.dec_embedding = nn.Linear(configs.dec_in, self.lstm_hidden_size)
        
        self.lstm = nn.LSTM(
            input_size=self.lstm_hidden_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_layers,
            batch_first=True
        )
        
        self.residual_proj = nn.Linear(self.lstm_hidden_size, configs.c_out)
        
        # 用於將 Encoder 的最後一個狀態轉換為 LSTM 的初始隱藏狀態
        self.hidden_proj = nn.Linear(self.d_model, self.lstm_hidden_size)
        
        # ==========================================
        # 4. Sigmoid Gating 模塊
        # ==========================================
        # 透過 Encoder 提取的上下文向量來決定殘差的啟用程度
        self.gate = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # --- A. DLinear 線性預測 ---
        res_init, trend_init = self.decomp(x_enc)
        res_init = res_init.permute(0, 2, 1)     # [Batch, Channel, seq_len]
        trend_init = trend_init.permute(0, 2, 1) # [Batch, Channel, seq_len]
        
        seasonal_output = self.linear_seasonal(res_init).permute(0, 2, 1)
        trend_output = self.linear_trend(trend_init).permute(0, 2, 1)
        
        y_dlinear = seasonal_output + trend_output # [Batch, pred_len, c_out]

        # --- B. Transformer Encoder (特徵提取) ---
        enc_in = self.enc_embedding(x_enc)
        enc_in = self.pos_encoder(enc_in)
        enc_out = self.transformer_encoder(enc_in) # [Batch, seq_len, d_model]
        
        # 取得最後一個時間步的上下文向量
        context_vector = enc_out[:, -1, :] # [Batch, d_model]

        # --- C. LSTM Decoder (殘差預測) ---
        # 將 Context Vector 作為 LSTM 的初始狀態 (h_0, c_0)
        h_0 = self.hidden_proj(context_vector).unsqueeze(0).repeat(self.lstm_layers, 1, 1)
        c_0 = torch.zeros_like(h_0)
        
        dec_in = self.dec_embedding(x_dec)
        lstm_out, _ = self.lstm(dec_in, (h_0, c_0)) 
        
        # 取出預測長度 (pred_len) 的部分進行映射
        lstm_out = lstm_out[:, -self.pred_len:, :]
        y_residual = self.residual_proj(lstm_out) # [Batch, pred_len, c_out]

        # --- D. Sigmoid Gate (控制閥) ---
        # 基於上下文動態生成 0~1 的權重
        gate_weight = self.gate(context_vector).unsqueeze(1) # [Batch, 1, 1]

        # --- E. 融合 (Fusion) ---
        y_pred = y_dlinear + (gate_weight * y_residual)

        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            return y_pred
        else:
            return y_pred

# import torch
# import torch.nn as nn
# from layers.Embed import DataEmbedding

# class Model(nn.Module):
#     """
#     DLinear baseline + Transformer Encoder + LSTM Decoder with Sigmoid gate
#     """
#     def __init__(self, configs):
#         super().__init__()
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len

#         # 1) DLinear 基線 (per-channel)
#         self.linear_baseline = nn.Linear(self.seq_len, self.pred_len)

#         # 2) 殘差編碼：Transformer Encoder
#         self.enc_embedding = DataEmbedding(
#             configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
#         )
#         enc_layer = nn.TransformerEncoderLayer(
#             d_model=configs.d_model,
#             nhead=configs.n_heads,
#             dim_feedforward=configs.d_ff,
#             dropout=configs.dropout,
#             batch_first=True,
#         )
#         self.transformer = nn.TransformerEncoder(enc_layer, num_layers=configs.e_layers)

#         # 3) 殘差解碼：LSTM
#         self.lstm_dec = nn.LSTM(
#             input_size=configs.d_model,
#             hidden_size=configs.d_model,
#             num_layers=configs.d_layers,
#             batch_first=True,
#         )
#         self.residual_head = nn.Linear(configs.d_model, self.pred_len)

#         # 4) 閘門：是否啟用殘差（0~1）
#         self.gate = nn.Sequential(
#             nn.Linear(configs.d_model, configs.d_model // 2),
#             nn.ReLU(),
#             nn.Linear(configs.d_model // 2, 1),
#             nn.Sigmoid(),
#         )

#     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
#         # baseline
#         base = self.linear_baseline(x_enc.permute(0, 2, 1)).permute(0, 2, 1)  # [B, pred, C]

#         # 殘差特徵
#         enc = self.enc_embedding(x_enc, x_mark_enc)         # [B, seq, d_model]
#         enc = self.transformer(enc)                         # [B, seq, d_model]

#         # LSTM 解碼序列：以最後一個 hidden 作為 summary
#         lstm_out, _ = self.lstm_dec(enc)                    # [B, seq, d_model]
#         residual = self.residual_head(lstm_out[:, -1, :])   # [B, pred_len]
#         residual = residual.unsqueeze(-1)                   # [B, pred_len, 1]

#         # 閘門（可學習是否啟用殘差）
#         gate_w = self.gate(lstm_out[:, -1, :])              # [B, 1]
#         gate_w = gate_w.unsqueeze(-1)                       # [B, 1, 1]

#         # 最終輸出：baseline + gate * residual
#         out = base + gate_w * residual
#         return out