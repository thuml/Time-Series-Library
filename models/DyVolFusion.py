import torch
import torch.nn as nn
from layers.Embed import DataEmbedding

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # --- SOTA 核心 1: Channel-Independent Linear Baseline ---
        # 汲取 DLinear 的優勢，直接對每個特徵做時間維度的線性映射，提供穩定的趨勢基線
        self.linear_baseline = nn.Linear(self.seq_len, self.pred_len)
        
        # --- SOTA 核心 2: Deep Transformer 捕捉非線性波動 ---
        self.d_model = configs.d_model
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=configs.d_model, 
            nhead=configs.n_heads, 
            dim_feedforward=configs.d_ff,
            dropout=configs.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=configs.e_layers)
        
        # 將 Transformer 的輸出展平並預測未來的長度
        self.ai_head = nn.Linear(configs.d_model * self.seq_len, self.pred_len)
        
        # --- SOTA 核心 3: 動態融合閘門 (Dynamic Fusion) ---
        # 讓神經網路自己學習：現在該相信穩定的 Linear 基線，還是該相信 AI 的非線性預測
        self.fusion_weight = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        batch_size = x_enc.shape[0]
        
        # --- 1. Linear 物理基線預測 (保證模型的下限) ---
        # [Batch, Seq, Features] -> 轉置 -> [Batch, Features, Seq]
        x_enc_t = x_enc.permute(0, 2, 1)
        # 針對時間軸做線性映射 -> [Batch, Features, Pred]
        linear_out = self.linear_baseline(x_enc_t)
        # 轉回正常維度 -> [Batch, Pred, Features]
        linear_out = linear_out.permute(0, 2, 1)
        # 提取最後一欄 (Target_Vol) 作為基線預測
        base_pred = linear_out[:, :, -1:] # [Batch, Pred, 1]
        
        # --- 2. AI 深度預測 (尋找複雜特徵關聯) ---
        enc_in = self.enc_embedding(x_enc, x_mark_enc) # [Batch, Seq, d_model]
        trans_out = self.transformer(enc_in) # [Batch, Seq, d_model]
        
        # 展平後預測
        trans_flat = trans_out.reshape(batch_size, -1) # [Batch, Seq * d_model]
        ai_pred = self.ai_head(trans_flat).unsqueeze(-1) # [Batch, Pred, 1]
        
        # --- 3. 動態融合 (Dynamic Fusion) ---
        # 使用 sigmoid 確保權重在 0~1 之間
        w = torch.sigmoid(self.fusion_weight)
        final_pred = w * base_pred + (1 - w) * ai_pred
        
        return final_pred