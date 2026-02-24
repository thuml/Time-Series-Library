import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralGARCHLayer(nn.Module):
    def __init__(self):
        super(NeuralGARCHLayer, self).__init__()
        self.omega = nn.Parameter(torch.tensor([0.01]))
        # 初始權重隨意，因為會經過 Sigmoid
        self.alpha = nn.Parameter(torch.tensor([-2.0])) 
        self.beta = nn.Parameter(torch.tensor([1.0]))   

    def forward(self, returns):
        batch_size, seq_len = returns.shape
        
        omega = F.softplus(self.omega)
        
        # 關鍵修復：強制 alpha 最大只能是 0.2，beta 最大只能是 0.8
        # 這樣相加永遠 <= 1，保證 GARCH 絕對不會爆炸
        alpha = torch.sigmoid(self.alpha) * 0.2
        beta = torch.sigmoid(self.beta) * 0.8
        
        h_list = []
        h_0 = torch.var(returns, dim=1) + 1e-6 
        h_list.append(h_0)
        
        for t in range(1, seq_len):
            h_t = omega + alpha * (returns[:, t-1] ** 2) + beta * h_list[t-1]
            h_list.append(h_t)
        
        h = torch.stack(h_list, dim=1)
        return torch.sqrt(h)

class Model(nn.Module):
    """
    端到端 E2E 架構: Neural GARCH + LSTM-Transformer + Dynamic Gate
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        
        # 假設 configs.enc_in 的最後一個維度是 'Log_Return'
        self.enc_in = configs.enc_in 

        # ==========================================
        # 模組 1: 神經 GARCH 層 (內建於模型中)
        # ==========================================
        self.garch_layer = NeuralGARCHLayer()
        # 負責把 GARCH 最後一天的預測值，推廣到未來 pred_len 天的線性基準
        self.garch_proj = nn.Linear(1, self.pred_len)

        # ==========================================
        # 模組 2: AI 殘差預測 (LSTM + Transformer)
        # ==========================================
        self.lstm = nn.LSTM(input_size=self.enc_in, hidden_size=self.d_model, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.residual_head = nn.Linear(self.d_model, self.pred_len)

        # ==========================================
        # 模組 3: 動態閘門 (Gating Network)
        # ==========================================
        self.gate_net = nn.Sequential(
            nn.Linear(self.enc_in + 1, self.d_model // 2), # 原始特徵 + GARCH 當前波動率
            nn.ReLU(),
            nn.Linear(self.d_model // 2, self.pred_len), 
            nn.Sigmoid()
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        x_enc: 包含了 [Open, High, Low, Close, Volume, Log_Return]
        假設 Log_Return 在最後一個維度 (index: -1)
        """
        # 提取收益率序列
        returns = x_enc[:, :, -1] 
        
        # --- 1. GARCH 基準預測 ---
        # 算出歷史序列的 GARCH 波動率 [Batch, seq_len]
        garch_historical_vol = self.garch_layer(returns) 
        # 取最後一天的波動率，線性推射到未來 pred_len 天
        garch_base_pred = self.garch_proj(garch_historical_vol[:, -1].unsqueeze(1))
        
        # --- 2. AI 殘差預測 ---
        lstm_out, _ = self.lstm(x_enc)
        trans_out = self.transformer(lstm_out)
        ai_residual_pred = self.residual_head(trans_out[:, -1, :])
        
        # --- 3. 閘門權重計算 ---
        # 將最後一天的市場特徵，與 GARCH 算出的最後一天波動率合併，做為裁判的判斷依據
        gate_input = torch.cat([x_enc[:, -1, :], garch_historical_vol[:, -1].unsqueeze(1)], dim=1)
        gate_weight = self.gate_net(gate_input)
        
        # --- 4. 最終融合 ---
        # 最終預測 = GARCH基準 + (閘門權重 * AI預測殘差)
        final_prediction = garch_base_pred + (gate_weight * ai_residual_pred)
        
        return final_prediction.unsqueeze(-1)