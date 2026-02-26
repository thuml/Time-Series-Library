import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding # 引入 TSlib 強大的 Embedding 層

class NeuralGARCHLayer(nn.Module):
    def __init__(self, pred_len):
        super(NeuralGARCHLayer, self).__init__()
        self.pred_len = pred_len
        self.omega = nn.Parameter(torch.tensor([0.01]))
        self.alpha = nn.Parameter(torch.tensor([-2.0])) 
        self.beta = nn.Parameter(torch.tensor([1.0]))   
        
        # 修正 2：讓初始變異數變成可學習的參數，杜絕「偷看未來」
        self.h_0_param = nn.Parameter(torch.tensor([0.1]))

    def forward(self, returns):
        batch_size, seq_len = returns.shape
        
        omega = F.softplus(self.omega)
        alpha = torch.sigmoid(self.alpha) * 0.2
        beta = torch.sigmoid(self.beta) * 0.8
        
        # 初始化第一天的變異數
        h_t = F.softplus(self.h_0_param).expand(batch_size)
        
        # 修正 1：算出完整的歷史變異數，迴圈跑到涵蓋「最後一天(今天)」的收益率
        for t in range(seq_len):
            h_t = omega + alpha * (returns[:, t] ** 2) + beta * h_t
            
        # 此時的 h_t 已經吸收了輸入序列的最後一天資訊，這是預測明天的基準
        h_future = h_t
        
        # 修正 3：使用正統 GARCH 數學公式遞迴預測未來 pred_len 天
        future_h_list = []
        for i in range(self.pred_len):
            future_h_list.append(h_future)
            # 因為未來還沒有真實的 return，根據 GARCH 理論，預期收益率的平方等於預期的變異數
            # 數學推導：E[h_{t+2}] = omega + alpha * E[r_{t+1}^2] + beta * h_{t+1} = omega + (alpha+beta)*h_{t+1}
            h_future = omega + (alpha + beta) * h_future 
            
        # 將未來 pred_len 天的變異數預測堆疊起來
        future_h_tensor = torch.stack(future_h_list, dim=1) # [Batch, pred_len]
        
        # 回傳標準差 (波動率)
        return torch.sqrt(future_h_tensor)

class Model(nn.Module):
    """
    真正嚴謹的端到端: Auto-Regressive GARCH + Time-Embedded Transformer + Dynamic Gate
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        
        # 1. 嚴謹的 GARCH 層
        self.garch_layer = NeuralGARCHLayer(self.pred_len)

        # 2. 修正 4：引入 TSlib 標準的 DataEmbedding (給予模型「時間觀念」和「順序觀念」)
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

        # 3. AI 引擎 (因為輸入已經被 Embedding 轉成 d_model 維度，所以 input_size=d_model)
        self.lstm = nn.LSTM(input_size=self.d_model, hidden_size=self.d_model, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=configs.n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=configs.e_layers)
        
        # 預測未來的殘差
        self.residual_head = nn.Linear(self.d_model, self.pred_len)

        # 4. 動態閘門：改吃 Transformer 整理好的高階特徵來做判斷
        self.gate_net = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2), 
            nn.ReLU(),
            nn.Linear(self.d_model // 2, self.pred_len), 
            nn.Sigmoid()
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # 提取倒數第二欄 (Log_Return) 餵給 GARCH 算統計預測
        returns = x_enc[:, :, -2] 
        garch_future_pred = self.garch_layer(returns) # [Batch, pred_len]
        
        # --- AI 路徑 ---
        # 加上位置編碼和時間特徵！這是 Transformer 復活的關鍵
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        
        lstm_out, _ = self.lstm(enc_out)
        trans_out = self.transformer(lstm_out)
        
        # 取最後一個時間點的深層特徵推論未來殘差
        ai_residual_pred = self.residual_head(trans_out[:, -1, :]) # [Batch, pred_len]
        
        # --- 動態閘門 ---
        gate_weight = self.gate_net(trans_out[:, -1, :]) # [Batch, pred_len]
        
        # --- 最終融合 ---
        # 預測 = 正統 GARCH 統計曲線 + (動態權重 * AI 非線性殘差修正)
        final_prediction = garch_future_pred + (gate_weight * ai_residual_pred)
        
        return final_prediction.unsqueeze(-1)