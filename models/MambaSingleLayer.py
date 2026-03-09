import torch.nn as nn

from layers.Embed import PositionalEmbedding
from layers.MambaBlock import Mamba_TimeVariant


class TokenEmbedding_cls(nn.Module):
    """TokenEmbedding with configurable kernel size(`d_kernel`).
    """
    def __init__(self, c_in, d_model, d_kernel=3):
        super().__init__()
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=d_kernel, padding='same', padding_mode='replicate', bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding_cls(nn.Module):
    """DataEmbedding with configurable kernel size(`d_kernel`) and sequence length(`seq_len`).

    To solve the warning for EigenWorms dataset (seq_len=17984) while keeping consistency comparing with other models, we set max_len=max(5000, seq_len)."""
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1, d_kernel=3, seq_len=5000):
        super(DataEmbedding_cls, self).__init__()
        self.value_embedding = TokenEmbedding_cls(c_in=c_in, d_model=d_model, d_kernel=d_kernel)
        self.position_embedding = PositionalEmbedding(d_model=d_model, max_len=max(5000, seq_len))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)



class Model(nn.Module):
    """MambaSL: Exploring Single-Layer Mamba for Time Series Classification
    
    - Paper Link: https://openreview.net/pdf?id=YDl4vqQqGP
    - Original Repo: https://github.com/yoom618/MambaSL. removed all extra codes for ablation study and further analysis.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out
        self.dropout = configs.dropout
        self.num_kernels = configs.num_kernels
        
        self.mamba = nn.Sequential(
            Mamba_TimeVariant(
                d_model = configs.d_model,
                d_state = configs.d_ff,
                d_conv = configs.d_conv,
                expand = configs.expand,
                timevariant_dt = bool(configs.tv_dt),    # only available in Mamba_TimeVariant
                timevariant_B = bool(configs.tv_B),      # only available in Mamba_TimeVariant
                timevariant_C = bool(configs.tv_C),      # only available in Mamba_TimeVariant
                use_D = bool(configs.use_D),             # use D(skip connection) or not
                device = configs.device,
            ),
            nn.LayerNorm(configs.d_model),
            nn.SiLU(),  # simply choose the same activation fn as Mamba Block
        )
        
        if self.task_name in ['classification']:  # one class per one sequence sample
            
            self.embedding = DataEmbedding_cls(configs.enc_in, configs.d_model,
                                        configs.embed, configs.freq, configs.dropout, 
                                        configs.num_kernels, configs.seq_len)
            
            self.out_layer = nn.Sequential(
                nn.Dropout(configs.dropout),
                nn.Linear(configs.d_model, configs.num_class, bias=False)
            )
            nn.init.xavier_uniform_(self.out_layer[1].weight)
            
            self.attn_weight = nn.Sequential(
                nn.Linear(configs.d_model, configs.n_heads, bias=True),
                nn.AdaptiveMaxPool1d(1),
                nn.Softmax(dim=1),
            )
            for m in self.attn_weight.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    if m.bias is not None: m.bias.data.fill_(1.0)
            
        else:
            raise ValueError(f"task_name: {configs.task_name} is not valid.")



    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, mask=None):
        
        if self.task_name in ['classification']:

            mamba_in = self.embedding(x_enc)  # (B, L_in, D)
            mamba_out = self.mamba(mamba_in)  # (B, L_in, D)
            
            ### [proposed] use the gating value to make the final prediction
            logit_out = self.out_layer(mamba_out)  # (B, L_in, D) -> (B, L_in, C_out)
            logit_out *= x_mark_enc.unsqueeze(2)  # (B, L_in, C_out)  # Mask out the padded sequence for variable length data (e.g. JapaneseVowels)
        
            ### Compute attention weights for weighted sum of logit_out
            w_out = self.attn_weight(mamba_out)  # (B, L_in, D) -> (B, L_in, n_head) -> (B, L_in, 1)

            ### calculate the weighted average of the hidden states to make the final prediction
            out = logit_out * w_out  # (B, L_in, C_out)
            out = out.sum(1)  # (B, C_out)

            return out

        
        else:
            raise ValueError(f"task_name: {self.task_name} is not valid.")