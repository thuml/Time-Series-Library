import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn

        
class CyclicEncoderLayer(nn.Module):
    def __init__(self, attention_var, attention_cycle, d_model, d_temp, num_cycles, n_features, d_ff=None, dropout=0.1, activation="relu", full_mlp=False):
        super(CyclicEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention_var = attention_var
        self.attention_cycle = attention_cycle
        self.N = num_cycles
        self.full_mlp = full_mlp
        
        if full_mlp:
            
            self.conv1 = nn.Conv1d(in_channels=d_model*self.N, out_channels=d_ff, kernel_size=1)
            self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model*self.N, kernel_size=1)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            
        else:
            
            self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
            self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.dim_reduction = nn.Linear(n_features*d_model, d_temp)
        self.post_attention_proj = nn.Linear(d_temp, n_features*d_model)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        
        # (batch_size, num_cycles * num_variates, d_model)
        B, NC, D = x.size()
        C = int(NC / self.N)
        
        # Reshape for attending over num_cycles independent of variates
        x_var = x.view(B, C, self.N, D).permute(0, 2, 1, 3).reshape(B * self.N, C, D)
        
        new_x, attn_var = self.attention_var(
            x_var, x_var, x_var,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        
        # Reshape for attending over num_cycles and move variates to the last dimension 
        new_x = new_x.reshape(B, self.N, C*D)
        new_x = self.dim_reduction(new_x.reshape(B, self.N, C*D))
        
        new_x, attn_cycle = self.attention_cycle(
            new_x, new_x, new_x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        new_x = self.post_attention_proj(new_x)
        # Reshape back to the original shape (batch_size, num_cycles * num_variates, d_model)
        new_x = new_x.reshape(B, self.N, C, D).permute(0, 2, 1, 3).reshape(B, self.N*C, D)
        
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)        
        
        if self.full_mlp:
            
            # Reshape to go from (batch_size, num_variates, num_cycles * d_model) to (batch_size, num_variates, num_cycles * d_model)
            y = y.reshape(B, C, self.N, D).reshape(B, C, D*self.N)

            y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
            y = self.dropout(self.conv2(y).transpose(-1, 1))

            # Reshape to go from (batch_size, num_variates, num_cycles * d_model) to original (batch_size, num_variates, num_cycles * d_model)
            y = y.reshape(B, C, self.N, D).permute(0, 2, 1, 3).reshape(B, self.N, C*D).reshape(B, self.N, C, D).permute(0, 2, 1, 3).reshape(B, self.N * C, D)
        
        else:
            
            y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
            y = self.dropout(self.conv2(y).transpose(-1, 1))
        
            
        return self.norm2(x + y), (attn_var, attn_cycle)
    
    
class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
