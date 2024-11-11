import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        # Create a long enough P matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x
    

class PeriodicityReshape(nn.Module):
    def __init__(self, main_cycle):
        super(PeriodicityReshape, self).__init__()
        if main_cycle < 1:
            raise ValueError(f'Invalid main_cycle: {main_cycle}. Must be >= 1.')
        self.main_cycle = main_cycle
        
    def __assert_seq_len(self, x):
        _, n_steps, _ = x.shape
        seq_too_long = (n_steps % self.main_cycle) # is > 0 if n_steps is not a multiple of main_cycle -> True
        if seq_too_long:
            raise ValueError(f'''Number of steps {n_steps} is not a multiple of the main cycle ({n_steps}%{self.main_cycle}={n_steps%self.main_cycle}).
                             Suggested: Fill the sequence with zeros at the end to make it a multiple of the main cycle.''') 
            
    def apply(self, x, batch_size, n_features):
        self.__assert_seq_len(x)
        x = x.reshape(batch_size, -1, self.main_cycle, n_features).permute(0, 3, 1, 2)
        x = x.reshape(batch_size, -1, self.main_cycle).permute(0, 2, 1)
        return x

    def revert(self,x, batch_size, n_features):
        x = x.permute(0, 2, 1).reshape(batch_size, n_features, -1, self.main_cycle)
        x = x.permute(0, 2, 3, 1).reshape(batch_size, -1, n_features)
        return x

    def forward(self, x, n_features, direction):
        batch_size = x.shape[0]
        if direction == 'apply':
            return self.apply(x, batch_size, n_features)
        elif direction == 'revert':
            return self.revert(x, batch_size, n_features)
        else:
            raise ValueError(f'Invalid direction: {direction}. Use "apply" or "revert".')