import torch, math
import torch.nn as nn
import torch.nn.functional as F

# Positional Encoding - https://pytorch.org/tutorials/beginner/transformer_tutorial.html

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, details= False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)  # Shape: (max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.details = details

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        if self.details: print('before positional encoding: ', x.shape)
        x = x + self.pe[:x.size(1)]  # Add positional encodings
        return self.dropout(x)
    
class ConvLayer(nn.Module):
    def __init__(self, c_in, n_layers):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.n_layers = n_layers
        layers = []
        for _ in range(n_layers):
            layers.extend([
                nn.Conv1d(in_channels=c_in, out_channels=c_in, kernel_size=3, padding=padding, padding_mode='circular'),
                nn.BatchNorm1d(c_in),
                nn.ELU(),
                nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
            ])
        
        self.ConvNet = nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch, channels, seq_len]
        x = self.ConvNet(x)
        return x.permute(0, 2, 1)  # [batch, seq_len, channels]
