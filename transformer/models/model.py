import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import PositionalEncoding, ConvLayer, Encoder, EncoderLayer
from torch.nn.modules.transformer import TransformerEncoderLayer


class Transformer(nn.Module):
    def __init__(self, 
                 seq_len= 20,
                 embed_size = 12,
                 nhead = 4,
                 dim_feedforward = 2048,
                 dropout = 0.1,
                 conv1d_kernel_size = 3,
                 d_model_class = 512,
                 n_hidden_layers = 3,
                 n_classes = 2,
                 device = "cuda",
                 details = True):
        super(Transformer, self).__init__()

        # set class parameters
        self.device = device
        self.conv1d_kernel_size = conv1d_kernel_size
        self.seq_len = seq_len
        self.embed_size = embed_size 
        self.details = details

        # input embedding components
        self.input_embedding  = nn.Linear(embed_size, embed_size)

        # positional encoder component
        self.position_encoder = PositionalEncoding(d_model= embed_size, 
                                                   dropout= dropout,
                                                   max_len= seq_len,
                                                   details= details)
        
        # transformer encoder layer
        self.transformer_encoder = TransformerEncoderLayer(
            d_model = embed_size,
            nhead = nhead,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
            batch_first = True
        )

        # classification
        self.conv_out = ConvLayer(c_in= embed_size, n_layers= 2)
        self.input_layer = nn.Linear(embed_size * seq_len, d_model_class)

        hidden_dims = [d_model_class // (2 ** i) for i in range(1, n_hidden_layers + 1)]  # [d/2, d/4, d/8, d/16]
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip([d_model_class] + hidden_dims[:-1], hidden_dims)]
        )
        self.output_layer = nn.Linear(hidden_dims[-1], n_classes)

        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        if self.details: print('Input shape:', x.shape)
        src_mask = self._generate_square_subsequent_mask().to(device= self.device)

        # input embedding
        x = self.input_embedding(x)  # Shape: (batch_size, seq_len, embed_size)
        if self.details: print('After input embedding:', x.shape)

        # positional encoding
        x = self.position_encoder(x)  # Shape: (batch_size, seq_len, embed_size)
        if self.details: print('After positional encoding:', x.shape)

        # transformer encoder layer
        x = self.transformer_encoder(x, src_mask= src_mask)  # Shape: (batch_size, seq_len, embed_size)
        if self.details: print('After transformer encoder:', x.shape)

        # classification
        x = self.conv_out(x)  # Shape: (batch_size, c_out, seq_len)
        if self.details: print('After convolution layer:', x.shape)

        x = x.flatten(1)
        if self.details: print('After flatten:', x.shape)

        x = self.input_layer(x)  # Shape: (batch_size, d_model_class)
        if self.details: print('After input layer:', x.shape)

        for layer in self.hidden_layers:
            x = layer(x)
            x = self.relu(x)
            x = self.dropout(x)
            if self.details: print('After hidden layer:', x.shape)

        x = self.output_layer(x)
        if self.details: print('After classification:', x.shape)
        # x = self.softmax(x)  # Shape: (batch_size, n_classes)
        return x
        
    # Function Copied from PyTorch Library to create upper-triangular source mask
    def _generate_square_subsequent_mask(self):
        return torch.triu(
            torch.full((self.seq_len, self.seq_len), float('-inf'), dtype=torch.float32, device=self.device),
            diagonal=1,
        )
