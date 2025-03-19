import torch
import torch.nn as nn
import torch.nn.functional as F
from models.embed import PositionalEncoding
from torch.nn.modules.transformer import TransformerEncoderLayer


class Transformer(nn.Module):
    def __init__(self, 
                 seq_len= 12,
                 embed_size = 16,
                 nhead = 4,
                 dim_feedforward = 2048,
                 dropout = 0.1,
                 conv1d_emb = False,
                 conv1d_kernel_size = 3,
                 n_classes = 2,
                 batch_size = 1,
                 device = "cuda",
                 details = True):
        super(Transformer, self).__init__()

        # set class parameters
        self.device = device
        self.conv1d_emb = conv1d_emb
        self.conv1d_kernel_size = conv1d_kernel_size
        self.seq_len = seq_len
        self.embed_size = embed_size 
        self.details = details

        # input embedding components
        if conv1d_emb:
            if conv1d_kernel_size%2==0:
                raise Exception("conv1d_kernel_size must be an odd number to preserve dimensions.")
            self.conv1d_padding = conv1d_kernel_size - 1
            self.input_embedding  = nn.Conv1d(1, embed_size, kernel_size= conv1d_kernel_size)
        else: self.input_embedding  = nn.Linear(embed_size, embed_size)

        # positional encoder component
        self.position_encoder = PositionalEncoding(d_model= embed_size, 
                                                   dropout= dropout,
                                                   max_len= seq_len)
        
        # transformer encoder layer
        self.transformer_encoder = TransformerEncoderLayer(
            d_model = embed_size,
            nhead = nhead,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
            batch_first = True
        )

        # classification
        self.conv_out = nn.Conv1d(seq_len, batch_size, kernel_size=3, padding=1)
        self.linear = nn.Linear(embed_size * batch_size, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # src_mask = self._generate_square_subsequent_mask(x.shape[0]).to(self.device)
        # if self.details: print("src_mask shape:", src_mask.shape)
        
        if self.details: print('before input layer: '+ str(x.size()))
        # input embedding
        if self.conv1d_emb: 
            x = F.pad(x, (0, 0, self.conv1d_padding, 0), "constant", -1)
            x = self.input_embedding(x.transpose(1, 2))
            x = x.transpose(1, 2)
        else: 
            x = self.input_embedding(x)

        if self.details: print('before positional encoder: '+ str(x.size()))
        # positional encodding
        x = self.position_encoder(x)

        if self.details: print('before encoder layer: '+ str(x.size()))
        # transformer encoder layer
        x = self.transformer_encoder(x).reshape((-1, self.seq_len*self.embed_size))

        if self.details: print('before classification layer: '+ str(x.size()))
        #classification layer
        x = self.conv_out(x)
        x = self.linear(x)
        # x = self.softmax(x) # Apparently already in CrossEntropyLoss()

        if self.details: print('after classification layer: '+ str(x.size()))
        return x
        
    # Function Copied from PyTorch Library to create upper-triangular source mask
    def _generate_square_subsequent_mask(self, shape):
        return torch.triu(
            torch.full((shape, shape), float('-inf'), dtype=torch.float32, device=self.device),
            diagonal=1,
        )