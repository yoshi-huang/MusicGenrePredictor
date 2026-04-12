import math
import torch
import torch.nn as nn

class PositionEncoder(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim=128, embed_dim=128, seq_len=128,
                 num_heads=4, num_layers=6, FFN_dim=256, dropout=0):
        super().__init__()
        self.EmbeddingBlock = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.PositionEncoder = PositionEncoder(embed_dim, seq_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=FFN_dim,
            dropout=dropout, activation="relu", batch_first=True,
        )
        self.TransformerEncoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.Classifier = nn.Sequential(
            nn.BatchNorm1d(embed_dim),
            nn.Linear(embed_dim, 10),
        )

    def forward(self, x):
        x = self.EmbeddingBlock(x)
        x = self.PositionEncoder(x)
        x = self.TransformerEncoder(x)
        x = x.mean(dim=1)
        return self.Classifier(x)
