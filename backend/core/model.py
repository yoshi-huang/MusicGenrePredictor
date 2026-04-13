# -----------------------------------------------------------------------------
# Transformer model definition (shared by GUI and DCBOT)
# -----------------------------------------------------------------------------

from math import log
import os
import torch
import torch.nn as nn


class PositionEncoder(nn.Module):

    def __init__(self, d_model, seq_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-log(10_000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class TransformerEncoder(nn.Module):

    def __init__(self, input_dim, embed_dim, seq_len,
                 num_heads, num_layers, FFN_dim, dropout):
        super().__init__()

        self.EmbeddingBlock = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.PositionEncoder = PositionEncoder(embed_dim, seq_len, dropout)
        self.EncoderLayer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=FFN_dim, dropout=dropout,
            activation='relu', batch_first=True
        )
        self.TransformerEncoder = nn.TransformerEncoder(self.EncoderLayer, num_layers=num_layers)
        self.Classifier = nn.Sequential(
            nn.BatchNorm1d(embed_dim),
            nn.Linear(embed_dim, 10)
        )

    def forward(self, x):
        x = self.EmbeddingBlock(x)
        x = self.PositionEncoder(x)
        x = self.TransformerEncoder(x)
        x = x.mean(dim=1)
        return self.Classifier(x)


def load_model(weights_path: str = None) -> TransformerEncoder:
    """Load and return an eval-mode TransformerEncoder from a .pth file."""
    from core.config import paths, model_cfg
    if weights_path is None:
        weights_path = paths["model_weights"]
    # Resolve relative paths against backend root (where transformer_parms.pth lives)
    if not os.path.isabs(weights_path):
        _backend_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        weights_path = os.path.join(_backend_root, weights_path)
    m = model_cfg
    model = TransformerEncoder(
        m["input_dim"], m["embed_dim"], m["seq_len"],
        m["num_heads"], m["num_layers"], m["ffn_dim"], m["dropout"]
    ).to("cpu")
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()
    return model
