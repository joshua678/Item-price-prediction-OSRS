import torch
from torch import nn


class LSTMQuantileRegressor(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_items: int,
        n_quantiles: int,
        n_horizons: int = 1,
        id_emb_dim: int = 4,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.id_emb_dim = int(id_emb_dim)
        self.num_layers = int(num_layers)
        self.n_horizons = int(n_horizons)
        self.n_quantiles = int(n_quantiles)
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1. Got: {self.num_layers}")
        if self.n_horizons < 1:
            raise ValueError(f"n_horizons must be >= 1. Got: {self.n_horizons}")

        if self.id_emb_dim > 0:
            self.id_emb = nn.Embedding(n_items, self.id_emb_dim)
        else:
            self.id_emb = None  # type: ignore[assignment]

        self.lstm = nn.LSTM(
            input_size=n_features + (self.id_emb_dim if self.id_emb_dim > 0 else 0),
            hidden_size=hidden_size,
            num_layers=self.num_layers,
            dropout=dropout if self.num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )

        self.dropout = nn.Dropout(dropout)

        out_size = self.n_horizons * self.n_quantiles
        self.head_high = nn.Linear(hidden_size * self.num_layers, out_size)
        self.head_low = nn.Linear(hidden_size * self.num_layers, out_size)

    def forward(self, x: torch.Tensor, item_id: torch.Tensor):
        B, T, _ = x.shape

        if self.id_emb is not None:
            id_vec = self.id_emb(item_id).unsqueeze(1).expand(B, T, -1)
            x = torch.cat([x, id_vec], dim=-1)

        _, (h, _) = self.lstm(x)  # h: (num_layers, B, hidden_size)
        h_top = self.dropout(h.transpose(0, 1).reshape(B, -1))
        qH = self.head_high(h_top).reshape(B, self.n_horizons, self.n_quantiles)  # (B, H, Q)
        qL = self.head_low(h_top).reshape(B, self.n_horizons, self.n_quantiles)   # (B, H, Q)
        return qH, qL
