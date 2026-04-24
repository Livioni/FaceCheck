from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass(frozen=True)
class LandmarkTokenConfig:
    in_dim: int = 27
    embed_dim: int = 768
    hidden_dim: int = 256
    dropout: float = 0.0


class LandmarkMLP(nn.Module):
    def __init__(self, cfg: LandmarkTokenConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.in_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(p=cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x.unsqueeze(1)


class BinaryHead(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.drop = nn.Dropout(p=dropout)
        self.fc = nn.Linear(embed_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(x))

