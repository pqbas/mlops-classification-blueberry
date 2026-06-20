from __future__ import annotations

import torch
from torch import nn

from src.nn.backbone import Encoder


class MetricModel(nn.Module):
    """Deep metric learning ordinal: solo Encoder, sin decoder ni cabeza.

    El embedding (B, latent_dim) se organiza para que la distancia euclidiana
    refleje la diferencia de madurez entre muestras. Lo comparten las dos
    variantes de loss (triplet de margen ordinal y distancia continua); la loss
    vive en train.py porque opera sobre el batch completo, no por muestra.
    """

    def __init__(self, latent_dim: int = 128, dropout: float = 0.0) -> None:
        super().__init__()
        self.encoder = Encoder(latent_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Devuelve el embedding (B, latent_dim), identico a embed."""
        return self.encoder.embed(x)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Embedding congelado (B, latent_dim) para el probe."""
        return self.encoder.embed(x)
