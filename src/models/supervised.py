from __future__ import annotations

import torch
from torch import nn

from src.nn.backbone import Encoder


class SupervisedModel(nn.Module):
    """Baseline: Encoder compartido + cabeza lineal de clasificacion.

    Se entrena con cross-entropy sobre las etiquetas. El embedding es la salida
    del encoder (vector pooled), que se extrae congelado para la sonda downstream.
    """

    def __init__(self, latent_dim: int = 128, num_classes: int = 7, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = Encoder(latent_dim, dropout=dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(latent_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna los logits (B, num_classes)."""
        return self.head(self.encoder.embed(x))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Devuelve el embedding congelado (B, latent_dim)."""
        return self.encoder.embed(x)
