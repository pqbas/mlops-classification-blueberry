from __future__ import annotations

import torch
from torch import nn

from src.nn.backbone import Decoder, Encoder


class AutoencoderModel(nn.Module):
    """Autoencoder vanilla: Encoder + Decoder, latente continuo (sin KL, no VAE).

    Se entrena con perdida de reconstruccion (MSE). El latente es el mapa
    espacial continuo (B, latent_dim, 8, 8); el embedding del probe lo reduce a
    vector por pooling. Hipotesis H1: este espacio continuo preserva mejor la
    trayectoria de madurez que la cuantizacion discreta del VQ-VAE.
    """

    def __init__(self, latent_dim: int = 128, dropout: float = 0.0) -> None:
        super().__init__()
        self.encoder = Encoder(latent_dim, dropout=dropout)
        self.decoder = Decoder(latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Retorna (reconstruccion, latente espacial (B, latent_dim, 8, 8))."""
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Devuelve el latente continuo congelado (B, latent_dim) por pooling."""
        return self.encoder.embed(x)
