from __future__ import annotations

import torch
from torch import nn

from src.nn.backbone import Decoder, Encoder
from src.nn.quantizers import VectorQuantizer


class VQVAEModel(nn.Module):
    """VQ-VAE: Encoder + cuantizacion vectorial + Decoder, latente discreto.

    Cada celda del mapa (B, latent_dim, 8, 8) se mapea al codigo mas cercano de
    un codebook aprendido. Se entrena con reconstruccion (MSE) + perdida VQ
    (codebook + commitment). El embedding del probe es el latente cuantizado
    reducido a vector por pooling, util para contrastar geometria discreta vs
    continua del autoencoder.
    """

    def __init__(self, latent_dim: int = 128, num_codes: int = 512, dropout: float = 0.0) -> None:
        super().__init__()
        self.encoder = Encoder(latent_dim, dropout=dropout)
        self.quantizer = VectorQuantizer(num_codes=num_codes, code_dim=latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retorna (reconstruccion, latente_cuantizado (B,C,8,8), perdida_vq)."""
        z = self.encoder(x)
        q, vq_loss, _idx = self.quantizer(z)
        recon = self.decoder(q)
        return recon, q, vq_loss

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Devuelve el latente cuantizado congelado (B, latent_dim) por pooling."""
        z = self.encoder(x)
        q, _loss, _idx = self.quantizer(z)
        return q.mean(dim=(2, 3))
