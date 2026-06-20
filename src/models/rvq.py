from __future__ import annotations

import torch
from torch import nn

from src.nn.backbone import Decoder, Encoder
from src.nn.quantizers import ResidualVQ


class RVQVAEModel(nn.Module):
    """RVQ-VAE: VQ-VAE con cuantizacion vectorial residual en cascada.

    En lugar de un solo codebook, cuantiza el residuo de forma iterativa a
    traves de `num_quantizers` codebooks: cada etapa codifica lo que la
    anterior no pudo. Aproxima un latente continuo con multiples codigos
    discretos, situandose entre el VQ-VAE (discreto puro) y el autoencoder
    (continuo) en la tension continuo vs. discreto.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        num_codes: int = 512,
        num_quantizers: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(latent_dim, dropout=dropout)
        self.quantizer = ResidualVQ(num_quantizers=num_quantizers, num_codes=num_codes, code_dim=latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retorna (reconstruccion, latente_cuantizado (B,C,8,8), perdida_vq).

        El latente cuantizado es la suma de los codigos de todas las etapas;
        la perdida_vq agrega commitment de cada etapa.
        """
        z = self.encoder(x)
        q, vq_loss, _idx = self.quantizer(z)
        recon = self.decoder(q)
        return recon, q, vq_loss

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Devuelve el latente cuantizado congelado (B, latent_dim) por pooling."""
        z = self.encoder(x)
        q, _loss, _idx = self.quantizer(z)
        return q.mean(dim=(2, 3))
