from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class VectorQuantizer(nn.Module):
    """Cuantizacion vectorial: mapea cada celda del mapa latente al codigo mas
    cercano de un codebook aprendido (un solo nivel).

    Entrada (B, C, H, W) con C = code_dim; retorna el latente cuantizado con
    straight-through estimator, la perdida VQ (codebook + commitment) y los
    indices de codigo. Compartido por VQ-VAE.
    """

    def __init__(self, num_codes: int = 512, code_dim: int = 128, commitment: float = 0.25) -> None:
        super().__init__()
        self.code_dim = code_dim
        self.commitment = commitment
        self.codebook = nn.Embedding(num_codes, code_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retorna (z_cuantizado (B,C,H,W), perdida_vq, indices (B,H,W))."""
        b, c, h, w = z.shape
        z_perm = z.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        flat = z_perm.view(-1, c)  # (N, C)

        # Distancia L2 a cada codigo; el mas cercano define el indice.
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(1)
        )
        idx = dist.argmin(1)
        q = self.codebook(idx).view(z_perm.shape)  # (B, H, W, C)

        codebook_loss = F.mse_loss(q, z_perm.detach())       # acerca el codebook al encoder
        commit_loss = F.mse_loss(q.detach(), z_perm)         # acerca el encoder al codebook
        loss = codebook_loss + self.commitment * commit_loss

        q = z_perm + (q - z_perm).detach()  # straight-through: gradiente pasa directo
        q = q.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        return q, loss, idx.view(b, h, w)


class ResidualVQ(nn.Module):
    """Cuantizacion vectorial residual: aplica `num_quantizers` codebooks en
    cascada, cada uno cuantiza el residuo del anterior.

    El latente cuantizado es la suma de los codigos de todas las etapas; la
    perdida agrega el VQ de cada nivel. Compartido por RVQ-VAE.
    """

    def __init__(self, num_quantizers: int = 4, num_codes: int = 512, code_dim: int = 128, commitment: float = 0.25) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            VectorQuantizer(num_codes, code_dim, commitment) for _ in range(num_quantizers)
        )

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Retorna (z_cuantizado (B,C,H,W), perdida_vq, indices_por_nivel)."""
        residual = z
        quantized = torch.zeros_like(z)
        total_loss = z.new_zeros(())
        indices: list[torch.Tensor] = []
        for vq in self.layers:
            q, loss, idx = vq(residual)
            quantized = quantized + q
            residual = residual - q.detach()  # el siguiente nivel ve el residuo puro
            total_loss = total_loss + loss
            indices.append(idx)
        return quantized, total_loss, indices
