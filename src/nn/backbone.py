from __future__ import annotations

import torch
from torch import nn

from src.nn.layers import ConvBNAct, ResBlock, UpBlock


class Encoder(nn.Module):
    """CNN encoder pequeno compartido por los cinco paradigmas (diseno A).

    Entrenado desde cero, sin preentrenamiento. ResNet reducido con SiLU. Cada
    etapa baja resolucion con un conv stride-2 y refina con un ResBlock. Mapea
    (B, 3, 128, 128) a un mapa espacial (B, latent_dim, 8, 8). `embed` aplica
    global average pooling para obtener el vector (B, latent_dim) del probe.
    Mantener identica capacidad para todos los metodos asegura comparacion justa.
    """

    def __init__(self, latent_dim: int = 128, dropout: float = 0.0) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.stage1 = nn.Sequential(ConvBNAct(3, 32, stride=2), ResBlock(32))                  # 64x64
        self.stage2 = nn.Sequential(ConvBNAct(32, 64, stride=2), ResBlock(64))                 # 32x32
        self.stage3 = nn.Sequential(ConvBNAct(64, 128, stride=2), ResBlock(128))               # 16x16
        self.stage4 = nn.Sequential(ConvBNAct(128, latent_dim, stride=2), ResBlock(latent_dim))  # 8x8
        self.drop = nn.Dropout2d(dropout)  # spatial dropout sobre el mapa latente

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Devuelve el mapa espacial (B, latent_dim, 8, 8)."""
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.drop(x)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Vector (B, latent_dim) por global average pooling del mapa 8x8."""
        feat = self.forward(x)
        return feat.mean(dim=(2, 3))


class Decoder(nn.Module):
    """Decoder espejo del Encoder. Mapea (B, latent_dim, 8, 8) a (B, 3, 128, 128).

    Usado por autoencoder, VQ-VAE y RVQ-VAE para reconstruir y para latent
    traversals. Upsamplea con UpBlock (ConvTranspose stride-2 + ResBlock).
    """

    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.up1 = UpBlock(latent_dim, 128)   # 8 -> 16
        self.up2 = UpBlock(128, 64)           # 16 -> 32
        self.up3 = UpBlock(64, 32)            # 32 -> 64
        self.up4 = UpBlock(32, 16)            # 64 -> 128
        self.head = nn.Conv2d(16, 3, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruye la imagen en [0, 1] desde el mapa latente."""
        z = self.up1(z)
        z = self.up2(z)
        z = self.up3(z)
        z = self.up4(z)
        return torch.sigmoid(self.head(z))
