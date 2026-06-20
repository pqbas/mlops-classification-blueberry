from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F

from src.nn.backbone import Encoder


class ConvPredictor(nn.Module):
    """Predictor totalmente convolucional: 3 conv depthwise-separable 3x3 + BN +
    ReLU, mantiene la grilla 8x8 y `dim` canales. Sigue a CNN-JEPA: separable
    en vez de conv densa para no inflar parametros con canales altos."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(3):
            layers += [
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),  # depthwise
                nn.Conv2d(dim, dim, kernel_size=1, bias=False),                          # pointwise
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class JEPAModel(nn.Module):
    """CNN-JEPA: context encoder (enmascarado disperso) + target encoder EMA +
    predictor convolucional. Predice latentes objetivo en las celdas
    enmascaradas de la grilla 8x8. embed() usa el context encoder sin mascara
    (lo que guarda el probe como encoder.pt)."""

    def __init__(
        self,
        latent_dim: int = 128,
        dropout: float = 0.0,
        momentum: float = 0.996,
        num_blocks: int = 4,
        mask_ratio: float = 0.75,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(latent_dim, dropout=dropout)       # context/student
        self.target = copy.deepcopy(self.encoder)                 # EMA, sin gradiente
        self.target.requires_grad_(False)
        self.predictor = ConvPredictor(latent_dim)
        self.momentum = momentum
        self.num_blocks = num_blocks
        self.mask_ratio = mask_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna la loss L2 entre prediccion y target en las celdas enmascaradas."""
        b = x.size(0)
        with torch.no_grad():
            tgt = self.target(x)                                  # (B, C, 8, 8) full forward
            # LayerNorm sobre canales (estandar I-JEPA): vuelve la escala del
            # target estacionaria, sin esto la loss crece con el EMA y el
            # checkpoint por val_loss premia la epoca 0 casi sin entrenar.
            tgt = F.layer_norm(tgt.permute(0, 2, 3, 1), (tgt.size(1),)).permute(0, 3, 1, 2)
        grid = tgt.shape[-1]
        mask = _sample_block_mask(b, grid, self.num_blocks, self.mask_ratio, x.device)  # (B,8,8) bool
        ctx = self._encode_masked(x, mask)                        # context con celdas enmascaradas en cero
        pred = self.predictor(ctx)                                # (B, C, 8, 8)

        m = mask.unsqueeze(1)                                     # (B, 1, 8, 8)
        diff = (pred - tgt) ** 2
        masked = diff * m
        return masked.sum() / m.sum().clamp(min=1.0) / tgt.size(1)

    @torch.no_grad()
    def ema_update(self) -> None:
        """target <- m*target + (1-m)*encoder, para parametros y buffers."""
        for tp, ep in zip(self.target.parameters(), self.encoder.parameters()):
            tp.mul_(self.momentum).add_(ep, alpha=1.0 - self.momentum)
        for tb, eb in zip(self.target.buffers(), self.encoder.buffers()):
            tb.copy_(eb)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Embedding congelado (B, latent_dim) del context encoder por GAP."""
        return self.encoder.embed(x)

    def _encode_masked(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Corre las 4 etapas del context encoder y pone a cero la salida de cada
        etapa en las celdas enmascaradas (mascara 8x8 upscaleada a cada
        resolucion). Aproxima la conv dispersa de CNN-JEPA sin libreria especial."""
        visible = (~mask).float().unsqueeze(1)                    # (B, 1, 8, 8), 1 = visible
        enc = self.encoder
        h = enc.stage1(x)
        h = h * _upscale(visible, h.shape[-1])
        h = enc.stage2(h)
        h = h * _upscale(visible, h.shape[-1])
        h = enc.stage3(h)
        h = h * _upscale(visible, h.shape[-1])
        h = enc.stage4(h)
        h = h * _upscale(visible, h.shape[-1])
        return enc.drop(h)


def _upscale(mask: torch.Tensor, size: int) -> torch.Tensor:
    """Lleva una mascara (B,1,g,g) a (B,1,size,size) por vecino mas cercano."""
    if mask.shape[-1] == size:
        return mask
    return F.interpolate(mask, size=(size, size), mode="nearest")


def _sample_block_mask(batch: int, grid: int, num_blocks: int, mask_ratio: float, device) -> torch.Tensor:
    """(B, grid, grid) bool, True = celda enmascarada. Muestrea bloques
    rectangulares por imagen hasta cubrir ~mask_ratio de la grilla; garantiza al
    menos una celda enmascarada y una visible por imagen."""
    target_cells = max(1, int(round(mask_ratio * grid * grid)))
    mask = torch.zeros(batch, grid, grid, dtype=torch.bool, device=device)
    for i in range(batch):
        attempts = 0
        while mask[i].sum() < target_cells and attempts < num_blocks * 4:
            attempts += 1
            bh = int(torch.randint(1, grid + 1, (1,)).item())
            bw = int(torch.randint(1, grid + 1, (1,)).item())
            top = int(torch.randint(0, grid - bh + 1, (1,)).item())
            left = int(torch.randint(0, grid - bw + 1, (1,)).item())
            mask[i, top:top + bh, left:left + bw] = True
        if mask[i].all():                                        # deja al menos una visible
            mask[i, 0, 0] = False
        if not mask[i].any():                                    # garantiza al menos una enmascarada
            mask[i, 0, 0] = True
    return mask
