from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from src.config import Config
from src.dataset import get_dataloaders


def run(config: Config, output_dir: Path) -> None:
    """Entrena el modelo del paradigma indicado en `config.method` y guarda el
    encoder congelado en `output_dir`. Para type='train_probe', encadena la
    sonda downstream tras el entrenamiento.

    Dispatch por metodo: supervised | autoencoder | vqvae | rvq | jepa.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.params.seed)
    device = _device(config.device)

    train_loader, val_loader, _test_loader, class_names = get_dataloaders(
        config.dataset,
        batch_size=config.params.batch_size,
        imgsz=config.params.imgsz,
        seed=config.params.seed,
        workers=config.params.workers,
        augmentation=config.augmentation,
    )

    model = build_model(config, num_classes=len(class_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.params.lr, weight_decay=config.params.weight_decay)

    # Mapeo label -> rango de madurez, solo lo usan las losses metric.
    rank_map = _ripeness_rank_map(class_names, device)

    # 100 epocas fijas (sin early stopping). Se conserva el checkpoint de la
    # mejor metrica de val: accuracy en supervised, val_loss en el resto.
    by_acc = config.method == "supervised"
    best_score = -float("inf") if by_acc else float("inf")
    best_epoch = -1
    history: list[dict] = []
    for epoch in range(config.params.epochs):
        train_loss = _run_epoch(model, train_loader, config.method, device, optimizer, rank_map)
        val_loss, val_acc = _evaluate(model, val_loader, config.method, device, rank_map)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc})
        print(f"[train] epoch {epoch} method={config.method} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc}")

        if by_acc:
            assert val_acc is not None
            score, improved = val_acc, val_acc > best_score
        else:
            score, improved = val_loss, val_loss < best_score
        if improved:
            best_score = score
            best_epoch = epoch
            torch.save(model.encoder.state_dict(), output_dir / "encoder.pt")

    _write_metrics(output_dir, config, history, best_score, best_epoch, by_acc, class_names)

    if config.type == "train_probe":
        from src import probe

        probe.run(config, output_dir)


def build_model(config: Config, num_classes: int = 7) -> nn.Module:
    """Instancia el modelo correspondiente a `config.method`."""
    latent_dim = config.params.latent_dim
    dropout = config.params.dropout
    if config.method == "supervised":
        from src.models.supervised import SupervisedModel

        return SupervisedModel(latent_dim, num_classes=num_classes, dropout=dropout)
    if config.method == "autoencoder":
        from src.models.autoencoder import AutoencoderModel

        return AutoencoderModel(latent_dim, dropout=dropout)
    if config.method == "vqvae":
        from src.models.vqvae import VQVAEModel

        return VQVAEModel(latent_dim, dropout=dropout)
    if config.method == "rvq":
        from src.models.rvq import RVQVAEModel

        return RVQVAEModel(latent_dim, dropout=dropout)
    if config.method == "jepa":
        from src.models.jepa import JEPAModel

        return JEPAModel(latent_dim, dropout=dropout)
    if config.method in ("metric_triplet", "metric_continuous"):
        from src.models.metric import MetricModel

        return MetricModel(latent_dim, dropout=dropout)
    raise NotImplementedError(f"Metodo '{config.method}' aun no implementado")


def _run_epoch(model: nn.Module, loader, method: str, device, optimizer, rank_map) -> float:
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = _loss(model, x, y, method, rank_map)
        loss.backward()
        optimizer.step()
        if hasattr(model, "ema_update"):
            model.ema_update()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


@torch.no_grad()
def _evaluate(model: nn.Module, loader, method: str, device, rank_map) -> tuple[float, float | None]:
    model.eval()
    total, n, correct = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = _loss(model, x, y, method, rank_map)
        total += loss.item() * x.size(0)
        n += x.size(0)
        if method == "supervised":
            correct += (model(x).argmax(dim=1) == y).sum().item()
    val_acc = correct / n if method == "supervised" and n else None
    return total / max(n, 1), val_acc


def _loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor, method: str, rank_map: torch.Tensor) -> torch.Tensor:
    if method == "supervised":
        return nn.functional.cross_entropy(model(x), y)
    if method == "autoencoder":
        recon, _ = model(x)
        return nn.functional.mse_loss(recon, x)
    if method in ("vqvae", "rvq"):
        recon, _q, vq_loss = model(x)
        return nn.functional.mse_loss(recon, x) + vq_loss
    if method == "jepa":
        return model(x)
    if method in ("metric_triplet", "metric_continuous"):
        emb = model(x)
        ranks = rank_map[y].float()
        if method == "metric_triplet":
            return _ordinal_triplet_loss(emb, ranks)
        return _continuous_metric_loss(emb, ranks)
    raise NotImplementedError(f"Loss para metodo '{method}' aun no implementado")


def _ordinal_triplet_loss(emb: torch.Tensor, ranks: torch.Tensor, margin_scale: float = 1.0) -> torch.Tensor:
    """Triplet de margen ordinal sobre todos los tripletes validos del batch.

    Para cada ancla i y par (j, k) con |rank_i - rank_j| < |rank_i - rank_k|,
    exige dist(i, j) + margen < dist(i, k), con margen proporcional a la
    diferencia de rango entre el positivo y el negativo. Promedia los tripletes
    activos. Distancia euclidiana, sin normalizar.
    """
    dist = torch.cdist(emb, emb)                       # (B, B) distancia entre pares
    rdist = (ranks[:, None] - ranks[None, :]).abs()    # (B, B) distancia de rango
    d_pos = dist[:, :, None]                            # dist(i, j)
    d_neg = dist[:, None, :]                            # dist(i, k)
    margin = margin_scale * (rdist[:, None, :] - rdist[:, :, None])  # rank_k - rank_j
    valid = (rdist[:, :, None] < rdist[:, None, :]).float()          # j mas cercano que k
    losses = torch.relu(d_pos - d_neg + margin) * valid
    return losses.sum() / valid.sum().clamp(min=1.0)


def _continuous_metric_loss(emb: torch.Tensor, ranks: torch.Tensor) -> torch.Tensor:
    """Fuerza que la distancia euclidiana entre cada par de embeddings sea
    igual a su diferencia de rango |rank_i - rank_j| (MSE sobre los pares fuera
    de la diagonal). Impone una geometria lineal del eje de madurez."""
    dist = torch.cdist(emb, emb)
    target = (ranks[:, None] - ranks[None, :]).abs().float()
    off_diag = ~torch.eye(dist.size(0), dtype=torch.bool, device=dist.device)
    return nn.functional.mse_loss(dist[off_diag], target[off_diag])


def _ripeness_rank_map(class_names: list[str], device) -> torch.Tensor:
    """Tensor (num_classes,) que mapea cada label al rango de madurez de
    RIPENESS_ORDER. Cae al indice alfabetico si los nombres no calzan."""
    from src.viz import RIPENESS_ORDER

    base = [n.replace("_CHOP", "") for n in class_names]
    if all(b in RIPENESS_ORDER for b in base):
        ranks = [RIPENESS_ORDER.index(b) for b in base]
    else:
        ranks = list(range(len(class_names)))
    return torch.tensor(ranks, dtype=torch.long, device=device)


def _device(device: str) -> torch.device:
    if device == "gpu" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _write_metrics(output_dir: Path, config: Config, history: list[dict], best_score: float, best_epoch: int, by_acc: bool, class_names: list[str]) -> None:
    metrics = {
        "method": config.method,
        "latent_dim": config.params.latent_dim,
        "checkpoint_metric": "val_acc" if by_acc else "val_loss",
        "best_score": best_score,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "epochs_max": config.params.epochs,
        "class_names": class_names,
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
