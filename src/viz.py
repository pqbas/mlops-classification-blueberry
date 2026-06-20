from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend headless (Modal / sin display)
import matplotlib.pyplot as plt
import numpy as np

# Orden de madurez verde -> azul, para colorear como una trayectoria continua.
RIPENESS_ORDER = ["VERDE", "CREMOSO", "ROSADO", "PINTON1", "PINTON2", "GUINDA", "AZUL"]

# Nombres legibles por metodo, para los titulos de las figuras.
PRETTY_NAMES = {
    "supervised": "Supervised",
    "autoencoder": "Autoencoder",
    "vqvae": "VQ-VAE",
    "rvq": "RVQ-VAE",
    "jepa": "JEPA",
}


def plot_embedding(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    method: str = "umap",
    class_names: list[str] | None = None,
    model_name: str | None = None,
) -> Path:
    """Proyecta embeddings a 2D con UMAP o t-SNE, coloreados por madurez, y
    guarda la figura. Si las clases siguen RIPENESS_ORDER se usa un colormap
    secuencial para ver si emerge la trayectoria continua verde -> azul.
    `model_name` es el metodo (supervised, autoencoder, ...) para el titulo."""
    coords = _reduce_2d(embeddings, method)
    ranks, names = _ripeness_ranks(class_names, labels)

    fig, ax = plt.subplots(figsize=(8, 7))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=ranks, cmap="viridis", s=12, alpha=0.8)
    if names is not None:
        cbar = fig.colorbar(scatter, ax=ax, ticks=range(len(names)))
        cbar.ax.set_yticklabels(names)
        cbar.set_label("madurez (verde -> azul)")
    pretty = PRETTY_NAMES.get(model_name or "", model_name or "")
    title = f"{pretty} - {method.upper()}" if pretty else f"{method.upper()} del embedding"
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _reduce_2d(embeddings: np.ndarray, method: str) -> np.ndarray:
    if method == "tsne":
        from sklearn.manifold import TSNE

        return TSNE(n_components=2, random_state=42, init="pca").fit_transform(embeddings)
    import umap

    return umap.UMAP(n_components=2, random_state=42).fit_transform(embeddings)


def _ripeness_ranks(class_names: list[str] | None, labels: np.ndarray):
    """Mapea cada label a su rango de madurez (no a su indice alfabetico), para
    que el colormap secuencial lea como la trayectoria. Cae a labels crudos si
    los nombres no calzan con RIPENESS_ORDER."""
    if class_names is None:
        return labels, None
    base = [n.replace("_CHOP", "") for n in class_names]
    if not all(b in RIPENESS_ORDER for b in base):
        return labels, None
    idx_to_rank = {i: RIPENESS_ORDER.index(b) for i, b in enumerate(base)}
    ranks = np.array([idx_to_rank[int(l)] for l in labels])
    return ranks, RIPENESS_ORDER


def plot_loss_curve(
    history: list[dict],
    output_path: Path,
    model_name: str | None = None,
    best_epoch: int | None = None,
) -> Path:
    """Grafica train_loss y val_loss por epoca desde el history de metrics.json.
    Marca con linea vertical la epoca del mejor checkpoint si se pasa."""
    epochs = [h["epoch"] for h in history]
    train = [h["train_loss"] for h in history]
    val = [h["val_loss"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train, label="train", color="tab:blue")
    ax.plot(epochs, val, label="val", color="tab:orange")
    if best_epoch is not None and best_epoch >= 0:
        ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, label=f"best (ep {best_epoch})")
    pretty = PRETTY_NAMES.get(model_name or "", model_name or "")
    ax.set_title(f"{pretty} - loss" if pretty else "loss")
    ax.set_xlabel("epoca")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def latent_traversal(decoder, z_start, z_end, steps: int, output_path: Path) -> Path:
    """Interpola linealmente en el espacio latente entre dos mapas y decodifica
    cada paso, visualizando la transicion de color codificada (autoencoder /
    VQ-VAE / RVQ). z_start y z_end son mapas latentes (C, 8, 8)."""
    import torch

    decoder.eval()
    with torch.no_grad():
        alphas = torch.linspace(0, 1, steps)
        zs = torch.stack([(1 - a) * z_start + a * z_end for a in alphas])
        imgs = decoder(zs).cpu().clamp(0, 1)

    fig, axes = plt.subplots(1, steps, figsize=(2 * steps, 2))
    for ax, img in zip(axes, imgs):
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
