from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import Config
from src.dataset import get_dataloaders
from src.nn.backbone import Encoder


def run(config: Config, output_dir: Path) -> None:
    """Evalua el embedding congelado con sonda lineal y k-NN sobre el split
    test, y escribe probe.json (accuracy, f1 macro). El encoder se carga
    congelado desde `output_dir/encoder.pt`.
    """
    device = torch.device("cuda" if (config.device == "gpu" and torch.cuda.is_available()) else "cpu")

    encoder = Encoder(config.params.latent_dim).to(device)
    encoder.load_state_dict(torch.load(output_dir / "encoder.pt", map_location=device))
    encoder.eval()

    train_loader, _val_loader, test_loader, class_names = get_dataloaders(
        config.dataset,
        batch_size=config.params.batch_size,
        imgsz=config.params.imgsz,
        seed=config.params.seed,
        workers=config.params.workers,
        augmentation=None,  # sin augmentation para extraer representaciones limpias
    )

    x_train, y_train = extract_embeddings(encoder, train_loader, device)
    x_test, y_test = extract_embeddings(encoder, test_loader, device)

    results = {
        "method": config.method,
        "latent_dim": config.params.latent_dim,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "class_names": class_names,
        "linear": _eval_linear(x_train, y_train, x_test, y_test),
        "knn": _eval_knn(x_train, y_train, x_test, y_test),
    }
    (output_dir / "probe.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[probe] {config.method} linear_acc={results['linear']['accuracy']:.4f} knn_acc={results['knn']['accuracy']:.4f}")


@torch.no_grad()
def extract_embeddings(encoder: Encoder, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (embeddings, labels) recorriendo el loader con el encoder congelado."""
    feats, labels = [], []
    for x, y in loader:
        emb = encoder.embed(x.to(device))
        feats.append(emb.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def _eval_linear(x_train, y_train, x_test, y_test) -> dict:
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    return _scores(y_test, pred)


def _eval_knn(x_train, y_train, x_test, y_test, k: int = 5) -> dict:
    clf = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    return _scores(y_test, pred)


def _scores(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }
