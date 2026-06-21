"""Exporta el clasificador supervisado a un unico .npz sin sklearn y verifica
paridad.

El artefacto es autocontenido (un solo archivo, encaja con el esquema
ClassificationModel del robot): los pesos del encoder van como arrays
``enc__<param>`` (state_dict en numpy) junto a la sonda lineal reducida a numpy
puro (StandardScaler + LogisticRegression -> logits = W @ z + b sobre el
embedding estandarizado, argmax = clase). El robot (Jetson, sin sklearn)
reconstruye el Encoder desde las claves ``enc__`` y clasifica con un solo
np.load.

Parity: compara la prediccion numpy contra sklearn sobre el split test. Debe dar
label_match=1.0000 antes de subir el .npz.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.dataset import get_dataloaders
from src.nn.backbone import Encoder
from src.probe import extract_embeddings

RUN = Path("runs/sup_aug_full/0")
DATASET = Path("data/blueberry_five_classes_chopped_depurated")
LATENT_DIM = 64
IMGSZ = 128
BATCH = 64
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder(LATENT_DIM).to(device)
encoder.load_state_dict(torch.load(RUN / "encoder.pt", map_location=device))
encoder.eval()

train_loader, _val, test_loader, class_names = get_dataloaders(
    DATASET, batch_size=BATCH, imgsz=IMGSZ, seed=SEED, workers=4, augmentation=None
)
class_names = [c.replace("_CHOP", "") for c in class_names]

x_train, y_train = extract_embeddings(encoder, train_loader, device)
x_test, y_test = extract_embeddings(encoder, test_loader, device)

clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
clf.fit(x_train, y_train)
pred_sklearn = clf.predict(x_test)
print(f"sklearn test accuracy={accuracy_score(y_test, pred_sklearn):.4f}")

scaler: StandardScaler = clf.named_steps["standardscaler"]
logreg: LogisticRegression = clf.named_steps["logisticregression"]
mean = scaler.mean_.astype(np.float32)            # (D,)
scale = scaler.scale_.astype(np.float32)          # (D,)
W = logreg.coef_.astype(np.float32)               # (n_classes, D)
b = logreg.intercept_.astype(np.float32)          # (n_classes,)
classes = logreg.classes_.astype(np.int64)        # (n_classes,) indice -> label original


def numpy_predict(emb: np.ndarray) -> np.ndarray:
    """Replica clf.predict en numpy puro: estandariza, logits W@z+b, argmax."""
    z = (emb - mean) / scale
    logits = z @ W.T + b
    return classes[logits.argmax(axis=1)]


pred_numpy = numpy_predict(x_test)
label_match = float((pred_numpy == pred_sklearn).mean())
print(f"parity: label_match={label_match:.4f} numpy_accuracy={accuracy_score(y_test, pred_numpy):.4f}")

# Fold the frozen encoder weights into the same file so the deploy artifact is a
# single self-contained npz (one filename / one file_hash for the robot's
# ClassificationModel). State_dict keys keep their dots under the ``enc__``
# prefix; np.savez stores them as zip members so dotted names survive.
enc_arrays = {f"enc__{k}": v.detach().cpu().numpy() for k, v in encoder.state_dict().items()}

out = RUN / "classifier.npz"
np.savez(
    out,
    mean=mean,
    scale=scale,
    coef=W,
    intercept=b,
    classes=classes,
    class_names=np.array(class_names),
    latent_dim=np.int64(LATENT_DIM),
    imgsz=np.int64(IMGSZ),
    **enc_arrays,
)
print(f"saved {out} ({len(enc_arrays)} encoder tensors + probe)")
