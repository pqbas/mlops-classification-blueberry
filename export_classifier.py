"""Reconstruye el clasificador supervisado completo (encoder CNN congelado +
sonda lineal) desde un encoder.pt y lo guarda como un solo .pkl reutilizable.

Replica probe.py: extrae embeddings del split train, ajusta StandardScaler +
LogisticRegression y verifica accuracy en test. Guarda encoder_state, pipeline
sklearn, class_names y latent_dim en un unico archivo joblib.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.dataset import get_dataloaders
from src.nn.backbone import Encoder
from src.probe import extract_embeddings

RUN = Path("runs/sup_aug_full/0")
DATASET = "data/blueberry_five_classes_chopped_depurated"
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

x_train, y_train = extract_embeddings(encoder, train_loader, device)
x_test, y_test = extract_embeddings(encoder, test_loader, device)

clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
print(f"test accuracy={accuracy_score(y_test, pred):.4f} f1_macro={f1_score(y_test, pred, average='macro'):.4f}")

class_names = [c.replace("_CHOP", "") for c in class_names]

out = RUN / "classifier.pkl"
joblib.dump(
    {"encoder_state": encoder.state_dict(), "classifier": clf, "class_names": class_names, "latent_dim": LATENT_DIM, "imgsz": IMGSZ},
    out,
)
print(f"saved {out}")
