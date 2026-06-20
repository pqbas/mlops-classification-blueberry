from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

_EXTS = {".jpg", ".jpeg", ".png"}


def discover_classes(data_dir: Path) -> list[str]:
    """Nombres de clase (subcarpetas) en orden alfabetico estable."""
    return sorted(p.name for p in Path(data_dir).iterdir() if p.is_dir())


def build_splits(data_dir: Path, seed: int = 42) -> dict:
    """Recorre `data_dir` (una subcarpeta por clase) y divide 70/15/15
    estratificado por clase. Retorna {'train': [...], 'val': [...], 'test': [...]}
    donde cada item es (ruta_imagen, indice_clase)."""
    data_dir = Path(data_dir)
    classes = discover_classes(data_dir)
    class_to_idx = {name: i for i, name in enumerate(classes)}

    paths: list[Path] = []
    labels: list[int] = []
    for name in classes:
        for img in sorted((data_dir / name).iterdir()):
            if img.suffix.lower() in _EXTS:
                paths.append(img)
                labels.append(class_to_idx[name])

    # 70 / 15 / 15 estratificado: primero separa test (15%), luego val (15/85).
    train_val_p, test_p, train_val_y, test_y = train_test_split(
        paths, labels, test_size=0.15, stratify=labels, random_state=seed
    )
    train_p, val_p, train_y, val_y = train_test_split(
        train_val_p,
        train_val_y,
        test_size=0.15 / 0.85,
        stratify=train_val_y,
        random_state=seed,
    )
    return {
        "train": list(zip(train_p, train_y)),
        "val": list(zip(val_p, val_y)),
        "test": list(zip(test_p, test_y)),
    }


class BlueberryDataset(Dataset):
    """Carga imagen JPG del fruto segmentado, resize 128x128 por estiramiento,
    augmentation opcional de brillo/color en train. Devuelve (tensor, label)."""

    def __init__(
        self,
        samples: list,
        imgsz: int = 128,
        train: bool = False,
        augmentation: dict | None = None,
    ) -> None:
        self.samples = samples
        tfms: list = [transforms.Resize((imgsz, imgsz))]
        if train:
            # El fruto no tiene orientacion canonica: flips y rotacion completa
            # son label-preserving y no alteran el color (senal de madurez).
            # fill=255 mantiene el fondo blanco al rotar.
            tfms += [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(180, fill=255),
            ]
            if augmentation:
                tfms.append(
                    transforms.ColorJitter(
                        brightness=augmentation.get("brightness", 0.0),
                        saturation=augmentation.get("saturation", 0.0),
                        hue=augmentation.get("hue", 0.0),
                    )
                )
        tfms.append(transforms.ToTensor())  # a [0, 1], (C, H, W)
        self.transform = transforms.Compose(tfms)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def get_dataloaders(
    data_dir: Path,
    batch_size: int = 64,
    imgsz: int = 128,
    seed: int = 42,
    workers: int = 8,
    augmentation: dict | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Devuelve (train_loader, val_loader, test_loader, class_names) con el mismo
    split para todos los modelos, de modo que la comparacion sea justa."""
    splits = build_splits(data_dir, seed=seed)
    class_names = discover_classes(Path(data_dir))

    train_ds = BlueberryDataset(
        splits["train"], imgsz, train=True, augmentation=augmentation
    )
    val_ds = BlueberryDataset(splits["val"], imgsz, train=False)
    test_ds = BlueberryDataset(splits["test"], imgsz, train=False)

    def _loader(ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=True,
        )

    return (
        _loader(train_ds, True),
        _loader(val_ds, False),
        _loader(test_ds, False),
        class_names,
    )
