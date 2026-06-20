from __future__ import annotations

import copy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Literal

import yaml

Runtime = Literal["localhost", "modal"]
Device = Literal["gpu", "cpu"]
OpType = Literal["train", "probe", "train_probe"]
Method = Literal["supervised", "autoencoder", "vqvae", "rvq", "jepa", "metric_triplet", "metric_continuous"]

_VALID_RUNTIMES = {"localhost", "modal"}
_VALID_DEVICES = {"gpu", "cpu"}
_VALID_TYPES = {"train", "probe", "train_probe"}
_VALID_METHODS = {"supervised", "autoencoder", "vqvae", "rvq", "jepa", "metric_triplet", "metric_continuous"}


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class TrainParams:
    """Hiperparametros de entrenamiento comunes a los cinco paradigmas."""

    epochs: int
    batch_size: int
    imgsz: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.1
    latent_dim: int = 128
    workers: int = 8
    seed: int = 42


@dataclass(frozen=True)
class Config:
    """Configuracion de un trial. `method` y `latent_dim` son los ejes del sweep."""

    name: str
    type: OpType
    runtime: Runtime
    device: Device
    dataset: Path
    method: Method
    params: TrainParams
    raw: dict
    augmentation: dict | None = None


def load_yaml(path: Path) -> dict:
    """Lee y valida que el YAML sea un mapping en el nivel raiz."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"YAML no encontrado: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"YAML invalido en {path}: se esperaba un mapping en la raiz")
    return data


def expand_configs(data: dict) -> list[tuple[Config, str | None]]:
    """Producto cartesiano sobre `method` y `latent_dim` (campos coma-separados).

    Retorna [(cfg, suffix)]; suffix None si hay un solo trial, o
    '<idx>_<method>_l<latent>' si hay sweep, para nombrar el subdirectorio.
    """
    params_data = data.get("params") or {}
    methods = _split_str(data.get("method"))
    latents = _split_int(params_data.get("latent_dim"))
    if not methods:
        raise ConfigError("Falta el campo 'method'")
    if not latents:
        latents = [128]

    combos = list(product(methods, latents))
    multi = len(combos) > 1
    results: list[tuple[Config, str | None]] = []
    for idx, (method, latent) in enumerate(combos):
        variant = copy.deepcopy(data)
        variant["method"] = method
        variant.setdefault("params", {})
        variant["params"]["latent_dim"] = latent
        suffix = f"{idx}_{method}_l{latent}" if multi else None
        results.append((parse_dict(variant), suffix))
    return results


def parse_dict(data: dict) -> Config:
    """Valida y construye un Config a partir del dict del YAML."""
    name = _require_str(data, "name")
    op_type = _require_choice(data, "type", _VALID_TYPES)
    runtime = _require_choice(data, "runtime", _VALID_RUNTIMES)
    device = _require_choice(data, "device", _VALID_DEVICES)
    dataset = Path(_require_str(data, "dataset"))
    method = _require_choice(data, "method", _VALID_METHODS)

    params = _require_mapping(data, "params")
    augmentation = data.get("augmentation")
    if augmentation is not None and not isinstance(augmentation, dict):
        raise ConfigError("Campo 'augmentation' debe ser un mapping")

    return Config(
        name=name,
        type=op_type,  # type: ignore[arg-type]
        runtime=runtime,  # type: ignore[arg-type]
        device=device,  # type: ignore[arg-type]
        dataset=dataset,
        method=method,  # type: ignore[arg-type]
        params=TrainParams(
            epochs=_require_int(params, "epochs"),
            batch_size=_require_int(params, "batch_size"),
            imgsz=_optional_int(params, "imgsz", 128),
            lr=_optional_float(params, "lr", 1e-3),
            weight_decay=_optional_float(params, "weight_decay", 1e-4),
            dropout=_optional_float(params, "dropout", 0.1),
            latent_dim=_optional_int(params, "latent_dim", 128),
            workers=_optional_int(params, "workers", 8),
            seed=_optional_int(params, "seed", 42),
        ),
        raw=data,
        augmentation=augmentation,
    )


def _split_str(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(value)]


def _split_int(value) -> list[int]:
    if value is None:
        return []
    if isinstance(value, bool):
        raise ConfigError("Se esperaba un entero, no bool")
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        out = []
        for piece in value.split(","):
            piece = piece.strip()
            if piece:
                out.append(int(piece))
        return out
    raise ConfigError(f"Tipo no soportado para sweep entero: {type(value).__name__}")


def _require_str(data: dict, key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"Campo '{key}' requerido como string no vacio")
    return value


def _require_int(data: dict, key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"Campo '{key}' requerido como entero")
    return value


def _optional_int(data: dict, key: str, default: int) -> int:
    if key not in data:
        return default
    return _require_int(data, key)


def _optional_float(data: dict, key: str, default: float) -> float:
    if key not in data:
        return default
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"Campo '{key}' requerido como numero")
    return float(value)


def _require_mapping(data: dict, key: str) -> dict:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"Campo '{key}' requerido como mapping")
    return value


def _require_choice(data: dict, key: str, choices: set[str]) -> str:
    value = _require_str(data, key)
    if value not in choices:
        raise ConfigError(f"Campo '{key}'='{value}' invalido. Permitidos: {sorted(choices)}")
    return value
