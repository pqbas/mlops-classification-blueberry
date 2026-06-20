from __future__ import annotations

from pathlib import Path

from src.config import Config


def execute(runs: list[tuple[Config, Path]]) -> None:
    """Ejecuta los trials secuencialmente en la maquina local (sin Modal).
    Util para smoke tests antes de lanzar el job remoto."""
    raise NotImplementedError
