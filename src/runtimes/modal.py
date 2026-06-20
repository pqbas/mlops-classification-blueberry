from __future__ import annotations

import copy
import os
import time
from pathlib import Path

os.environ.setdefault("MODAL_PROFILE", "pcubasm1")

import modal

from src.config import Config

APP_NAME = "blueberry-ssl"
DATASETS_VOLUME = "blueberry-datasets"
OUTPUTS_VOLUME = "blueberry-runs"
DATASETS_MOUNT = "/datasets"
OUTPUTS_MOUNT = "/outputs"
GPU = "T4"
TIMEOUT_SECONDS = 7200
POLL_INTERVAL_SECONDS = 5

_image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "numpy>=1.26",
        "pillow>=10.0",
        "scikit-learn>=1.4",
        "umap-learn>=0.5",
        "matplotlib>=3.8",
        "pyyaml>=6.0",
    )
    .add_local_python_source("src")
)

_dataset_vol = modal.Volume.from_name(DATASETS_VOLUME, create_if_missing=True)
_outputs_vol = modal.Volume.from_name(OUTPUTS_VOLUME, create_if_missing=True)

app = modal.App(APP_NAME)


@app.function(
    image=_image,
    gpu=GPU,
    volumes={DATASETS_MOUNT: _dataset_vol, OUTPUTS_MOUNT: _outputs_vol},
    timeout=TIMEOUT_SECONDS,
)
def _run_experiment(config_raw: dict, output_subdir: str) -> None:
    from src import train as train_mod
    from src.config import parse_dict

    cfg = parse_dict(config_raw)
    out_root = Path(OUTPUTS_MOUNT) / output_subdir
    out_root.mkdir(parents=True, exist_ok=True)

    train_mod.run(cfg, out_root)
    _outputs_vol.commit()


def execute(runs: list[tuple[Config, Path]]) -> None:
    """Sube el dataset al volumen una vez, lanza un contenedor GPU por trial
    con .spawn() (paralelo), hace polling y descarga los outputs de cada uno."""
    if not runs:
        return

    dataset_dir = runs[0][0].dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {dataset_dir}")
    dataset_name = dataset_dir.name
    _ensure_dataset_volume(dataset_dir, dataset_name)

    print(f"[modal] Job remoto GPU={GPU} trials={len(runs)}")
    with modal.enable_output():
        with app.run():
            pending: dict = {}
            for cfg, out_dir in runs:
                raw_remote = _rewrite_for_remote(cfg, dataset_name)
                subdir = str(out_dir.relative_to(Path("runs")))
                call = _run_experiment.spawn(raw_remote, subdir)
                pending[call] = (out_dir, subdir)

            while pending:
                finished = []
                for call, (out_dir, subdir) in pending.items():
                    try:
                        call.get(timeout=0)
                    except TimeoutError:
                        continue
                    finished.append(call)
                    print(f"[modal] Trial terminado, descargando {out_dir}")
                    _download_dir(subdir, out_dir)
                for call in finished:
                    del pending[call]
                if pending:
                    time.sleep(POLL_INTERVAL_SECONDS)


def _ensure_dataset_volume(local_dir: Path, name: str) -> None:
    if _volume_has_entries(name):
        print(f"[modal] Dataset '{name}' ya esta en el volumen, omito upload")
        return
    print(f"[modal] Subiendo dataset '{name}' al volumen (puede tardar)")
    with _dataset_vol.batch_upload(force=False) as batch:
        batch.put_directory(str(local_dir), name)


def _volume_has_entries(name: str) -> bool:
    try:
        return any(True for _ in _dataset_vol.iterdir(name))
    except Exception:
        return False


def _rewrite_for_remote(config: Config, dataset_name: str) -> dict:
    raw = copy.deepcopy(config.raw)
    raw["dataset"] = f"{DATASETS_MOUNT}/{dataset_name}"
    return raw


def _download_dir(remote_path: str, local_dir: Path) -> None:
    from modal.volume import FileEntryType

    local_dir.mkdir(parents=True, exist_ok=True)
    for entry in _outputs_vol.iterdir(remote_path, recursive=False):
        name = Path(entry.path).name
        target = local_dir / name
        if entry.type == FileEntryType.DIRECTORY:
            _download_dir(entry.path, target)
        else:
            with target.open("wb") as f:
                for chunk in _outputs_vol.read_file(entry.path):
                    f.write(chunk)
