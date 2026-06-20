# intercon2026-blueberry-classification

Aprendizaje de representaciones self-supervised para clasificar madurez de arandanos. Compara cinco paradigmas (supervised, autoencoder, VQ-VAE, RVQ-VAE, JEPA) sobre un backbone CNN compartido, entrenado desde cero, con sonda lineal/kNN downstream sobre el embedding congelado.

## Entorno (uv)

```bash
uv sync          # crea/sincroniza .venv
uv run python -m src run experiments/compare_methods.yaml
```

### GPU local: torch debe ser cu124

La laptop de desarrollo (RTX 2070 Max-Q) tiene driver NVIDIA que soporta hasta CUDA 12.4. El torch por defecto que resuelve uv es `+cu130` y falla con `CUDA initialization: driver too old`.

- `torch==2.6.0` / `torchvision==0.21.0` estan fijados al indice `pytorch-cu124` en `pyproject.toml` (`[[tool.uv.index]]` + `[tool.uv.sources]`).
- Si `import torch` falla con `libcudnn.so.9: cannot open shared object file`, las libs nvidia quedaron como stubs: correr `uv sync --reinstall` para traer los `.so` reales.
- El entrenamiento real corre en Modal (GPU T4); el GPU local es solo para smoke tests.

## Estructura

- `src/nn/`: primitivas reutilizables (`layers.py`, `backbone.py` Encoder/Decoder, `quantizers.py` VQ/RVQ).
- `src/models/`: cada paradigma compone el backbone (`supervised`, `autoencoder`, `vqvae`, `rvq`, `jepa`).
- `src/dataset.py`: split 70/15/15 estratificado, resize 128x128 por estiramiento, ColorJitter en train.
- `src/train.py`, `src/probe.py`: entrenamiento y sonda downstream.
- `src/runtimes/`: `local.py` (secuencial) y `modal.py` (paralelo en Modal).
- `experiments/*.yaml`: configs de sweep (eje `method` x `latent_dim`).
