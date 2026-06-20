from __future__ import annotations

import argparse
import importlib
from datetime import datetime
from pathlib import Path

from src.config import expand_configs, load_yaml


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ops", description="Blueberry SSL runner")
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="Ejecuta un experimento desde un YAML")
    run_p.add_argument("yaml", type=Path, help="Ruta al YAML del experimento")
    args = parser.parse_args(argv)

    if args.command == "run":
        return _cmd_run(args.yaml)
    parser.error(f"Comando desconocido: {args.command}")
    return 2


def _cmd_run(yaml_path: Path) -> int:
    """Carga el YAML, expande el sweep sobre `method`, crea runs/<name>/<ts>/
    y delega en el runtime (local o modal)."""
    data = load_yaml(yaml_path)
    configs = expand_configs(data)
    if not configs:
        print("[run] Sin trials para ejecutar")
        return 1

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path("runs") / configs[0][0].name / stamp
    runs = [(cfg, base / (suffix or cfg.method)) for cfg, suffix in configs]

    runtime = configs[0][0].runtime
    module = importlib.import_module(f"src.runtimes.{_runtime_module(runtime)}")
    print(f"[run] runtime={runtime} trials={len(runs)} -> {base}")
    module.execute(runs)
    return 0


def _runtime_module(runtime: str) -> str:
    return "local" if runtime == "localhost" else runtime


if __name__ == "__main__":
    raise SystemExit(main())
