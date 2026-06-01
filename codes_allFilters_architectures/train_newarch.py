from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def resolve_path(value: str, base: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    candidate = (base / path).resolve()
    if candidate.exists():
        return str(candidate)
    return str(path.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified trainer for the new architecture.")
    parser.add_argument("--dataset", choices=["real", "sim"], required=True, help="Which dataset branch to train.")
    parser.add_argument("--train-folder", default=str(Path("..") / "trainingdata"))
    parser.add_argument("--params-file", default=str(Path("..") / "params" / "parameters.toml"))
    parser.add_argument("--vehicle-config", default="real_sensor_vehicle_config.toml")
    parser.add_argument("--folder", default=str(Path("..") / "datasetforvehiclesimulation_csv"))
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--max-samples-per-file", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    python = sys.executable

    if args.dataset == "real":
        train_folder = resolve_path(args.train_folder, root)
        params_file = resolve_path(args.params_file, root)
        vehicle_config = resolve_path(args.vehicle_config, root)
        script = root / "train_real_sensor_force_transformer.py"
        cmd = [
            python,
            str(script),
            "--train-folder",
            train_folder,
            "--params-file",
            params_file,
            "--vehicle-config",
            vehicle_config,
            "--model-out",
            args.model_out,
            "--epochs",
            str(args.epochs),
            "--device",
            args.device,
        ]
        if args.max_samples_per_file > 0:
            cmd += ["--max-samples-per-file", str(args.max_samples_per_file)]
    else:
        folder = resolve_path(args.folder, root)
        script = root / "train_sim_newarch_transformer.py"
        cmd = [
            python,
            str(script),
            "--folder",
            folder,
            "--model-out",
            args.model_out,
            "--epochs",
            str(args.epochs),
            "--device",
            args.device,
        ]
        if args.max_files > 0:
            cmd += ["--max-files", str(args.max_files)]
        if args.max_samples_per_file > 0:
            cmd += ["--max-rows-per-file", str(args.max_samples_per_file)]

    raise SystemExit(subprocess.run(cmd, check=False).returncode)


if __name__ == "__main__":
    main()
