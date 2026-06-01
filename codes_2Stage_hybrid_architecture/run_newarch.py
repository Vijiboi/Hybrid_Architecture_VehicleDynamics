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
    parser = argparse.ArgumentParser(description="Unified runner for the new architecture.")
    parser.add_argument("--dataset", choices=["real", "sim"], required=True, help="Which dataset branch to run.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--plot", required=True)
    parser.add_argument("--params-file", default=str(Path("..") / "params" / "parameters.toml"))
    parser.add_argument("--vehicle-config", default="real_sensor_vehicle_config.toml")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    python = sys.executable

    if args.dataset == "real":
        csv = resolve_path(args.csv, root)
        model = resolve_path(args.model, root)
        plot = resolve_path(args.plot, root)
        params_file = resolve_path(args.params_file, root)
        vehicle_config = resolve_path(args.vehicle_config, root)
        script = root / "run_real_sensor_force_ekf.py"
        cmd = [
            python,
            str(script),
            "--csv",
            csv,
            "--params-file",
            params_file,
            "--vehicle-config",
            vehicle_config,
            "--model",
            model,
            "--plot",
            plot,
            "--device",
            args.device,
        ]
        if args.max_samples > 0:
            cmd += ["--max-samples", str(args.max_samples)]
    else:
        csv = resolve_path(args.csv, root)
        model = resolve_path(args.model, root)
        plot = resolve_path(args.plot, root)
        script = root / "run_sim_newarch_ekf.py"
        cmd = [
            python,
            str(script),
            "--csv",
            csv,
            "--model",
            model,
            "--plot",
            plot,
            "--device",
            args.device,
        ]
        if args.max_samples > 0:
            cmd += ["--max-samples", str(args.max_samples)]

    raise SystemExit(subprocess.run(cmd, check=False).returncode)


if __name__ == "__main__":
    main()
