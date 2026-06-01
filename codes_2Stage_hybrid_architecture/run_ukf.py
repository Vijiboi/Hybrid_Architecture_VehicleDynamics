from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified classical UKF runner for real or converted simulation data.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--plot", required=True)
    parser.add_argument("--sample-time", type=float, default=0.008)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    python = sys.executable
    script = root / "tum_ukf.py"
    cmd = [
        python,
        str(script),
        "--csv",
        args.csv,
        "--plot",
        args.plot,
        "--sample-time",
        str(args.sample_time),
    ]
    raise SystemExit(subprocess.run(cmd, check=False).returncode)


if __name__ == "__main__":
    main()
