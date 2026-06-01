from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vehicle_sim_loader import load_vehicle_sim_csv


def convert_one(input_csv: Path, output_csv: Path) -> None:
    data = load_vehicle_sim_csv(input_csv)
    mass = float(data.m_kg)
    ax_mps2 = (np.asarray(pd.read_csv(input_csv)["Fxf"], dtype=float) + np.asarray(pd.read_csv(input_csv)["Fxr"], dtype=float)) / max(mass, 1.0)

    out = pd.DataFrame(
        {
            "vx_mps": data.vx_mps,
            "vy_mps": data.vy_mps,
            "dpsi_radps": data.yaw_rate_rps,
            "ax_mps2": ax_mps2,
            "ay_mps2": data.ay_mps2,
            "deltawheel_rad": data.delta_rad,
            "TwheelRL_Nm": np.zeros_like(data.time_s),
            "TwheelRR_Nm": np.zeros_like(data.time_s),
            "pBrakeF_bar": np.zeros_like(data.time_s),
            "pBrakeR_bar": np.zeros_like(data.time_s),
            "time_s": data.time_s,
            "beta_true_rad": data.beta_true_rad,
            "yaw_acc_rps2": data.yaw_acc_rps2,
            "fyf_true_n": data.fyf_true_n,
            "fyr_true_n": data.fyr_true_n,
        }
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    metadata = {
        "m": mass,
        "Iz": float(data.iz_kgm2),
        "lf": float(data.lf_m),
        "lr": float(data.lr_m),
        "cf": float(data.cf_nprad),
        "cr": float(data.cr_nprad),
        "Ts": float(np.median(np.diff(data.time_s))) if len(data.time_s) > 1 else 0.01,
    }
    output_csv.with_name(f"{output_csv.stem}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Converted {input_csv.name} -> {output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert the simulation CSVs into a TUM-like layout.")
    parser.add_argument("--input", help="Input simulation CSV file.")
    parser.add_argument("--folder", help="Input folder with simulation CSV files.")
    parser.add_argument("--output", help="Output CSV file for single-file mode.")
    parser.add_argument("--output-folder", help="Output folder for batch mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.input:
        if not args.output:
            raise ValueError("--output is required when using --input")
        convert_one(Path(args.input), Path(args.output))
        return

    if args.folder:
        if not args.output_folder:
            raise ValueError("--output-folder is required when using --folder")
        folder = Path(args.folder)
        out_folder = Path(args.output_folder)
        for csv_path in sorted(folder.glob("*.csv")):
            if csv_path.name.endswith("_metadata.json") or csv_path.name == "scenario_summary.csv":
                continue
            convert_one(csv_path, out_folder / csv_path.name)
        return

    raise ValueError("Provide either --input/--output or --folder/--output-folder")


if __name__ == "__main__":
    main()
