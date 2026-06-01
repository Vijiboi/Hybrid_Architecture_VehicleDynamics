import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.io import loadmat


def derive_global_position(time_s: np.ndarray, vx_mps: np.ndarray, vy_mps: np.ndarray, yaw_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Convert body-frame velocities into global-frame velocities, then integrate.
    vx_global = vx_mps * np.cos(yaw_rad) - vy_mps * np.sin(yaw_rad)
    vy_global = vx_mps * np.sin(yaw_rad) + vy_mps * np.cos(yaw_rad)

    x_m = cumulative_trapezoid(vx_global, time_s, initial=0.0)
    y_m = cumulative_trapezoid(vy_global, time_s, initial=0.0)
    return x_m, y_m


def load_mat_as_tables(mat_path: Path) -> tuple[pd.DataFrame, dict[str, float | int]]:
    data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    time_series: dict[str, np.ndarray] = {}
    metadata: dict[str, float | int] = {}

    for key, value in data.items():
        if key.startswith("__"):
            continue

        if isinstance(value, np.ndarray) and value.ndim == 1:
            time_series[key] = np.asarray(value)
        elif np.isscalar(value):
            metadata[key] = value.item() if hasattr(value, "item") else value

    df = pd.DataFrame(time_series)

    if {"Time", "Vx", "Vy", "yaw"}.issubset(df.columns):
        global_x_m, global_y_m = derive_global_position(
            time_s=df["Time"].to_numpy(dtype=float),
            vx_mps=df["Vx"].to_numpy(dtype=float),
            vy_mps=df["Vy"].to_numpy(dtype=float),
            yaw_rad=df["yaw"].to_numpy(dtype=float),
        )
        df["global_x_m"] = global_x_m
        df["global_y_m"] = global_y_m

    return df, metadata


def convert_all(source_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []

    for mat_path in sorted(source_dir.glob("*.mat")):
        df, metadata = load_mat_as_tables(mat_path)

        csv_path = output_dir / f"{mat_path.stem}.csv"
        json_path = output_dir / f"{mat_path.stem}_metadata.json"
        df.to_csv(csv_path, index=False)
        json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        summary_row: dict[str, object] = {
            "scenario": mat_path.stem,
            "rows": len(df),
            "time_start_s": float(df["Time"].iloc[0]) if "Time" in df else np.nan,
            "time_end_s": float(df["Time"].iloc[-1]) if "Time" in df else np.nan,
            "columns": ",".join(df.columns),
        }
        summary_row.update(metadata)
        summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "scenario_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert vehicle simulation MAT datasets into CSV files.")
    parser.add_argument(
        "--source-dir",
        default=r"..\datasetforvehiclesimulation",
        help="Directory containing the .mat files.",
    )
    parser.add_argument(
        "--output-dir",
        default=r"..\datasetforvehiclesimulation_csv",
        help="Directory where converted CSV files will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    convert_all(source_dir, output_dir)
    print(f"Converted MAT files from {source_dir}")
    print(f"Saved CSV files to {output_dir}")


if __name__ == "__main__":
    main()
