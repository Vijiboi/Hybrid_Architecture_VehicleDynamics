import argparse
from pathlib import Path

import numpy as np

from hybrid_friction_common import build_friction_training_matrices
from hybrid_friction_common import compute_linear_tire_forces
from hybrid_friction_common import compute_slip_angles
from hybrid_friction_common import ForceCorrectionConfig
from hybrid_friction_common import fit_friction_residual_model
from vehicle_sim_loader import load_vehicle_sim_csv


def collect_scenario_files(folder: Path) -> list[Path]:
    def scenario_key(path: Path) -> tuple[int, str]:
        prefix = path.stem.split("_", 1)[0]
        return (int(prefix), path.name) if prefix.isdigit() else (10_000, path.name)

    return sorted((path for path in folder.glob("*.csv") if path.name != "scenario_summary.csv"), key=scenario_key)


def select_files_by_name(all_files: list[Path], requested_names: list[str] | None) -> list[Path]:
    if not requested_names:
        return all_files
    lookup = {path.name: path for path in all_files}
    selected: list[Path] = []
    missing: list[str] = []
    for name in requested_names:
        if name in lookup:
            selected.append(lookup[name])
        else:
            missing.append(name)
    if missing:
        raise ValueError(f"Requested files not found: {missing}")
    return selected


def residual_rmse(model, X: np.ndarray, y: np.ndarray) -> float:
    y_hat = model.predict_residual_forces(X)
    return float(np.sqrt(np.mean((y_hat - y) ** 2)))


def total_force_rmse(model, datasets, window_size: int) -> float:
    errors: list[np.ndarray] = []
    for data in datasets:
        params = (
            float(data.cf_nprad),
            float(data.cr_nprad),
            float(data.lf_m),
            float(data.lr_m),
        )
        X, _ = build_friction_training_matrices(data, window_size)
        residual_hat = model.predict_residual_forces(X)
        for i in range(len(data.time_s)):
            alpha_f, alpha_r = compute_slip_angles(
                beta_rad=float(data.beta_true_rad[i]),
                yaw_rate_rps=float(data.yaw_rate_rps[i]),
                delta_rad=float(data.delta_rad[i]),
                vx_mps=float(data.vx_mps[i]),
                lf_m=params[2],
                lr_m=params[3],
            )
            linear_fyf, linear_fyr = compute_linear_tire_forces(alpha_f, alpha_r, params[0], params[1])
            fy_hat = np.array([linear_fyf, linear_fyr], dtype=float) + residual_hat[i]
            fy_true = np.array([float(data.fyf_true_n[i]), float(data.fyr_true_n[i])], dtype=float)
            errors.append(fy_hat - fy_true)
    errors_array = np.vstack(errors)
    return float(np.sqrt(np.mean(errors_array**2)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small neural friction residual model on the vehicle simulation CSV scenarios.")
    parser.add_argument(
        "--folder",
        default=str(Path("..") / "datasetforvehiclesimulation_csv"),
        help="Folder containing the simulation CSV scenarios.",
    )
    parser.add_argument("--model-out", default="friction_residual_model.npz", help="Output model file.")
    parser.add_argument("--window-size", type=int, default=5, help="Number of past samples used as causal context.")
    parser.add_argument("--epochs", type=int, default=800, help="Training epochs.")
    parser.add_argument("--hidden", type=int, nargs="*", default=[48, 48], help="Hidden layer widths.")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=1.0e-3, help="Adam learning rate.")
    parser.add_argument("--l2", type=float, default=1.0e-5, help="L2 weight decay.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--val-files", type=int, default=2, help="How many scenarios to keep for validation.")
    parser.add_argument("--train-files-list", nargs="*", default=None, help="Optional explicit CSV filenames to use for training.")
    parser.add_argument("--val-files-list", nargs="*", default=None, help="Optional explicit CSV filenames to use for validation.")
    parser.add_argument("--residual-scale", type=float, default=0.15, help="Blend factor applied to the learned force correction in the filter.")
    parser.add_argument("--residual-clip", type=float, default=800.0, help="Absolute clip on each learned residual force in Newtons.")
    parser.add_argument("--total-force-clip", type=float, default=5000.0, help="Absolute clip on the final lateral tire force in Newtons.")
    parser.add_argument("--slip-angle-clip", type=float, default=0.25, help="Slip-angle clip used before force prediction, in radians.")
    parser.add_argument("--full-correction-speed", type=float, default=12.0, help="Speed where the full learned-force blend becomes active, in m/s.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folder = Path(args.folder)
    scenario_files = collect_scenario_files(folder)
    if len(scenario_files) < 1:
        raise ValueError(f"Need at least 2 scenario CSVs in {folder}")

    if args.train_files_list is not None or args.val_files_list is not None:
        train_files = select_files_by_name(scenario_files, args.train_files_list)
        val_files = select_files_by_name(scenario_files, args.val_files_list)
        if not train_files:
            raise ValueError("No training files selected.")
        if not val_files:
            val_files = train_files
    else:
        if len(scenario_files) < 2:
            raise ValueError(f"Need at least 2 scenario CSVs in {folder} for automatic train/validation split.")
        val_count = max(1, min(args.val_files, len(scenario_files) - 1))
        train_files = scenario_files[:-val_count]
        val_files = scenario_files[-val_count:]

    train_datasets = [load_vehicle_sim_csv(path) for path in train_files]
    val_datasets = [load_vehicle_sim_csv(path) for path in val_files]

    X_train_list: list[np.ndarray] = []
    y_train_list: list[np.ndarray] = []
    X_val_list: list[np.ndarray] = []
    y_val_list: list[np.ndarray] = []

    for dataset in train_datasets:
        X_block, y_block = build_friction_training_matrices(dataset, args.window_size)
        X_train_list.append(X_block)
        y_train_list.append(y_block)
    for dataset in val_datasets:
        X_block, y_block = build_friction_training_matrices(dataset, args.window_size)
        X_val_list.append(X_block)
        y_val_list.append(y_block)

    X_train = np.vstack(X_train_list)
    y_train = np.vstack(y_train_list)
    X_val = np.vstack(X_val_list)
    y_val = np.vstack(y_val_list)

    correction_config = ForceCorrectionConfig(
        residual_scale=args.residual_scale,
        residual_clip_n=args.residual_clip,
        total_force_clip_n=args.total_force_clip,
        slip_angle_clip_rad=args.slip_angle_clip,
        full_correction_speed_mps=args.full_correction_speed,
    )

    model, history = fit_friction_residual_model(
        X_train,
        y_train,
        window_size=args.window_size,
        config=correction_config,
        hidden_dims=tuple(args.hidden),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        l2=args.l2,
        seed=args.seed,
    )
    model.save(args.model_out)

    print(f"Saved friction residual model to {args.model_out}")
    print(f"Training scenarios: {len(train_files)}")
    print(f"Validation scenarios: {len(val_files)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Final training loss: {history['loss'][-1]:.6f}")
    print(f"Residual scale in filter: {correction_config.residual_scale:.3f}")
    print(f"Residual clip: {correction_config.residual_clip_n:.1f} N")
    print(f"Residual-force RMSE (train): {residual_rmse(model, X_train, y_train):.6f} N")
    print(f"Residual-force RMSE (val):   {residual_rmse(model, X_val, y_val):.6f} N")
    print(f"Total-force RMSE (train):    {total_force_rmse(model, train_datasets, args.window_size):.6f} N")
    print(f"Total-force RMSE (val):      {total_force_rmse(model, val_datasets, args.window_size):.6f} N")
    print("Training files:")
    for path in train_files:
        print(f"  {path.name}")
    print("Validation files:")
    for path in val_files:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
