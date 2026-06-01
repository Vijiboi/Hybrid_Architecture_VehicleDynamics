import argparse
from pathlib import Path

import numpy as np

from real_sensor_common import build_sensor_windows
from real_sensor_common import build_structured_force_and_beta_targets
from real_sensor_common import load_project_configs
from real_sensor_common import load_real_sensor_csv
from real_sensor_common import model_output_to_beta_measurement
from real_sensor_common import model_output_to_wheel_forces
from real_sensor_common import slice_real_sensor_dataset
from real_sensor_common import train_force_model
from real_sensor_common import RealSensorForcePredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a force transformer on the realistic 10-channel trainingdata files.")
    parser.add_argument("--train-folder", default=str(Path("..") / "trainingdata"))
    parser.add_argument("--params-file", default=str(Path("..") / "params" / "parameters.toml"))
    parser.add_argument("--vehicle-config", default="real_sensor_vehicle_config.toml")
    parser.add_argument("--model-out", default="real_sensor_force_transformer.pt")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--max-samples-per-file", type=int, default=0, help="Optional cap per training CSV for quick tuning runs.")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nn_cfg, vehicle, _ = load_project_configs(args.params_file, args.vehicle_config)
    train_files = sorted(Path(args.train_folder).glob("data_to_train_*.csv"))
    if len(train_files) < 2:
        raise ValueError("Need at least 2 training CSV files")

    val_count = max(1, int(np.ceil(len(train_files) * nn_cfg.val_split)))
    train_split = train_files[:-val_count]
    val_split = train_files[-val_count:]

    X_train_list = []
    y_train_list = []
    X_val_list = []
    y_val_list = []
    train_beta_truth_list = []
    val_beta_truth_list = []

    for path in train_split:
        ds = slice_real_sensor_dataset(load_real_sensor_csv(path, vehicle), args.max_samples_per_file)
        X_train_list.append(build_sensor_windows(ds.sensor_matrix, nn_cfg.input_timesteps))
        y_train_list.append(build_structured_force_and_beta_targets(ds, vehicle))
        train_beta_truth_list.append(ds.beta_ref_rad)
    for path in val_split:
        ds = slice_real_sensor_dataset(load_real_sensor_csv(path, vehicle), args.max_samples_per_file)
        X_val_list.append(build_sensor_windows(ds.sensor_matrix, nn_cfg.input_timesteps))
        y_val_list.append(build_structured_force_and_beta_targets(ds, vehicle))
        val_beta_truth_list.append(ds.beta_ref_rad)

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_val = np.concatenate(X_val_list, axis=0)
    y_val = np.concatenate(y_val_list, axis=0)

    bundle, history = train_force_model(
        X_train,
        y_train,
        X_val,
        y_val,
        nn_cfg,
        target_mode="structured_body_force_and_split_beta",
        epochs=args.epochs,
        device=args.device,
    )
    bundle.save(args.model_out)

    predictor = RealSensorForcePredictor(bundle, device=args.device)
    train_pred_struct = predictor.predict(X_train)
    val_pred_struct = predictor.predict(X_val)

    train_wheel_truth = []
    val_wheel_truth = []
    train_sensor_rows = []
    val_sensor_rows = []
    for path in train_split:
        ds = slice_real_sensor_dataset(load_real_sensor_csv(path, vehicle), args.max_samples_per_file)
        train_wheel_truth.append(ds.wheel_force_targets_n)
        train_sensor_rows.append(ds.sensor_matrix)
    for path in val_split:
        ds = slice_real_sensor_dataset(load_real_sensor_csv(path, vehicle), args.max_samples_per_file)
        val_wheel_truth.append(ds.wheel_force_targets_n)
        val_sensor_rows.append(ds.sensor_matrix)

    train_pred = model_output_to_wheel_forces(train_pred_struct, np.concatenate(train_sensor_rows, axis=0), bundle, vehicle)
    val_pred = model_output_to_wheel_forces(val_pred_struct, np.concatenate(val_sensor_rows, axis=0), bundle, vehicle)
    train_truth = np.concatenate(train_wheel_truth, axis=0)
    val_truth = np.concatenate(val_wheel_truth, axis=0)
    train_rmse = float(np.sqrt(np.mean((train_pred - train_truth) ** 2)))
    val_rmse = float(np.sqrt(np.mean((val_pred - val_truth) ** 2)))
    train_beta_truth = np.concatenate(train_beta_truth_list, axis=0)
    val_beta_truth = np.concatenate(val_beta_truth_list, axis=0)

    print(f"Saved model to {args.model_out}")
    print(f"Training files: {len(train_split)}")
    print(f"Validation files: {len(val_split)}")
    if args.max_samples_per_file > 0:
        print(f"Samples per file cap: {args.max_samples_per_file}")
    print(f"Train wheel-force RMSE: {train_rmse:.6f} N")
    print(f"Val wheel-force RMSE:   {val_rmse:.6f} N")
    train_beta_pred = model_output_to_beta_measurement(train_pred_struct)
    val_beta_pred = model_output_to_beta_measurement(val_pred_struct)
    if train_beta_pred is not None and val_beta_pred is not None:
        print(f"Train beta RMSE:       {float(np.sqrt(np.mean((train_beta_pred - train_beta_truth) ** 2))):.6f} rad")
        print(f"Val beta RMSE:         {float(np.sqrt(np.mean((val_beta_pred - val_beta_truth) ** 2))):.6f} rad")
    print(f"Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"Final val loss:   {history['val_loss'][-1]:.6f}")


if __name__ == "__main__":
    main()
