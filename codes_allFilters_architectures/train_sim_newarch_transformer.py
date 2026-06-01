import argparse
from pathlib import Path

import numpy as np

from sim_newarch_common import SimNNSettings
from sim_newarch_common import SimForcePredictor
from sim_newarch_common import build_sensor_windows
from sim_newarch_common import build_targets
from sim_newarch_common import load_sim_dataset
from sim_newarch_common import scenario_sensor_matrix
from sim_newarch_common import train_model
from vehicle_sim_loader import VehicleSimDataset


def trim_dataset(data: VehicleSimDataset, max_rows: int | None) -> VehicleSimDataset:
    if max_rows is None or max_rows <= 0:
        return data
    payload: dict[str, object] = {}
    for key, value in data.__dict__.items():
        if isinstance(value, np.ndarray) and value.ndim > 0:
            payload[key] = value[:max_rows]
        else:
            payload[key] = value
    return type(data)(**payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a mini-transformer on the converted simulation CSV dataset.")
    parser.add_argument("--folder", default=str(Path("..") / "datasetforvehiclesimulation_csv"))
    parser.add_argument("--model-out", default="sim_newarch_force_transformer.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=4.0e-4)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--max-files", type=int, default=0, help="Limit how many CSV files to use, 0 means all.")
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=0,
        help="Limit rows loaded from each CSV, 0 means all rows.",
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folder = Path(args.folder)
    csv_files = sorted([p for p in folder.glob("*.csv") if p.name != "scenario_summary.csv"])
    if args.max_files and args.max_files > 0:
        csv_files = csv_files[: args.max_files]
    if len(csv_files) < 2:
        raise ValueError(f"Need at least 2 CSV files in {folder}")

    val_count = max(1, int(np.ceil(len(csv_files) * 0.25)))
    train_files = csv_files[:-val_count]
    val_files = csv_files[-val_count:]

    X_train_list = []
    y_train_list = []
    X_val_list = []
    y_val_list = []

    for path in train_files:
        data = trim_dataset(load_sim_dataset(path), args.max_rows_per_file)
        X_train_list.append(build_sensor_windows(scenario_sensor_matrix(data), args.window_size))
        y_train_list.append(build_targets(data))
    for path in val_files:
        data = trim_dataset(load_sim_dataset(path), args.max_rows_per_file)
        X_val_list.append(build_sensor_windows(scenario_sensor_matrix(data), args.window_size))
        y_val_list.append(build_targets(data))

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_val = np.concatenate(X_val_list, axis=0)
    y_val = np.concatenate(y_val_list, axis=0)

    config = SimNNSettings(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        input_timesteps=args.window_size,
    )

    bundle, history = train_model(
        X_train,
        y_train,
        X_val,
        y_val,
        config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    bundle.save(args.model_out)

    predictor = SimForcePredictor(bundle, device=args.device)
    train_pred = predictor.predict(X_train)
    val_pred = predictor.predict(X_val)
    train_rmse = float(np.sqrt(np.mean((train_pred - y_train) ** 2)))
    val_rmse = float(np.sqrt(np.mean((val_pred - y_val) ** 2)))

    print(f"Saved model to {args.model_out}")
    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    print(f"Train target RMSE: {train_rmse:.6f}")
    print(f"Val target RMSE:   {val_rmse:.6f}")
    print(f"Final train loss:  {history['train_loss'][-1]:.6f}")
    print(f"Final val loss:    {history['val_loss'][-1]:.6f}")


if __name__ == "__main__":
    main()
