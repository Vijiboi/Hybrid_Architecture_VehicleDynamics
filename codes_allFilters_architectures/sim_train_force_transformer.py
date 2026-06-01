'''
import argparse
from pathlib import Path

import numpy as np

from force_transformer_common import ForceModelConfig
from force_transformer_common import build_force_training_arrays
from force_transformer_common import collect_scenario_files
from force_transformer_common import fit_force_transformer
from force_transformer_common import force_rmse
from force_transformer_common import ForcePredictor
from force_transformer_common import load_csv_datasets
from force_transformer_common import select_files_by_name
from vehicle_filter_common import load_vehicle_sim_prepared_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny transformer that maps causal sensor windows to tire forces.")
    parser.add_argument("--folder", default=str(Path("..") / "datasetforvehiclesimulation_csv"), help="Folder containing the simulation CSV scenarios.")
    parser.add_argument("--model-out", default="force_transformer_model.pt", help="Output PyTorch model file.")
    parser.add_argument("--window-size", type=int, default=8, help="Causal sensor window length.")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=1.0e-3, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1.0e-4, help="AdamW weight decay.")
    parser.add_argument("--d-model", type=int, default=64, help="Transformer width.")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of transformer encoder layers.")
    parser.add_argument("--dim-feedforward", type=int, default=128, help="Transformer feed-forward width.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Transformer dropout.")
    parser.add_argument("--force-clip", type=float, default=6000.0, help="Force clip applied at inference time, in Newtons.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--val-files", type=int, default=2, help="How many scenarios to keep for validation when no explicit split is given.")
    parser.add_argument("--train-files-list", nargs="*", default=None, help="Optional explicit CSV filenames for training.")
    parser.add_argument("--val-files-list", nargs="*", default=None, help="Optional explicit CSV filenames for validation.")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda.")
    return parser.parse_args()


def gather_train_val_files(args: argparse.Namespace) -> tuple[list[Path], list[Path]]:
    folder = Path(args.folder)
    scenario_files = collect_scenario_files(folder)
    if len(scenario_files) < 1:
        raise ValueError(f"No scenario CSV files found in {folder}")

    if args.train_files_list is not None or args.val_files_list is not None:
        train_files = select_files_by_name(scenario_files, args.train_files_list)
        val_files = select_files_by_name(scenario_files, args.val_files_list)
        if not train_files:
            raise ValueError("No training files selected.")
        if not val_files:
            val_files = train_files
        return train_files, val_files

    if len(scenario_files) < 2:
        raise ValueError("Need at least 2 scenarios for automatic train/validation split.")

    val_count = max(1, min(args.val_files, len(scenario_files) - 1))
    return scenario_files[:-val_count], scenario_files[-val_count:]


def main() -> None:
    args = parse_args()
    train_files, val_files = gather_train_val_files(args)

    train_datasets = load_csv_datasets(train_files)
    val_datasets = load_csv_datasets(val_files)

    X_train = np.concatenate([build_force_training_arrays(ds, args.window_size)[0] for ds in train_datasets], axis=0)
    y_train = np.concatenate([build_force_training_arrays(ds, args.window_size)[1] for ds in train_datasets], axis=0)
    X_val = np.concatenate([build_force_training_arrays(ds, args.window_size)[0] for ds in val_datasets], axis=0)
    y_val = np.concatenate([build_force_training_arrays(ds, args.window_size)[1] for ds in val_datasets], axis=0)

    config = ForceModelConfig(
        window_size=args.window_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        force_clip_n=args.force_clip,
    )
    bundle, history = fit_force_transformer(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        config=config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
    )
    bundle.save(args.model_out)
    predictor = ForcePredictor(bundle, device=args.device)
    train_pred = predictor.predict_forces(X_train)
    val_pred = predictor.predict_forces(X_val)

    print(f"Saved transformer force model to {args.model_out}")
    print(f"Training scenarios: {len(train_files)}")
    print(f"Validation scenarios: {len(val_files)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"Final val loss:   {history['val_loss'][-1]:.6f}")
    print(f"Force RMSE (train): {force_rmse(y_train, train_pred):.6f} N")
    print(f"Force RMSE (val):   {force_rmse(y_val, val_pred):.6f} N")
    print("Training files:")
    for path in train_files:
        print(f"  {path.name}")
    print("Validation files:")
    for path in val_files:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
'''
# sim_train_force_transformer.py
import argparse
from pathlib import Path
import numpy as np

from force_transformer_common import (
    ForceModelConfig,
    build_force_training_arrays,
    collect_scenario_files,
    fit_force_transformer,
    force_rmse,
    ForcePredictor,
    load_csv_datasets,
    select_files_by_name
)
from vehicle_filter_common import load_vehicle_sim_prepared_data

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny transformer that maps causal sensor windows to tire forces.")
    parser.add_argument("--folder", default=str(Path("..") / "datasetforvehiclesimulation_csv"), help="Folder containing the simulation CSV scenarios.")
    parser.add_argument("--model-out", default="force_transformer_model.pt", help="Output PyTorch model file.")
    parser.add_argument("--window-size", type=int, default=8, help="Causal sensor window length.")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=1.0e-3, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1.0e-4, help="AdamW weight decay.")
    parser.add_argument("--d-model", type=int, default=64, help="Transformer width.")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of transformer encoder layers.")
    parser.add_argument("--dim-feedforward", type=int, default=128, help="Transformer feed-forward width.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Transformer dropout.")
    parser.add_argument("--force-clip", type=float, default=6000.0, help="Force clip applied at inference time, in Newtons.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--val-files", type=int, default=2, help="How many scenarios to keep for validation when no explicit split is given.")
    parser.add_argument("--train-files-list", nargs="*", default=None, help="Optional explicit CSV filenames for training.")
    parser.add_argument("--val-files-list", nargs="*", default=None, help="Optional explicit CSV filenames for validation.")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda.")
    return parser.parse_args()

def gather_train_val_files(args: argparse.Namespace) -> tuple[list[Path], list[Path]]:
    folder = Path(args.folder)
    scenario_files = collect_scenario_files(folder)
    if len(scenario_files) < 1:
        raise ValueError(f"No scenario CSV files found in {folder}")

    if args.train_files_list is not None or args.val_files_list is not None:
        train_files = select_files_by_name(scenario_files, args.train_files_list)
        val_files = select_files_by_name(scenario_files, args.val_files_list)
        if not train_files:
            raise ValueError("No training files selected.")
        if not val_files:
            val_files = train_files
        return train_files, val_files

    if len(scenario_files) < 2:
        raise ValueError("Need at least 2 scenarios for automatic train/validation split.")

    val_count = max(1, min(args.val_files, len(scenario_files) - 1))
    return scenario_files[:-val_count], scenario_files[-val_count:]

def main() -> None:
    args = parse_args()
    train_files, val_files = gather_train_val_files(args)

    print("Loading datasets...")
    train_datasets = load_csv_datasets(train_files)
    val_datasets = load_csv_datasets(val_files)

    print("Building temporal feature structures...")
    X_train = np.concatenate([build_force_training_arrays(ds, args.window_size)[0] for ds in train_datasets], axis=0)
    y_train = np.concatenate([build_force_training_arrays(ds, args.window_size)[1] for ds in train_datasets], axis=0)
    X_val = np.concatenate([build_force_training_arrays(ds, args.window_size)[0] for ds in val_datasets], axis=0)
    y_val = np.concatenate([build_force_training_arrays(ds, args.window_size)[1] for ds in val_datasets], axis=0)

    print(f"Dataset arrays successfully created.")
    print(f"X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
    print(f"X_val shape:   {X_val.shape} | y_val shape:   {y_val.shape}")

    config = ForceModelConfig(
        window_size=args.window_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        force_clip_n=args.force_clip,
    )
    
    print("Initiating fit_force_transformer optimization loop...")
    bundle, history = fit_force_transformer(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        config=config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
    )
    
    bundle.save(args.model_out)
    predictor = ForcePredictor(bundle, device=args.device)
    train_pred = predictor.predict_forces(X_train)
    val_pred = predictor.predict_forces(X_val)

    print("\n" + "="*50)
    print(f"Saved transformer force model to {args.model_out}")
    print(f"Training scenarios: {len(train_files)}")
    print(f"Validation scenarios: {len(val_files)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"Final val loss:   {history['val_loss'][-1]:.6f}")
    print(f"Force RMSE (train): {force_rmse(y_train, train_pred):.6f} N")
    print(f"Force RMSE (val):   {force_rmse(y_val, val_pred):.6f} N")
    print("="*50)
    
    print("Training files:")
    for path in train_files:
        print(f"  {path.name}")
    print("Validation files:")
    for path in val_files:
        print(f"  {path.name}")

if __name__ == "__main__":
    main()