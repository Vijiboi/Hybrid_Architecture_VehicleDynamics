import argparse
from pathlib import Path

import numpy as np

from sim_newarch_common import SimForceModelBundle
from sim_newarch_common import SimForcePredictor
from sim_newarch_common import build_sensor_windows
from sim_newarch_common import load_sim_dataset
from sim_newarch_common import save_summary_plot
from sim_newarch_common import scenario_sensor_matrix
from sim_newarch_common import SimVehicleParams
from sim_newarch_common import run_sim_ekf
from vehicle_sim_loader import VehicleSimDataset


def trim_dataset(data: VehicleSimDataset, max_samples: int | None) -> VehicleSimDataset:
    if max_samples is None or max_samples <= 0:
        return data
    payload: dict[str, object] = {}
    for key, value in data.__dict__.items():
        if isinstance(value, np.ndarray) and value.ndim > 0:
            payload[key] = value[:max_samples]
        else:
            payload[key] = value
    return type(data)(**payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the new simulation estimated model on one converted CSV.")
    parser.add_argument("--csv", default=str(Path("..") / "datasetforvehiclesimulation_csv" / "1_RunTimeDataset_DLC_u85_v60.csv"))
    parser.add_argument("--model", default="sim_newarch_force_transformer.pt")
    parser.add_argument("--plot", default="sim_newarch_ekf.png")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit the number of samples used, 0 means all.")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = trim_dataset(load_sim_dataset(args.csv), args.max_samples)
    params = SimVehicleParams.from_dataset(data)

    bundle = SimForceModelBundle.load(args.model, device=args.device)
    predictor = SimForcePredictor(bundle, device=args.device)
    sensor_windows = build_sensor_windows(scenario_sensor_matrix(data), bundle.config.input_timesteps)
    predicted_outputs = predictor.predict(sensor_windows)

    result = run_sim_ekf(data, predicted_outputs, params)
    print(f"Beta RMSE: {float(result['beta_rmse']):.6f} rad")
    print(f"Yaw-rate RMSE: {float(result['yaw_rmse']):.6f} rad/s")
    print(f"Force RMSE: {float(result['force_rmse']):.6f} N")

    save_summary_plot(data, result, predicted_outputs, args.plot, "Sim new-arch EKF")
    print(f"Saved plot to {args.plot}")


if __name__ == "__main__":
    main()
