import argparse
from pathlib import Path

from real_sensor_common import build_sensor_windows
from real_sensor_common import load_project_configs
from real_sensor_common import load_real_sensor_csv
from real_sensor_common import model_output_to_beta_measurement
from real_sensor_common import model_output_to_wheel_forces
from real_sensor_common import RealSensorForceModelBundle
from real_sensor_common import RealSensorForcePredictor
from real_sensor_common import run_real_sensor_force_ekf
from real_sensor_common import save_real_sensor_summary_plot
from real_sensor_common import slice_real_sensor_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the real-sensor twin-track EKF on one training/testing CSV.")
    parser.add_argument("--csv", default=str(Path("..") / "testingdata" / "data_to_run.csv"))
    parser.add_argument("--params-file", default=str(Path("..") / "params" / "parameters.toml"))
    parser.add_argument("--vehicle-config", default="real_sensor_vehicle_config.toml")
    parser.add_argument("--model", default="real_sensor_force_transformer.pt")
    parser.add_argument("--plot", default="real_sensor_force_ekf.png")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap on rows for quick EKF smoke runs.")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, vehicle, ekf_cfg = load_project_configs(args.params_file, args.vehicle_config)
    dataset = slice_real_sensor_dataset(load_real_sensor_csv(args.csv, vehicle), args.max_samples)

    bundle = RealSensorForceModelBundle.load(args.model, device=args.device)
    predictor = RealSensorForcePredictor(bundle, device=args.device)
    model_output = predictor.predict(build_sensor_windows(dataset.sensor_matrix, bundle.window_size))
    wheel_force_pred = model_output_to_wheel_forces(model_output, dataset.sensor_matrix, bundle, vehicle)
    beta_meas = model_output_to_beta_measurement(model_output)

    result = run_real_sensor_force_ekf(dataset, wheel_force_pred, vehicle, ekf_cfg, beta_meas=beta_meas)
    print(f"Beta RMSE: {float(result['beta_rmse']):.6f} rad")
    print(f"Yaw-rate RMSE: {float(result['yaw_rmse']):.6f} rad/s")
    print(f"Wheel-force RMSE: {float(result['force_rmse']):.6f} N")

    save_real_sensor_summary_plot(dataset, result, wheel_force_pred, args.plot, "Real-sensor force EKF")
    print(f"Saved plot to {args.plot}")


if __name__ == "__main__":
    main()
