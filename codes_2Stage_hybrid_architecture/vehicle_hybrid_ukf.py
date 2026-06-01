import argparse

import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter

from hybrid_friction_common import FrictionResidualModel
from hybrid_friction_common import HybridVehicleParams
from hybrid_friction_common import build_measurement_window
from hybrid_friction_common import hybrid_vehicle_fx
from vehicle_filter_common import load_vehicle_sim_prepared_data
from vehicle_filter_common import reconstruct_path
from vehicle_filter_common import rmse
from vehicle_filter_common import save_vehicle_sim_plot


def stabilize_covariance(P: np.ndarray) -> np.ndarray:
    P = 0.5 * (P + P.T)
    smallest = float(np.min(np.linalg.eigvalsh(P)))
    if smallest < 1e-8:
        P = P + np.eye(P.shape[0]) * (1e-8 - smallest)
    return P


def clip_vehicle_state(state: np.ndarray) -> np.ndarray:
    clipped = state.copy()
    clipped[0] = float(np.clip(clipped[0], -0.35, 0.35))
    clipped[1] = float(np.clip(clipped[1], -1.5, 1.5))
    return clipped


def run_vehicle_hybrid_ukf(data_file: str, model_file: str, plot_file: str) -> None:
    data = load_vehicle_sim_prepared_data(data_file)
    params = HybridVehicleParams.from_prepared_data(data)
    model = FrictionResidualModel.load(model_file)

    points = MerweScaledSigmaPoints(n=2, alpha=0.3, beta=2.0, kappa=0.0)
    ukf = UnscentedKalmanFilter(dim_x=2, dim_z=1, dt=float(np.median(np.diff(data.time_s))), fx=hybrid_vehicle_fx, hx=lambda state: np.array([float(state[1])], dtype=float), points=points)
    ukf.x = np.array([data.beta_true_rad[0], data.yaw_rate_rps[0]], dtype=float)
    ukf.P = np.diag([0.02**2, 0.05**2])
    ukf.Q = np.diag([0.003**2, 0.05**2])
    ukf.R = np.array([[0.02**2]], dtype=float)

    estimates = [ukf.x.copy()]

    for i in range(1, len(data.time_s)):
        dt = float(data.time_s[i] - data.time_s[i - 1])
        window_flat = build_measurement_window(data, i, model.window_size)
        delta_rad = float(data.delta_rad[i])
        vx_mps = float(data.vx_mps[i])

        ukf.P = stabilize_covariance(ukf.P)
        ukf.predict(
            dt=dt,
            measurement_window_flat=window_flat,
            delta_rad=delta_rad,
            vx_mps=vx_mps,
            params=params,
            model=model,
        )
        ukf.x = clip_vehicle_state(ukf.x)
        ukf.P = stabilize_covariance(ukf.P)

        z = np.array([float(data.yaw_rate_rps[i])], dtype=float)
        ukf.update(
            z,
            hx=lambda state: np.array([float(state[1])], dtype=float),
        )
        ukf.x = clip_vehicle_state(ukf.x)
        ukf.P = stabilize_covariance(ukf.P)
        estimates.append(ukf.x.copy())

    estimates = np.asarray(estimates)
    beta_rmse = rmse(data.beta_true_rad, estimates[:, 0])
    yaw_rate_rmse = rmse(data.yaw_rate_rps, estimates[:, 1])
    path_x_est, path_y_est, _ = reconstruct_path(data.time_s, data.vx_mps, estimates[:, 0], estimates[:, 1], data.yaw_true_rad[0])
    path_rmse = rmse(np.hypot(data.global_x_m, data.global_y_m), np.hypot(path_x_est, path_y_est))

    print(f"Hybrid UKF beta RMSE: {beta_rmse:.6f} rad")
    print(f"Hybrid UKF yaw-rate RMSE: {yaw_rate_rmse:.6f} rad/s")
    print(f"Hybrid UKF path RMSE: {path_rmse:.6f} m")

    save_vehicle_sim_plot(data, estimates[:, 0], estimates[:, 1], plot_file, "Hybrid Friction UKF")
    print(f"Saved plot to {plot_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the hybrid friction-aware UKF on prepared vehicle simulation data.")
    parser.add_argument("--data", default="vehicle_sim_data.npz", help="Prepared NPZ data file.")
    parser.add_argument("--model", default="friction_residual_model.npz", help="Trained friction residual model.")
    parser.add_argument("--plot", default="vehicle_hybrid_ukf.png", help="Output plot filename.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_vehicle_hybrid_ukf(args.data, args.model, args.plot)
