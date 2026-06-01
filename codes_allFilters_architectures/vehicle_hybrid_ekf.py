import argparse

import numpy as np

from hybrid_friction_common import FrictionResidualModel
from hybrid_friction_common import HybridVehicleParams
from hybrid_friction_common import build_measurement_window
from hybrid_friction_common import hybrid_vehicle_fx
from hybrid_friction_common import numerical_state_jacobian
from vehicle_filter_common import load_vehicle_sim_prepared_data
from vehicle_filter_common import reconstruct_path
from vehicle_filter_common import rmse
from vehicle_filter_common import save_vehicle_sim_plot


def clip_vehicle_state(state: np.ndarray) -> np.ndarray:
    clipped = state.copy()
    clipped[0] = float(np.clip(clipped[0], -0.35, 0.35))
    clipped[1] = float(np.clip(clipped[1], -1.5, 1.5))
    return clipped


def run_vehicle_hybrid_ekf(data_file: str, model_file: str, plot_file: str) -> None:
    data = load_vehicle_sim_prepared_data(data_file)
    params = HybridVehicleParams.from_prepared_data(data)
    model = FrictionResidualModel.load(model_file)

    x = np.array([data.beta_true_rad[0], data.yaw_rate_rps[0]], dtype=float)
    P = np.diag([0.02**2, 0.05**2])
    Q = np.diag([0.003**2, 0.05**2])
    R = np.array([[0.02**2]], dtype=float)

    estimates = [x.copy()]

    for i in range(1, len(data.time_s)):
        dt = float(data.time_s[i] - data.time_s[i - 1])
        window_flat = build_measurement_window(data, i, model.window_size)
        delta_rad = float(data.delta_rad[i])
        vx_mps = float(data.vx_mps[i])

        fx_local = lambda state: hybrid_vehicle_fx(
            state,
            dt,
            measurement_window_flat=window_flat,
            delta_rad=delta_rad,
            vx_mps=vx_mps,
            params=params,
            model=model,
        )
        x_pred = clip_vehicle_state(fx_local(x))
        F = numerical_state_jacobian(fx_local, x)
        P_pred = F @ P @ F.T + Q

        hx_local = lambda state: np.array([float(state[1])], dtype=float)
        z = np.array([float(data.yaw_rate_rps[i])], dtype=float)
        z_pred = hx_local(x_pred)
        H = numerical_state_jacobian(hx_local, x_pred)
        innovation = z - z_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        x = clip_vehicle_state(x_pred + K @ innovation)
        P = (np.eye(2) - K @ H) @ P_pred
        P = 0.5 * (P + P.T)
        estimates.append(x.copy())

    estimates = np.asarray(estimates)
    beta_rmse = rmse(data.beta_true_rad, estimates[:, 0])
    yaw_rate_rmse = rmse(data.yaw_rate_rps, estimates[:, 1])
    path_x_est, path_y_est, _ = reconstruct_path(data.time_s, data.vx_mps, estimates[:, 0], estimates[:, 1], data.yaw_true_rad[0])
    path_rmse = rmse(np.hypot(data.global_x_m, data.global_y_m), np.hypot(path_x_est, path_y_est))

    print(f"Hybrid EKF beta RMSE: {beta_rmse:.6f} rad")
    print(f"Hybrid EKF yaw-rate RMSE: {yaw_rate_rmse:.6f} rad/s")
    print(f"Hybrid EKF path RMSE: {path_rmse:.6f} m")

    save_vehicle_sim_plot(data, estimates[:, 0], estimates[:, 1], plot_file, "Hybrid Friction EKF")
    print(f"Saved plot to {plot_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the hybrid friction-aware EKF on prepared vehicle simulation data.")
    parser.add_argument("--data", default="vehicle_sim_data.npz", help="Prepared NPZ data file.")
    parser.add_argument("--model", default="friction_residual_model.npz", help="Trained friction residual model.")
    parser.add_argument("--plot", default="vehicle_hybrid_ekf.png", help="Output plot filename.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_vehicle_hybrid_ekf(args.data, args.model, args.plot)
