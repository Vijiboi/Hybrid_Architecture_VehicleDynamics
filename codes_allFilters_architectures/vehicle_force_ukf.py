import argparse

import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter

from force_transformer_common import clip_vehicle_state
from force_transformer_common import ForceModelBundle
from force_transformer_common import ForcePredictor
from force_transformer_common import predict_forces_for_prepared_data
from force_transformer_common import save_force_hybrid_plot
from force_transformer_common import VehicleParams
from force_transformer_common import force_driven_fx
from vehicle_filter_common import load_vehicle_sim_prepared_data
from vehicle_filter_common import rmse


def stabilize_covariance(P: np.ndarray) -> np.ndarray:
    P = 0.5 * (P + P.T)
    smallest = float(np.min(np.linalg.eigvalsh(P)))
    if smallest < 1e-8:
        P = P + np.eye(P.shape[0]) * (1e-8 - smallest)
    return P


def run_vehicle_force_ukf(data_file: str, model_file: str, plot_file: str, device: str = "cpu") -> None:
    data = load_vehicle_sim_prepared_data(data_file)
    params = VehicleParams.from_prepared_data(data)
    bundle = ForceModelBundle.load(model_file, device=device)
    predictor = ForcePredictor(bundle, device=device)
    force_pred = predict_forces_for_prepared_data(predictor, data)

    points = MerweScaledSigmaPoints(n=2, alpha=0.3, beta=2.0, kappa=0.0)
    ukf = UnscentedKalmanFilter(
        dim_x=2,
        dim_z=1,
        dt=float(np.median(np.diff(data.time_s))),
        fx=lambda state, dt, vx_mps, fyf_n, fyr_n, params: force_driven_fx(state, dt, vx_mps, fyf_n, fyr_n, params),
        hx=lambda state: np.array([float(state[1])], dtype=float),
        points=points,
    )
    ukf.x = np.array([data.beta_true_rad[0], data.yaw_rate_rps[0]], dtype=float)
    ukf.P = np.diag([0.02**2, 0.05**2])
    ukf.Q = np.diag([0.002**2, 0.03**2])
    ukf.R = np.array([[0.02**2]], dtype=float)

    estimates = [ukf.x.copy()]

    for i in range(1, len(data.time_s)):
        dt = float(data.time_s[i] - data.time_s[i - 1])
        vx_mps = float(data.vx_mps[i])
        fyf_n = float(force_pred[i, 0])
        fyr_n = float(force_pred[i, 1])

        ukf.P = stabilize_covariance(ukf.P)
        ukf.predict(dt=dt, vx_mps=vx_mps, fyf_n=fyf_n, fyr_n=fyr_n, params=params)
        ukf.x = clip_vehicle_state(ukf.x)
        ukf.P = stabilize_covariance(ukf.P)

        z = np.array([float(data.yaw_rate_rps[i])], dtype=float)
        ukf.update(z)
        ukf.x = clip_vehicle_state(ukf.x)
        ukf.P = stabilize_covariance(ukf.P)
        estimates.append(ukf.x.copy())

    estimates = np.asarray(estimates)
    beta_rmse = rmse(data.beta_true_rad, estimates[:, 0])
    yaw_rate_rmse = rmse(data.yaw_rate_rps, estimates[:, 1])
    force_fit_rmse = float(np.sqrt(np.mean((force_pred - np.column_stack([data.fyf_true_n, data.fyr_true_n])) ** 2)))

    print(f"Force UKF beta RMSE: {beta_rmse:.6f} rad")
    print(f"Force UKF yaw-rate RMSE: {yaw_rate_rmse:.6f} rad/s")
    print(f"Force-model RMSE on this scenario: {force_fit_rmse:.6f} N")

    save_force_hybrid_plot(data, estimates[:, 0], estimates[:, 1], force_pred, plot_file, "Force-NN UKF")
    print(f"Saved plot to {plot_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the force-driven UKF on prepared vehicle simulation data.")
    parser.add_argument("--data", default="vehicle_sim_data.npz", help="Prepared NPZ data file.")
    parser.add_argument("--model", default="force_transformer_model.pt", help="Trained force transformer model.")
    parser.add_argument("--plot", default="vehicle_force_ukf.png", help="Output plot filename.")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_vehicle_force_ukf(args.data, args.model, args.plot, device=args.device)
