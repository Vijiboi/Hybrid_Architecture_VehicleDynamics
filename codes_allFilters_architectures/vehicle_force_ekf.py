'''
import argparse

import numpy as np

from force_transformer_common import axle_force_rmse_from_wheels
from force_transformer_common import clip_vehicle_state
from force_transformer_common import ForceModelBundle
from force_transformer_common import ForcePredictor
from force_transformer_common import numerical_state_jacobian
from force_transformer_common import predict_forces_for_prepared_data
from force_transformer_common import save_force_hybrid_plot
from force_transformer_common import VehicleParams
from vehicle_filter_common import load_vehicle_sim_prepared_data
from vehicle_filter_common import rmse


def clip_twintrack_state(state: np.ndarray) -> np.ndarray:
    clipped = state.copy()
    clipped[0] = float(np.clip(clipped[0], -0.45, 0.45))   # beta
    clipped[1] = float(np.clip(clipped[1], -1.8, 1.8))     # yaw rate
    clipped[2] = float(np.clip(clipped[2], -4000.0, 4000.0))  # total lateral-force correction
    clipped[3] = float(np.clip(clipped[3], -4.0, 4.0))     # lateral-acceleration bias
    return clipped

#start
def twintrack_force_sums(wheel_forces_n: np.ndarray, delta_rad: float) -> tuple[float, float]:
    # Check if the predictor is returning 2 elements (axle forces) or 4 elements (individual wheels)
    if len(wheel_forces_n) == 2:
        fy_front_total, fy_rear_total = wheel_forces_n[0], wheel_forces_n[1]
        front_body_lat = fy_front_total * np.cos(delta_rad)
        rear_body_lat = fy_rear_total
    else:
        fy_fl, fy_fr, fy_rl, fy_rr = [float(v) for v in wheel_forces_n]
        front_body_lat = (fy_fl + fy_fr) * np.cos(delta_rad)
        rear_body_lat = fy_rl + fy_rr
        
    return front_body_lat, rear_body_lat
#end


def twintrack_process_derivative(
    state: np.ndarray,
    vx_mps: float,
    delta_rad: float,
    wheel_forces_n: np.ndarray,
    params: VehicleParams,
) -> np.ndarray:
    beta_rad, yaw_rate_rps, delta_fy_bias_n, ay_bias_mps2 = state
    del ay_bias_mps2
    vx_safe = max(abs(float(vx_mps)), 0.5)
    front_lat, rear_lat = twintrack_force_sums(wheel_forces_n, delta_rad)
    total_lat = front_lat + rear_lat + float(delta_fy_bias_n)

    beta_dot = total_lat / (params.m_kg * vx_safe) - float(yaw_rate_rps)
    yaw_rate_dot = (params.lf_m * front_lat - params.lr_m * rear_lat) / params.iz_kgm2
    delta_fy_bias_dot = 0.0
    ay_bias_dot = 0.0
    return np.array([beta_dot, yaw_rate_dot, delta_fy_bias_dot, ay_bias_dot], dtype=float)


def twintrack_ekf_fx(
    state: np.ndarray,
    dt: float,
    vx_mps: float,
    delta_rad: float,
    wheel_forces_n: np.ndarray,
    params: VehicleParams,
) -> np.ndarray:
    return clip_twintrack_state(state + dt * twintrack_process_derivative(state, vx_mps, delta_rad, wheel_forces_n, params))


def twintrack_ekf_hx(
    state: np.ndarray,
    delta_rad: float,
    wheel_forces_n: np.ndarray,
    params: VehicleParams,
) -> np.ndarray:
    beta_rad, yaw_rate_rps, delta_fy_bias_n, ay_bias_mps2 = state
    del beta_rad
    front_lat, rear_lat = twintrack_force_sums(wheel_forces_n, delta_rad)
    total_lat = front_lat + rear_lat + float(delta_fy_bias_n)
    ay_pred = total_lat / params.m_kg + float(ay_bias_mps2)
    return np.array([float(yaw_rate_rps), ay_pred], dtype=float)


def run_vehicle_force_ekf(data_file: str, model_file: str, plot_file: str, device: str = "cpu") -> None:
    data = load_vehicle_sim_prepared_data(data_file)
    params = VehicleParams.from_prepared_data(data)
    bundle = ForceModelBundle.load(model_file, device=device)
    predictor = ForcePredictor(bundle, device=device)
    wheel_force_pred = predict_forces_for_prepared_data(predictor, data)

    x = np.array([data.beta_true_rad[0], data.yaw_rate_rps[0], 0.0, 0.0], dtype=float)
    P = np.diag([0.02**2, 0.05**2, 400.0**2, 0.20**2])
    Q = np.diag([0.003**2, 0.04**2, 80.0**2, 0.03**2])
    #Q = np.diag([0.01**2, 0.08**2, 200.0**2, 0.1**2])
    R = np.diag([0.02**2, 0.20**2])
    #R = np.diag([0.005**2, 0.01**2])
    estimates = [x.copy()]

    for i in range(1, len(data.time_s)):
        dt = float(data.time_s[i] - data.time_s[i - 1])
        vx_mps = float(data.vx_mps[i])
        delta_rad = float(data.delta_rad[i])
        wheel_forces_n = wheel_force_pred[i]

        fx_local = lambda state: twintrack_ekf_fx(state, dt, vx_mps, delta_rad, wheel_forces_n, params)
        x_pred = fx_local(x)
        F = numerical_state_jacobian(fx_local, x)
        P_pred = F @ P @ F.T + Q

        hx_local = lambda state: twintrack_ekf_hx(state, delta_rad, wheel_forces_n, params)
        z = np.array([float(data.yaw_rate_rps[i]), float(data.ay_mps2[i])], dtype=float)
        z_pred = hx_local(x_pred)
        H = numerical_state_jacobian(hx_local, x_pred)
        innovation = z - z_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        x = clip_twintrack_state(x_pred + K @ innovation)
        P = (np.eye(4) - K @ H) @ P_pred
        P = 0.5 * (P + P.T)
        estimates.append(x.copy())

    estimates = np.asarray(estimates)
    beta_rmse = rmse(data.beta_true_rad, estimates[:, 0])
    yaw_rate_rmse = rmse(data.yaw_rate_rps, estimates[:, 1])
    axle_force_fit_rmse = axle_force_rmse_from_wheels(data, wheel_force_pred)

    print(f"Twin-track EKF beta RMSE: {beta_rmse:.6f} rad")
    print(f"Twin-track EKF yaw-rate RMSE: {yaw_rate_rmse:.6f} rad/s")
    print(f"Wheel-force model axle RMSE on this scenario: {axle_force_fit_rmse:.6f} N")
    print(f"Final delta_Fy_bias estimate: {float(estimates[-1, 2]):.3f} N")
    print(f"Final ay_bias estimate: {float(estimates[-1, 3]):.3f} m/s^2")

    save_force_hybrid_plot(data, estimates[:, 0], estimates[:, 1], wheel_force_pred, plot_file, "Twin-track Force-NN EKF")
    print(f"Saved plot to {plot_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the twin-track force-driven EKF on prepared vehicle simulation data.")
    parser.add_argument("--data", default="vehicle_sim_data.npz", help="Prepared NPZ data file.")
    parser.add_argument("--model", default="force_transformer_model.pt", help="Trained wheel-force transformer model.")
    parser.add_argument("--plot", default="vehicle_force_ekf.png", help="Output plot filename.")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_vehicle_force_ekf(args.data, args.model, args.plot, device=args.device)
'''
# vehicle_force_ekf.py
import argparse
import numpy as np
from force_transformer_common import (
    axle_force_rmse_from_wheels,
    clip_vehicle_state,
    ForceModelBundle,
    ForcePredictor,
    numerical_state_jacobian,
    predict_forces_for_prepared_data,
    save_force_hybrid_plot,
    VehicleParams
)
from vehicle_filter_common import load_vehicle_sim_prepared_data, rmse

def clip_twintrack_state(state: np.ndarray) -> np.ndarray:
    clipped = state.copy()
    clipped[0] = float(np.clip(clipped[0], -0.45, 0.45))         # Sideslip beta (rad)
    clipped[1] = float(np.clip(clipped[1], -1.8, 1.8))           # Yaw rate (rad/s)
    clipped[2] = float(np.clip(clipped[2], -4000.0, 4000.0))     # Lumped lateral force bias (N)
    clipped[3] = float(np.clip(clipped[3], -4.0, 4.0))           # IMU ay bias (m/s^2)
    return clipped

def twintrack_force_sums(wheel_forces_n: np.ndarray, delta_rad: float) -> tuple[float, float]:
    """
    Transforms 4 independent wheel-level forces into body-frame lumped axle forces
    by resolving front wheel steering vectors.
    """
    fy_fl, fy_fr, fy_rl, fy_rr = [float(v) for v in wheel_forces_n]
    front_body_lat = (fy_fl + fy_fr) * np.cos(delta_rad)
    rear_body_lat = fy_rl + fy_rr
    return front_body_lat, rear_body_lat

def twintrack_process_derivative(
    state: np.ndarray,
    vx_mps: float,
    delta_rad: float,
    wheel_forces_n: np.ndarray,
    params: VehicleParams,
) -> np.ndarray:
    beta_rad, yaw_rate_rps, delta_fy_bias_n, ay_bias_mps2 = state
    vx_safe = max(abs(float(vx_mps)), 0.5)
    
    # Extract lumped body-centric forces from network outputs
    front_lat, rear_lat = twintrack_force_sums(wheel_forces_n, delta_rad)
    
    # Apply bias estimation on total lateral forces for model robustness
    total_lat = front_lat + rear_lat + float(delta_fy_bias_n)
    
    # State derivative propagation
    beta_dot = total_lat / (params.m_kg * vx_safe) - float(yaw_rate_rps)
    yaw_rate_dot = (params.lf_m * front_lat - params.lr_m * rear_lat) / params.iz_kgm2
    delta_fy_bias_dot = 0.0
    ay_bias_dot = 0.0
    
    return np.array([beta_dot, yaw_rate_dot, delta_fy_bias_dot, ay_bias_dot], dtype=float)

def twintrack_ekf_fx(
    state: np.ndarray,
    dt: float,
    vx_mps: float,
    delta_rad: float,
    wheel_forces_n: np.ndarray,
    params: VehicleParams,
) -> np.ndarray:
    return clip_twintrack_state(
        state + dt * twintrack_process_derivative(state, vx_mps, delta_rad, wheel_forces_n, params)
    )

def twintrack_ekf_hx(
    state: np.ndarray,
    delta_rad: float,
    wheel_forces_n: np.ndarray,
    params: VehicleParams,
) -> np.ndarray:
    beta_rad, yaw_rate_rps, delta_fy_bias_n, ay_bias_mps2 = state
    
    front_lat, rear_lat = twintrack_force_sums(wheel_forces_n, delta_rad)
    total_lat = front_lat + rear_lat + float(delta_fy_bias_n)
    
    # Observable measurement projection: [Yaw Rate, Measured Lateral Acceleration]
    ay_pred = total_lat / params.m_kg + float(ay_bias_mps2)
    return np.array([float(yaw_rate_rps), ay_pred], dtype=float)

def run_vehicle_force_ekf(data_file: str, model_file: str, plot_file: str, device: str = "cpu") -> None:
    data = load_vehicle_sim_prepared_data(data_file)
    params = VehicleParams.from_prepared_data(data)
    
    bundle = ForceModelBundle.load(model_file, device=device)
    predictor = ForcePredictor(bundle, device=device)
    
    # Continuous network evaluation along the dataset timeline
    wheel_force_pred = predict_forces_for_prepared_data(predictor, data)
    
    # State vector initialization using standard sensor boundaries
    x = np.array([data.beta_true_rad[0], data.yaw_rate_rps[0], 0.0, 0.0], dtype=float)
    P = np.diag([0.02**2, 0.05**2, 400.0**2, 0.20**2])
    Q = np.diag([0.003**2, 0.04**2, 10.0**2, 0.01**2])
    R = np.diag([0.02**2, 0.20**2])
    
    estimates = [x.copy()]
    
    for i in range(1, len(data.time_s)):
        dt = float(data.time_s[i] - data.time_s[i - 1])
        vx_mps = float(data.vx_mps[i])
        delta_rad = float(data.delta_rad[i])
        wheel_forces_n = wheel_force_pred[i]
        
        # --- Time Update / Prediction ---
        fx_local = lambda s: twintrack_ekf_fx(s, dt, vx_mps, delta_rad, wheel_forces_n, params)
        x_pred = fx_local(x)
        F = numerical_state_jacobian(fx_local, x)
        P_pred = F @ P @ F.T + Q
        
        # --- Measurement Update / Correction ---
        hx_local = lambda s: twintrack_ekf_hx(s, delta_rad, wheel_forces_n, params)
        z = np.array([float(data.yaw_rate_rps[i]), float(data.ay_mps2[i])], dtype=float)
        z_pred = hx_local(x_pred)
        H = numerical_state_jacobian(hx_local, x_pred)
        
        innovation = z - z_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        
        x = clip_twintrack_state(x_pred + K @ innovation)
        P = (np.eye(4) - K @ H) @ P_pred
        P = 0.5 * (P + P.T)  # Enforce numerical symmetry
        
        estimates.append(x.copy())
        
    estimates = np.asarray(estimates)
    
    # Compute system errors relative to ground truth validation states
    beta_rmse = rmse(data.beta_true_rad, estimates[:, 0])
    yaw_rate_rmse = rmse(data.yaw_rate_rps, estimates[:, 1])
    axle_force_fit_rmse = axle_force_rmse_from_wheels(data, wheel_force_pred)
    
    print(f"Twin-track EKF beta RMSE: {beta_rmse:.6f} rad")
    print(f"Twin-track EKF yaw-rate RMSE: {yaw_rate_rmse:.6f} rad/s")
    print(f"Wheel-force model axle RMSE on this scenario: {axle_force_fit_rmse:.6f} N")
    print(f"Final delta_Fy_bias estimate: {float(estimates[-1, 2]):.3f} N")
    print(f"Final ay_bias estimate: {float(estimates[-1, 3]):.3f} m/s^2")
    
    save_force_hybrid_plot(data, estimates[:, 0], estimates[:, 1], wheel_force_pred, plot_file, "Twin-track Force-NN EKF")
    print(f"Saved plot to {plot_file}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the twin-track force-driven EKF on prepared vehicle simulation data.")
    parser.add_argument("--data", default="vehicle_sim_data.npz", help="Prepared NPZ data file.")
    parser.add_argument("--model", default="force_transformer_model.pt", help="Trained wheel-force transformer model.")
    parser.add_argument("--plot", default="vehicle_force_ekf.png", help="Output plot filename.")
    parser.add_argument("--device", default="cpu", help="Torch device reference.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_vehicle_force_ekf(args.data, args.model, args.plot, device=args.device)
