import argparse

import numpy as np
from filterpy.kalman import ExtendedKalmanFilter

from vehicle_filter_common import load_vehicle_sim_prepared_data
from vehicle_filter_common import reconstruct_path
from vehicle_filter_common import rmse
from vehicle_filter_common import save_vehicle_sim_plot
from vehicle_filter_common import vehicle_F_jacobian
from vehicle_filter_common import vehicle_fx
from vehicle_filter_common import vehicle_hx
from vehicle_filter_common import vehicle_H_jacobian
from vehicle_filter_common import vehicle_dynamic_ekf_fx
from vehicle_filter_common import vehicle_dynamic_F_jacobian


def run_vehicle_ekf(data_file: str, plot_file: str) -> None:
    data = load_vehicle_sim_prepared_data(data_file)

    ekf = ExtendedKalmanFilter(dim_x=2, dim_z=1)
    ekf.x = np.array([data.beta_true_rad[0], data.yaw_rate_rps[0]], dtype=float)
    ekf.P = np.diag([0.02**2, 0.05**2])
    ekf.Q = np.diag([0.002**2, 0.03**2])
    ekf.R = np.array([[0.03**2]], dtype=float)
    '''

    estimates = [ekf.x.copy()]

    for i in range(1, len(data.time_s)):
        dt = data.time_s[i] - data.time_s[i - 1]
        F = vehicle_F_jacobian(dt)
        ekf.x = vehicle_fx(
            ekf.x,
            dt,
            float(data.ay_mps2[i]),
            float(data.yaw_acc_rps2[i]),
            float(data.vx_mps[i]),
        )
        ekf.P = F @ ekf.P @ F.T + ekf.Q

        z = np.array([data.yaw_rate_rps[i]], dtype=float)
        ekf.update(
            z,
            lambda _: vehicle_H_jacobian(),
            lambda x: vehicle_hx(x),
        )

        estimates.append(ekf.x.copy())
    '''
    #start
    # Extract physical constants from data object
    cf_val = float(data.cf_nprad)
    cr_val = float(data.cr_nprad)
    m_val = float(data.m_kg)
    iz_val = float(data.iz_kgm2)
    lf_val = float(data.lf_m)
    lr_val = float(data.lr_m)

    estimates = [ekf.x.copy()]
    max_iterations = 5
    epsilon = 1e-5

    for i in range(1, len(data.time_s)):
        dt = data.time_s[i] - data.time_s[i-1]
        
        # 1. PREDICT STEP (Using the new dynamic physics function)
        ekf.x = vehicle_dynamic_ekf_fx(
            ekf.x, dt, float(data.delta_rad[i]), float(data.vx_mps[i]),
            cf_val, cr_val, m_val, iz_val, lf_val, lr_val
        )
        # Compute dynamic Jacobian transition matrix
        F = vehicle_dynamic_F_jacobian(dt, float(data.vx_mps[i]), cf_val, cr_val, m_val, iz_val, lf_val, lr_val)
        ekf.P = F @ ekf.P @ F.T + ekf.Q
        
        # Save prior values before entering the iteration loop
        x_prior = ekf.x.copy()
        P_prior = ekf.P.copy()
        
        # 2. ITERATED UPDATE LOOP
        x_iter = ekf.x.copy()
        z = np.array([data.yaw_rate_rps[i]], dtype=float)
        H = vehicle_H_jacobian() # Constant matrix [[0.0, 1.0]]
        
        for j in range(max_iterations):
            # Compute Innovation using current iteration guess
            innovation = z - vehicle_hx(x_iter)
            
            # Compute iterated Kalman Gain matrix [cite: 1078]
            S = H @ P_prior @ H.T + ekf.R
            K = P_prior @ H.T @ np.linalg.inv(S)
            
            # Update state estimate for this iteration step
            x_next = x_prior + K @ (innovation - H @ (x_prior - x_iter))
            
            # Check convergence threshold
            if np.linalg.norm(x_next - x_iter) < epsilon:
                x_iter = x_next
                break
            x_iter = x_next
            
        # Finalize the iterated state values and update error covariance P matrix [cite: 1080]
        ekf.x = x_iter
        S_final = H @ P_prior @ H.T + ekf.R
        K_final = P_prior @ H.T @ np.linalg.inv(S_final)
        ekf.P = (np.eye(2) - K_final @ H) * P_prior
        
        estimates.append(ekf.x.copy())
    #end

    estimates = np.asarray(estimates)
    beta_rmse = rmse(data.beta_true_rad, estimates[:, 0])
    yaw_rate_rmse = rmse(data.yaw_rate_rps, estimates[:, 1])
    path_x_est, path_y_est, _ = reconstruct_path(data.time_s, data.vx_mps, estimates[:, 0], estimates[:, 1], data.yaw_true_rad[0])
    path_rmse = rmse(np.hypot(data.global_x_m, data.global_y_m), np.hypot(path_x_est, path_y_est))

    print(f"Vehicle EKF beta RMSE: {beta_rmse:.6f} rad")
    print(f"Vehicle EKF yaw-rate RMSE: {yaw_rate_rmse:.6f} rad/s")
    print(f"Vehicle EKF path RMSE: {path_rmse:.6f} m")

    save_vehicle_sim_plot(data, estimates[:, 0], estimates[:, 1], plot_file, "Vehicle EKF")
    print(f"Saved plot to {plot_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EKF on prepared vehicle simulation data.")
    parser.add_argument("--data", default="vehicle_sim_data.npz", help="Prepared NPZ data file.")
    parser.add_argument("--plot", default="vehicle_ekf.png", help="Output plot filename.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_vehicle_ekf(args.data, args.plot)
