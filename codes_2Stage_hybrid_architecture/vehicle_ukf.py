import argparse

import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter

from vehicle_filter_common import load_vehicle_sim_prepared_data
from vehicle_filter_common import reconstruct_path
from vehicle_filter_common import rmse
from vehicle_filter_common import save_vehicle_sim_plot
from vehicle_filter_common import vehicle_fx
from vehicle_filter_common import vehicle_hx
from vehicle_filter_common import vehicle_dynamic_fx


def stabilize_covariance(P: np.ndarray) -> np.ndarray:
    P = 0.5 * (P + P.T)
    smallest = float(np.min(np.linalg.eigvalsh(P)))
    if smallest < 1e-8:
        P = P + np.eye(P.shape[0]) * (1e-8 - smallest)
    return P


def run_vehicle_ukf(data_file: str, plot_file: str) -> None:
    data = load_vehicle_sim_prepared_data(data_file)

    points = MerweScaledSigmaPoints(n=2, alpha=0.3, beta=2.0, kappa=0.0)
    ukf = UnscentedKalmanFilter(dim_x=2, dim_z=1, dt=float(np.median(np.diff(data.time_s))), fx=vehicle_fx, hx=vehicle_hx, points=points)
    ukf.x = np.array([data.beta_true_rad[0], data.yaw_rate_rps[0]], dtype=float)
    ukf.P = np.diag([0.02**2, 0.05**2])
    ukf.Q = np.diag([0.002**2, 0.03**2])
    ukf.R = np.array([[0.03**2]], dtype=float)

    estimates = [ukf.x.copy()]
    
    for i in range(1, len(data.time_s)):
        dt = data.time_s[i] - data.time_s[i - 1]

        ukf.P = stabilize_covariance(ukf.P)
        ukf.predict(
            dt=dt,
            ay_input_mps2=float(data.ay_mps2[i]),
            yaw_acc_input_rps2=float(data.yaw_acc_rps2[i]),
            vx_mps=float(data.vx_mps[i]),
        )
        ukf.P = stabilize_covariance(ukf.P)

        z = np.array([data.yaw_rate_rps[i]], dtype=float)
        ukf.update(
            z,
        )
        ukf.P = stabilize_covariance(ukf.P)
        estimates.append(ukf.x.copy())
    '''
    #start
    cf_val = float(data.cf_nprad)
    cr_val = float(data.cr_nprad)
    m_val = float(data.m_kg)
    iz_val = float(data.iz_kgm2)
    lf_val = float(data.lf_m)
    lr_val = float(data.lr_m)

    ukf.fx = lambda state, dt, delta_rad, vx_mps: vehicle_dynamic_fx(
        state, dt, delta_rad, vx_mps, 
        cf=cf_val, cr=cr_val, m=m_val, iz=iz_val, lf=lf_val, lr=lr_val
    )

    estimates = [ukf.x.copy()]
    max_iterations = 5
    epsilon = 1e-5

    for i in range(1, len(data.time_s)):
        dt = data.time_s[i] - data.time_s[i-1]
        
        # 1. Standard Predict step
        ukf.P = stabilize_covariance(ukf.P)
        ukf.predict(
            dt=dt,
            delta_rad=float(data.delta_rad[i]),
            vx_mps=float(data.vx_mps[i])
        )
        
        # Save absolute snapshots after prediction phase
        x_prior = ukf.x.copy()
        P_prior = ukf.P.copy()
        
        # Initialize our local iterative state vector
        x_iter = ukf.x.copy()
        z = np.array([data.yaw_rate_rps[i]], dtype=float)
        
        # 2. Iterative Refinement Phase
        for j in range(max_iterations):
            # CRITICAL FIX: Always re-initialize the filter variables back 
            # to the step's prior snapshot to maintain perfect SPD properties
            ukf.x = x_iter.copy()
            ukf.P = stabilize_covariance(P_prior.copy())
            
            # Execute measurement update
            ukf.update(z)
            
            # Check convergence between this iteration and the last
            if np.linalg.norm(ukf.x - x_iter) < epsilon:
                x_iter = ukf.x.copy()
                break
                
            x_iter = ukf.x.copy()
            
        # Finalize step with stabilized results
        ukf.x = x_iter
        ukf.P = stabilize_covariance(ukf.P)
        estimates.append(ukf.x.copy()) 
    #end
    '''
    estimates = np.asarray(estimates)
    beta_rmse = rmse(data.beta_true_rad, estimates[:, 0])
    yaw_rate_rmse = rmse(data.yaw_rate_rps, estimates[:, 1])
    path_x_est, path_y_est, _ = reconstruct_path(data.time_s, data.vx_mps, estimates[:, 0], estimates[:, 1], data.yaw_true_rad[0])
    path_rmse = rmse(np.hypot(data.global_x_m, data.global_y_m), np.hypot(path_x_est, path_y_est))

    print(f"Vehicle UKF beta RMSE: {beta_rmse:.6f} rad")
    print(f"Vehicle UKF yaw-rate RMSE: {yaw_rate_rmse:.6f} rad/s")
    print(f"Vehicle UKF path RMSE: {path_rmse:.6f} m")

    save_vehicle_sim_plot(data, estimates[:, 0], estimates[:, 1], plot_file, "Vehicle UKF")
    print(f"Saved plot to {plot_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UKF on prepared vehicle simulation data.")
    parser.add_argument("--data", default="vehicle_sim_data.npz", help="Prepared NPZ data file.")
    parser.add_argument("--plot", default="vehicle_ukf.png", help="Output plot filename.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_vehicle_ukf(args.data, args.plot)