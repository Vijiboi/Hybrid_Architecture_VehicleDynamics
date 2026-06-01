import argparse
import numpy as np
from filterpy.kalman import KalmanFilter
from vehicle_filter_common import (
    load_vehicle_sim_prepared_data, 
    reconstruct_path, 
    rmse, 
    save_vehicle_sim_plot, 
    vehicle_F_jacobian
)

def run_vehicle_kf(data_file: str, plot_file: str) -> None:
    # Load data using your standard loader [cite: 102, 360]
    data = load_vehicle_sim_prepared_data(data_file)
    
    # 1. Initialize KF: dim_x=2 (beta, yaw_rate), dim_z=1 (measuring yaw_rate) [cite: 103]
    kf = KalmanFilter(dim_x=2, dim_z=1)
    
    # 2. Initial State [cite: 104]
    kf.x = np.array([data.beta_true_rad[0], data.yaw_rate_rps[0]], dtype=float)
    
    # 3. Covariance Matrices (The Q and R you asked about)
    # P: Initial uncertainty [cite: 105, 290]
    kf.P = np.diag([0.02**2, 0.05**2])
    
    # Q: Process Noise - how much the 'physics' might drift [cite: 106, 291]
    kf.Q = np.diag([0.002**2, 0.03**2])
    
    # R: Measurement Noise - variance of the yaw rate sensor 
    kf.R = np.array([[0.03**2]], dtype=float)
    
    # H: Measurement Function (Linear: we only measure the second state variable) [cite: 408]
    kf.H = np.array([[0.0, 1.0]], dtype=float)

    estimates = []
    estimates.append(kf.x.copy()) # [cite: 108]

    for i in range(1, len(data.time_s)):
        dt = data.time_s[i] - data.time_s[i-1] # [cite: 110]
        
        # 4. Update Transition Matrix F [cite: 111, 401]
        kf.F = vehicle_F_jacobian(dt)
        
        # 5. Define Control (Linear Input)
        vx_safe = max(abs(float(data.vx_mps[i])), 0.5) # [cite: 381]
        
        # B maps inputs [ay, yaw_acc] to the change in [beta, yaw_rate]
        kf.B = np.array([
            [dt / vx_safe, 0],
            [0,            dt]
        ], dtype=float)
        
        u = np.array([float(data.ay_mps2[i]), float(data.yaw_acc_rps2[i])]) # [cite: 115, 116]
        
        # 6. Predict step (Linear: x = Fx + Bu)
        kf.predict(u=u)
        
        # 7. Update step (Linear: use the yaw rate measurement)
        z = np.array([data.yaw_rate_rps[i]], dtype=float) # [cite: 120]
        kf.update(z)
        
        estimates.append(kf.x.copy()) # [cite: 125]

    # Post-processing (Standard for your scripts) [cite: 126-137]
    estimates = np.asarray(estimates)
    beta_rmse = rmse(data.beta_true_rad, estimates[:, 0])
    yaw_rate_rmse = rmse(data.yaw_rate_rps, estimates[:, 1])
    
    # Reconstruct path for plotting [cite: 129, 409]
    path_x_est, path_y_est, _ = reconstruct_path(
        data.time_s, data.vx_mps, estimates[:, 0],
        estimates[:, 1], data.yaw_true_rad[0]
    )
    
    print(f"Vehicle KF beta RMSE: {beta_rmse:.6f} rad")
    print(f"Vehicle KF yaw-rate RMSE: {yaw_rate_rmse:.6f} rad/s")
    
    save_vehicle_sim_plot(data, estimates[:, 0], estimates[:, 1], plot_file, "Vehicle KF") # [cite: 136]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Linear KF on prepared vehicle simulation data.")
    parser.add_argument("--data", default="vehicle_sim_data.npz", help="Prepared NPZ data file.")
    parser.add_argument("--plot", default="vehicle_kf.png", help="Output plot filename.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_vehicle_kf(args.data, args.plot)