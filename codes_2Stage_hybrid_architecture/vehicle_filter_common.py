'''
from dataclasses import dataclass

import matplotlib
import numpy as np
from scipy.integrate import cumulative_trapezoid

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class VehicleSimPreparedData:
    time_s: np.ndarray
    vx_mps: np.ndarray
    vy_mps: np.ndarray
    delta_rad: np.ndarray
    ay_mps2: np.ndarray
    yaw_rate_rps: np.ndarray
    alpha_f_rad: np.ndarray
    alpha_r_rad: np.ndarray
    fyf_true_n: np.ndarray
    fyr_true_n: np.ndarray
    beta_true_rad: np.ndarray
    yaw_true_rad: np.ndarray
    global_x_m: np.ndarray
    global_y_m: np.ndarray
    yaw_acc_rps2: np.ndarray
    cf_nprad: float
    cr_nprad: float
    m_kg: float
    iz_kgm2: float
    lf_m: float
    lr_m: float


def load_vehicle_sim_prepared_data(data_file: str) -> VehicleSimPreparedData:
    data = np.load(data_file)
    return VehicleSimPreparedData(
        time_s=data["time_s"],
        vx_mps=data["vx_mps"],
        vy_mps=data["vy_mps"] if "vy_mps" in data else data["vx_mps"] * data["beta_true_rad"],
        delta_rad=data["delta_rad"],
        ay_mps2=data["ay_mps2"],
        yaw_rate_rps=data["yaw_rate_rps"],
        alpha_f_rad=data["alpha_f_rad"] if "alpha_f_rad" in data else np.zeros_like(data["time_s"]),
        alpha_r_rad=data["alpha_r_rad"] if "alpha_r_rad" in data else np.zeros_like(data["time_s"]),
        fyf_true_n=data["fyf_true_n"] if "fyf_true_n" in data else np.zeros_like(data["time_s"]),
        fyr_true_n=data["fyr_true_n"] if "fyr_true_n" in data else np.zeros_like(data["time_s"]),
        beta_true_rad=data["beta_true_rad"],
        yaw_true_rad=data["yaw_true_rad"],
        global_x_m=data["global_x_m"],
        global_y_m=data["global_y_m"],
        yaw_acc_rps2=data["yaw_acc_rps2"],
        cf_nprad=float(data["cf_nprad"][0]),
        cr_nprad=float(data["cr_nprad"][0]),
        m_kg=float(data["m_kg"][0]),
        iz_kgm2=float(data["iz_kgm2"][0]),
        lf_m=float(data["lf_m"][0]),
        lr_m=float(data["lr_m"][0]),
    )


def safe_speed(vx_mps: float) -> float:
    return float(max(abs(vx_mps), 0.5))


def bicycle_state_derivative(
    state: np.ndarray,
    ay_input_mps2: float,
    yaw_acc_input_rps2: float,
    vx_mps: float,
) -> np.ndarray:
    beta_rad, yaw_rate_rps = state
    vx_safe = safe_speed(vx_mps)
    beta_dot = (ay_input_mps2 / vx_safe) - yaw_rate_rps
    yaw_rate_dot = yaw_acc_input_rps2
    return np.array([beta_dot, yaw_rate_dot], dtype=float)


def vehicle_fx(
    state: np.ndarray,
    dt: float,
    ay_input_mps2: float,
    yaw_acc_input_rps2: float,
    vx_mps: float,
) -> np.ndarray:
    derivative = bicycle_state_derivative(state, ay_input_mps2, yaw_acc_input_rps2, vx_mps)
    return state + dt * derivative


def vehicle_F_jacobian(dt: float) -> np.ndarray:
    return np.array([[1.0, -dt], [0.0, 1.0]], dtype=float)


def vehicle_hx(
    state: np.ndarray,
) -> np.ndarray:
    return np.array([state[1]], dtype=float)


def vehicle_H_jacobian() -> np.ndarray:
    return np.array([[0.0, 1.0]], dtype=float)


def reconstruct_path(time_s: np.ndarray, vx_mps: np.ndarray, beta_rad: np.ndarray, yaw_rate_rps: np.ndarray, yaw0_rad: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yaw_est_rad = cumulative_trapezoid(yaw_rate_rps, time_s, initial=0.0) + yaw0_rad
    vy_mps = vx_mps * beta_rad
    vx_global = vx_mps * np.cos(yaw_est_rad) - vy_mps * np.sin(yaw_est_rad)
    vy_global = vx_mps * np.sin(yaw_est_rad) + vy_mps * np.cos(yaw_est_rad)
    x_m = cumulative_trapezoid(vx_global, time_s, initial=0.0)
    y_m = cumulative_trapezoid(vy_global, time_s, initial=0.0)
    return x_m, y_m, yaw_est_rad


def rmse(truth: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.sqrt(np.mean((truth - estimate) ** 2)))

#start###############

def rmse_vs_time(truth: np.ndarray, estimate: np.ndarray, times: np.ndarray) -> tuple:
    """Compute RMSE from time 0 to each time step."""
    assert len(truth) == len(estimate) == len(times)

    rmse_vals = []
    for i in range(1, len(truth) + 1):
        rmse_vals.append(
            float(np.sqrt(np.mean((truth[:i] - estimate[:i]) ** 2)))
        )

    return np.array(rmse_vals), times[: len(rmse_vals)]
#end###############

def save_vehicle_sim_plot(
    data: VehicleSimPreparedData,
    beta_est_rad: np.ndarray,
    yaw_rate_est_rps: np.ndarray,
    plot_file: str,
    title: str,
) -> None:
    path_x_est, path_y_est, _ = reconstruct_path(
        data.time_s, data.vx_mps, beta_est_rad, yaw_rate_est_rps, data.yaw_true_rad[0]
    )

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    axes[0].plot(data.time_s, data.beta_true_rad, "k", linewidth=2, label="Beta true")
    axes[0].plot(data.time_s, beta_est_rad, label="Beta estimate")
    axes[0].set_title(f"{title}: sideslip")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Beta (rad)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(data.time_s, data.yaw_rate_rps, "k", linewidth=2, label="Yaw rate true")
    axes[1].plot(data.time_s, yaw_rate_est_rps, label="Yaw rate estimate")
    axes[1].set_title(f"{title}: yaw rate")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Yaw rate (rad/s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(data.global_y_m, data.global_x_m, "k", linewidth=2, label="True path")
    axes[2].plot(path_y_est, path_x_est, label="Estimated path")
    axes[2].set_title(f"{title}: reconstructed path")
    axes[2].set_xlabel("Global Y (m)")
    axes[2].set_ylabel("Global X (m)")
    axes[2].axis("equal")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(plot_file, dpi=180)
    plt.close(fig)






# for linear KF
def vehicle_linear_fx(state, dt, ay_input, yaw_acc_input, vx_mps):
    # Linear approximation: x_next = F*x + B*u
    F = vehicle_F_jacobian(dt)
    # Mapping inputs to state changes directly
    B = np.array([[dt/max(abs(vx_mps), 0.5), 0], [0, dt]])
    u = np.array([ay_input, yaw_acc_input])
    return F @ state + B @ u

def vehicle_linear_H():
    return np.array([[0.0, 1.0]], dtype=float) 

#dynamic model for UKF
def bicycle_dynamic_state_derivative(
    state: np.ndarray,
    delta_rad: float,
    vx_mps: float,
    cf: float,
    cr: float,
    m: float,
    iz: float,
    lf: float,
    lr: float
) -> np.ndarray:
    beta, yaw_rate = state
    vx = max(abs(vx_mps), 0.5) # Prevent division by zero
    
    # 1. Calculate Tire Slip Angles (Alpha_f and Alpha_r)
    alpha_f = delta_rad - beta - (lf * yaw_rate) / vx
    alpha_r = -beta + (lr * yaw_rate) / vx
    
    # 2. Compute Lateral Tire Forces (Fy = C * alpha)
    # Note: Your data loader assumes positive stiffness values [cite: 591, 950]
    Fyf = cf * alpha_f
    Fyr = cr * alpha_r
    
    # 3. Apply Newton-Euler equations for lateral and yaw motion
    beta_dot = (Fyf + Fyr) / (m * vx) - yaw_rate
    yaw_rate_dot = (lf * Fyf - lr * Fyr) / iz
    
    return np.array([beta_dot, yaw_rate_dot], dtype=float)

def vehicle_dynamic_fx(state: np.ndarray, dt: float, delta_rad: float, vx_mps: float, 
                       cf: float, cr: float, m: float, iz: float, lf: float, lr: float) -> np.ndarray:
    
    derivative = bicycle_dynamic_state_derivative(state, delta_rad, vx_mps, cf, cr, m, iz, lf, lr)
    return state + dt * derivative

#for iterative

def vehicle_dynamic_bicycle_derivative(state: np.ndarray, delta_rad: float, vx_mps: float,
                                       cf: float, cr: float, m: float, iz: float, lf: float, lr: float) -> np.ndarray:
    
    beta, yaw_rate = state
    vx = max(abs(vx_mps), 0.5)
    
    # Slip angles calculations
    alpha_f = delta_rad - beta - (lf * yaw_rate) / vx
    alpha_r = -beta + (lr * yaw_rate) / vx
    
    # Lateral tire forces
    Fyf = cf * alpha_f
    Fyr = cr * alpha_r
    
    # Equations of motion
    beta_dot = (Fyf + Fyr) / (m * vx) - yaw_rate
    yaw_rate_dot = (lf * Fyf - lr * Fyr) / iz
    return np.array([beta_dot, yaw_rate_dot], dtype=float)

def vehicle_dynamic_ekf_fx(state: np.ndarray, dt: float, delta_rad: float, vx_mps: float,
                           cf: float, cr: float, m: float, iz: float, lf: float, lr: float) -> np.ndarray:
    return state + dt * vehicle_dynamic_bicycle_derivative(state, delta_rad, vx_mps, cf, cr, m, iz, lf, lr)

def vehicle_dynamic_F_jacobian(dt: float, vx_mps: float, cf: float, cr: float, m: float, iz: float, lf: float, lr: float) -> np.ndarray:
    vx = max(abs(vx_mps), 0.5)
    
    d_betadot_d_beta = -(cf + cr) / (m * vx)
    d_betadot_d_yawrate = (-cf * lf + cr * lr) / (m * vx**2) - 1.0
    
    d_yawratedot_d_beta = (-cf * lf + cr * lr) / iz
    d_yawratedot_d_yawrate = -(cf * lf**2 + cr * lr**2) / (iz * vx)
    
    
    A = np.array([
        [d_betadot_d_beta, d_betadot_d_yawrate],
        [d_yawratedot_d_beta, d_yawratedot_d_yawrate]
    ], dtype=float)
    
    #first-order Taylor series: F = I + A * dt
    return np.eye(2) + A * dt
'''
# vehicle_filter_common_new.py
from dataclasses import dataclass
import numpy as np
from scipy.integrate import cumulative_trapezoid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

@dataclass
class VehicleSimPreparedData:
    time_s: np.ndarray
    vx_mps: np.ndarray
    vy_mps: np.ndarray
    delta_rad: np.ndarray
    ay_mps2: np.ndarray
    yaw_rate_rps: np.ndarray
    fyf_true_n: np.ndarray
    fyr_true_n: np.ndarray
    beta_true_rad: np.ndarray
    yaw_true_rad: np.ndarray
    global_x_m: np.ndarray
    global_y_m: np.ndarray
    yaw_acc_rps2: np.ndarray
    cf_nprad: float
    cr_nprad: float
    m_kg: float
    iz_kgm2: float
    lf_m: float
    lr_m: float

def load_vehicle_sim_prepared_data(data_file: str) -> VehicleSimPreparedData:
    data = np.load(data_file)
    return VehicleSimPreparedData(
        time_s=data["time_s"],
        vx_mps=data["vx_mps"],
        vy_mps=data["vy_mps"] if "vy_mps" in data else data["vx_mps"] * data["beta_true_rad"],
        delta_rad=data["delta_rad"],
        ay_mps2=data["ay_mps2"],
        yaw_rate_rps=data["yaw_rate_rps"],
        fyf_true_n=data["fyf_true_n"] if "fyf_true_n" in data else np.zeros_like(data["time_s"]),
        fyr_true_n=data["fyr_true_n"] if "fyr_true_n" in data else np.zeros_like(data["time_s"]),
        beta_true_rad=data["beta_true_rad"],
        yaw_true_rad=data["yaw_true_rad"],
        global_x_m=data["global_x_m"],
        global_y_m=data["global_y_m"],
        yaw_acc_rps2=data["yaw_acc_rps2"],
        cf_nprad=float(data["cf_nprad"][0]),
        cr_nprad=float(data["cr_nprad"][0]),
        m_kg=float(data["m_kg"][0]),
        iz_kgm2=float(data["iz_kgm2"][0]),
        lf_m=float(data["lf_m"][0]),
        lr_m=float(data["lr_m"][0]),
    )

def safe_speed(vx_mps: float) -> float:
    return float(max(abs(vx_mps), 0.5))

def bicycle_state_derivative(state: np.ndarray, ay_input_mps2: float, yaw_acc_input_rps2: float, vx_mps: float) -> np.ndarray:
    beta_rad, yaw_rate_rps = state
    vx_safe = safe_speed(vx_mps)
    beta_dot = (ay_input_mps2 / vx_safe) - yaw_rate_rps
    yaw_rate_dot = yaw_acc_input_rps2
    return np.array([beta_dot, yaw_rate_dot], dtype=float)

def vehicle_fx(state: np.ndarray, dt: float, ay_input_mps2: float, yaw_acc_input_rps2: float, vx_mps: float) -> np.ndarray:
    derivative = bicycle_state_derivative(state, ay_input_mps2, yaw_acc_input_rps2, vx_mps)
    return state + dt * derivative

def vehicle_F_jacobian(dt: float) -> np.ndarray:
    return np.array([[1.0, -dt], [0.0, 1.0]], dtype=float)

def vehicle_hx(state: np.ndarray) -> np.ndarray:
    return np.array([state[1]], dtype=float)

def vehicle_H_jacobian() -> np.ndarray:
    return np.array([[0.0, 1.0]], dtype=float)

def reconstruct_path(time_s: np.ndarray, vx_mps: np.ndarray, beta_rad: np.ndarray, yaw_rate_rps: np.ndarray, yaw0_rad: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yaw_est_rad = cumulative_trapezoid(yaw_rate_rps, time_s, initial=0.0) + yaw0_rad
    vy_mps = vx_mps * beta_rad
    vx_global = vx_mps * np.cos(yaw_est_rad) - vy_mps * np.sin(yaw_est_rad)
    vy_global = vx_mps * np.sin(yaw_est_rad) + vy_mps * np.cos(yaw_est_rad)
    x_m = cumulative_trapezoid(vx_global, time_s, initial=0.0)
    y_m = cumulative_trapezoid(vy_global, time_s, initial=0.0)
    return x_m, y_m, yaw_est_rad

def rmse(truth: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.sqrt(np.mean((truth - estimate) ** 2)))

def vehicle_linear_fx(state, dt, ay_input, yaw_acc_input, vx_mps):
    F = vehicle_F_jacobian(dt)
    B = np.array([[dt / max(abs(vx_mps), 0.5), 0], [0, dt]])
    u = np.array([ay_input, yaw_acc_input])
    return F @ state + B @ u

def vehicle_linear_H():
    return np.array([[0.0, 1.0]], dtype=float)