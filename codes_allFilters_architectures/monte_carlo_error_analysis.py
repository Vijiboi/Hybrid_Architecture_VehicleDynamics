from __future__ import annotations
import argparse
import copy
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive background safe for execution shells
import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# SENSOR NOISE AND VARIANCE CONFIGURATIONS
# ==============================================================================
@dataclass
class SimSensorNoiseProfile:
    """Noise for the simulation branch.
    Channels match force_transformer_common.SENSOR_FEATURE_NAMES: [cite: 19, 20]
    vx_mps, delta_rad, ay_mps2, yaw_rate_rps, yaw_acc_rps2 [cite: 21]
    """
    vx_mps: float = 0.02          # wheel-speed / GPS (~0.02 m/s) [cite: 22]
    delta_rad: float = 0.002      # steering encoder (~0.11 deg) [cite: 23]
    ay_mps2: float = 0.05         # IMU lateral accel (~0.5% g) [cite: 23]
    yaw_rate_rps: float = 0.005   # IMU gyro (~0.3 deg/s) [cite: 23]
    yaw_acc_rps2: float = 0.01    # numerical deriv of yaw rate [cite: 23]

    def scale(self, k: float) -> SimSensorNoiseProfile:
        return SimSensorNoiseProfile(
            vx_mps=self.vx_mps * k,         # [cite: 26]
            delta_rad=self.delta_rad * k,   # [cite: 27]
            ay_mps2=self.ay_mps2 * k,       # [cite: 28]
            yaw_rate_rps=self.yaw_rate_rps * k, # [cite: 29]
            yaw_acc_rps2=self.yaw_acc_rps2 * k  # [cite: 30]
        )

    def as_dict(self) -> dict:
        return {
            "vx_mps": self.vx_mps,          # [cite: 35]
            "delta_rad": self.delta_rad,    # [cite: 36]
            "ay_mps2": self.ay_mps2,        # [cite: 37]
            "yaw_rate_rps": self.yaw_rate_rps, # [cite: 38]
            "yaw_acc_rps2": self.yaw_acc_rps2, # [cite: 39]
        }


@dataclass
class RealSensorNoiseProfile:
    """Noise for the real-sensor (TUM) branch.
    Channels match real_sensor_common.SENSOR_COLUMNS order: [cite: 41, 42, 43]
    [0] vx_mps, [1] vy_mps, [2] dpsi_radps, [3] ax_mps2, [4] ay_mps2,
    [5] deltawheel_rad, [6] TwheelRL_Nm, [7] TwheelRR_Nm, [8] pBrakeF_bar, [9] pBrakeR_bar [cite: 46, 47, 48, 49, 51, 53, 54, 55, 56, 57]
    """
    vx_mps: float = 0.02          # GPS/wheel speed [cite: 58, 59]
    vy_mps: float = 0.01          # lateral speed sensor [cite: 60, 61]
    dpsi_radps: float = 0.005     # IMU gyro [cite: 62]
    ax_mps2: float = 0.05         # IMU longitudinal accel [cite: 63, 64]
    ay_mps2: float = 0.05         # IMU lateral accel [cite: 65, 66]
    deltawheel_rad: float = 0.002 # steering encoder [cite: 66]
    TwheelRL_Nm: float = 2.0      # torque sensor [cite: 67]
    TwheelRR_Nm: float = 2.0      # torque sensor [cite: 68, 69]
    pBrakeF_bar: float = 0.05     # pressure sensor [cite: 70, 71]
    pBrakeR_bar: float = 0.05     # pressure sensor [cite: 72]

    def scale(self, k: float) -> RealSensorNoiseProfile:
        return RealSensorNoiseProfile(
            vx_mps=self.vx_mps * k,                 # [cite: 75]
            vy_mps=self.vy_mps * k,                 # [cite: 76]
            dpsi_radps=self.dpsi_radps * k,         # [cite: 77]
            ax_mps2=self.ax_mps2 * k,               # [cite: 78]
            ay_mps2=self.ay_mps2 * k,               # [cite: 79]
            deltawheel_rad=self.deltawheel_rad * k, # [cite: 80]
            TwheelRL_Nm=self.TwheelRL_Nm * k,       # [cite: 81]
            TwheelRR_Nm=self.TwheelRR_Nm * k,       # [cite: 82]
            pBrakeF_bar=self.pBrakeF_bar * k,       # [cite: 83]
            pBrakeR_bar=self.pBrakeR_bar * k        # [cite: 84]
        )

    def as_dict(self) -> dict:
        return {
            "vx_mps": self.vx_mps,                  # [cite: 90]
            "vy_mps": self.vy_mps,                  # [cite: 91]
            "dpsi_radps": self.dpsi_radps,          # [cite: 92]
            "ax_mps2": self.ax_mps2,                # [cite: 93]
            "ay_mps2": self.ay_mps2,                # [cite: 94]
            "deltawheel_rad": self.deltawheel_rad,  # [cite: 95]
            "TwheelRL_Nm": self.TwheelRL_Nm,        # [cite: 96]
            "TwheelRR_Nm": self.TwheelRR_Nm,        # [cite: 97]
            "pBrakeF_bar": self.pBrakeF_bar,        # [cite: 98]
            "pBrakeR_bar": self.pBrakeR_bar,        # [cite: 99]
        }

    def as_array(self) -> np.ndarray:
        """Return sigmas in SENSOR_COLUMNS order (10 values).""" 
        return np.array([
            self.vx_mps, self.vy_mps, self.dpsi_radps,      # [cite: 103]
            self.ax_mps2, self.ay_mps2, self.deltawheel_rad, # [cite: 104]
            self.TwheelRL_Nm, self.TwheelRR_Nm,             # [cite: 105]
            self.pBrakeF_bar, self.pBrakeR_bar,             # [cite: 106]
        ], dtype=float)


# ==============================================================================
# EKF COVARIANCE PERTURBATION ENGINE
# ==============================================================================
@dataclass
class EKFCovariancePerturbation:
    """Log-normal spread applied to Q and R each trial. [cite: 113, 114]
    cov_spread = 0.15 means +-~15% coefficient of variation. [cite: 115]
    """
    cov_spread: float = 0.15                                # [cite: 117]


def perturb_ekf_matrices(
    Q_base: np.ndarray,
    R_base: np.ndarray,
    perturb: EKFCovariancePerturbation,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:                                # [cite: 118, 119, 120, 121, 122, 123]
    q_factors = np.exp(rng.normal(0.0, perturb.cov_spread, size=Q_base.shape[0])) # [cite: 124]
    r_factors = np.exp(rng.normal(0.0, perturb.cov_spread, size=R_base.shape[0])) # [cite: 124]
    return np.diag(np.diag(Q_base) * q_factors), np.diag(np.diag(R_base) * r_factors) # [cite: 124]


@dataclass
class TrialResult:
    """Holds error metrics and computational metrics for individual runs.""" 
    beta_rmse: float                                         # [cite: 131, 132]
    yaw_rate_rmse: float                                     # [cite: 133]
    force_rmse: float  # wheel-force RMSE [N]                # [cite: 134]


# ==============================================================================
# DYNAMIC SIMULATION MODULES LOADER (SIM BRANCH)
# ==============================================================================
def _load_sim_imports():
    """Import sim-branch modules; abort with a clear message if missing.""" 
    try:
        from force_transformer_common import (
            ForceModelBundle, ForcePredictor, VehicleParams, # [cite: 142, 144]
            numerical_state_jacobian,                       # [cite: 145]
            scenario_sensor_matrix, build_sensor_window_array, # [cite: 146]
        )
        from vehicle_filter_common import (
            VehicleSimPreparedData, load_vehicle_sim_prepared_data, rmse, # [cite: 147, 149]
        )
    except ImportError as exc:
        sys.exit(
            f"[MC-sim] Cannot import sim modules: {exc}\n"  # [cite: 151, 152]
            "Run from codes_2Stage_hybrid_architecture/ or add it to PYTHONPATH." # [cite: 153]
        )
    return (
        ForceModelBundle, ForcePredictor, VehicleParams,    # [cite: 155, 156]
        numerical_state_jacobian,                           # [cite: 157]
        scenario_sensor_matrix, build_sensor_window_array, # [cite: 159]
        VehicleSimPreparedData, load_vehicle_sim_prepared_data, rmse, # [cite: 160]
    )


def _add_sim_noise(
    data,
    profile: SimSensorNoiseProfile,
    rng: np.random.Generator,
) -> copy:                                                  # [cite: 161, 163, 164, 165, 166, 167]
    n = len(data.time_s)                                    # [cite: 169]
    noisy = copy.copy(data)                                 # [cite: 170]
    noisy.vx_mps = np.maximum(data.vx_mps + rng.normal(0.0, profile.vx_mps, n), 0.1) # [cite: 171]
    noisy.delta_rad = data.delta_rad + rng.normal(0.0, profile.delta_rad, n) # [cite: 172]
    noisy.ay_mps2 = data.ay_mps2 + rng.normal(0.0, profile.ay_mps2, n) # [cite: 173]
    noisy.yaw_rate_rps = data.yaw_rate_rps + rng.normal(0.0, profile.yaw_rate_rps, n) # [cite: 174]
    noisy.yaw_acc_rps2 = data.yaw_acc_rps2 + rng.normal(0.0, profile.yaw_acc_rps2, n) # [cite: 175]
    return noisy                                            # [cite: 175]


def _clip_sim(state: np.ndarray) -> np.ndarray:
    s = state.copy()                                        # [cite: 177]
    s[0] = np.clip(s[0], -0.45, 0.45)                       # [cite: 178]
    s[1] = np.clip(s[1], -1.8, 1.8)                         # [cite: 179]
    s[2] = np.clip(s[2], -4000.0, 4000.0)                   # [cite: 180]
    s[3] = np.clip(s[3], -4.0, 4.0)                         # [cite: 181]
    return s                                                # [cite: 182]


def _sim_force_sums(wf: np.ndarray, delta: float):
    fl = (float(wf[0]) + float(wf[1])) * np.cos(delta)     # [cite: 183, 184]
    rl = float(wf[2]) + float(wf[3])                        # [cite: 185]
    return fl, rl                                           # [cite: 186]


def _run_sim_ekf_trial(data, wf_pred, params, Q, R, numerical_state_jacobian):
    x = np.array([data.beta_true_rad[0], data.yaw_rate_rps[0], 0.0, 0.0]) # [cite: 187, 188]
    P = np.diag([0.02**2, 0.05**2, 400.0**2, 0.20**2])      # [cite: 189]
    estimates = [x.copy()]                                  # [cite: 190]
    
    for i in range(1, len(data.time_s)):                    # [cite: 191]
        dt = float(data.time_s[i] - data.time_s[i - 1])     # [cite: 192]
        vx = float(data.vx_mps[i])                          # [cite: 193]
        delt = float(data.delta_rad[i])                     # [cite: 194]
        wf = wf_pred[i]                                     # [cite: 195]
        
        def fx(s):
            yr, dfy = s[1], s[2]                            # [cite: 197]
            fl, rl = _sim_force_sums(wf, delt)              # [cite: 198]
            vxs = max(abs(vx), 0.5)                         # [cite: 199]
            return _clip_sim(s + dt * np.array([            # [cite: 200]
                (fl + rl + dfy) / (params.m_kg * vxs) - yr, # [cite: 202]
                (params.lf_m * fl - params.lr_m * rl) / params.iz_kgm2, # [cite: 203]
                0.0, 0.0                                    # [cite: 203]
            ]))                                             # [cite: 201]
            
        def hx(s):
            yr, dfy, ayb = s[1], s[2], s[3]                 # [cite: 204, 205]
            fl, rl = _sim_force_sums(wf, delt)              # [cite: 206, 207]
            return np.array([yr, (fl + rl + dfy) / params.m_kg + ayb]) # [cite: 208]
            
        x_pred = fx(x)                                      # [cite: 209]
        F = numerical_state_jacobian(fx, x)                 # [cite: 210]
        P_pred = F @ P @ F.T + Q                            # [cite: 211]
        z = np.array([float(data.yaw_rate_rps[i]), float(data.ay_mps2[i])]) # [cite: 212]
        H = numerical_state_jacobian(hx, x_pred)            # [cite: 213]
        inn = z - hx(x_pred)                                # [cite: 214]
        S = H @ P_pred @ H.T + R                            # [cite: 215]
        K = P @ H.T @ np.linalg.inv(S)                      # [cite: 216]
        x = _clip_sim(x_pred + K @ inn)                     # [cite: 217]
        P = (np.eye(4) - K @ H) @ P_pred                    # [cite: 218]
        P = 0.5 * (P + P.T)                                 # [cite: 219]
        estimates.append(x.copy())                          # [cite: 220]
        
    return np.asarray(estimates)                            # [cite: 221]


def _sim_predict_forces(predictor, data, scenario_sensor_matrix, build_sensor_window_array):
    sensors = scenario_sensor_matrix(data)                  # [cite: 222, 223]
    ws = predictor.bundle.config.window_size                # [cite: 224]
    windows = np.stack(
        [build_sensor_window_array(sensors, i, ws) for i in range(len(data.time_s))], # [cite: 225, 227]
        axis=0                                              # [cite: 228]
    )                                                       # [cite: 226]
    return predictor.predict_forces(windows)                # [cite: 229]


def run_sim_trial(
    clean_data, predictor, params,
    profile: SimSensorNoiseProfile,
    Q_base: np.ndarray, R_base: np.ndarray,
    ekf_perturb: EKFCovariancePerturbation,
    rng: np.random.Generator,
    numerical_state_jacobian, scenario_sensor_matrix, build_sensor_window_array, rmse_fn
) -> TrialResult:                                           # [cite: 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243]
    noisy = _add_sim_noise(clean_data, profile, rng)        # [cite: 244]
    Q, R = perturb_ekf_matrices(Q_base, R_base, ekf_perturb, rng) # [cite: 245]
    wf_pred = _sim_predict_forces(predictor, noisy, scenario_sensor_matrix, build_sensor_window_array) # [cite: 246]
    est = _run_sim_ekf_trial(noisy, wf_pred, params, Q, R, numerical_state_jacobian) # [cite: 247]
    
    axle_front_true = clean_data.fyf_true_n                 # [cite: 249]
    axle_rear_true = clean_data.fyr_true_n                   # [cite: 250]
    front_pred = wf_pred[:, 0] + wf_pred[:, 1]               # [cite: 251]
    rear_pred = wf_pred[:, 2] + wf_pred[:, 3]                 # [cite: 252]
    force_rmse = float(np.sqrt(np.mean(
        (front_pred - axle_front_true) ** 2 + (rear_pred - axle_rear_true) ** 2 # [cite: 254]
    ) / 2))                                                 # [cite: 253, 255]
    
    return TrialResult(
        beta_rmse=rmse_fn(clean_data.beta_true_rad, est[:, 0]), # [cite: 256, 258]
        yaw_rate_rmse=rmse_fn(clean_data.yaw_rate_rps, est[:, 1]), # [cite: 259]
        force_rmse=force_rmse                               # [cite: 260]
    )                                                       # [cite: 257]


def run_sim_monte_carlo(
    data_path: str, model_path: str,
    profile: SimSensorNoiseProfile,
    ekf_perturb: EKFCovariancePerturbation,
    n_trials: int, device: str, seed: int, verbose: bool = True
) -> list[TrialResult]:                                     # [cite: 261, 262, 263, 264, 265, 266, 267, 268, 269, 270]
    (
        ForceModelBundle, ForcePredictor, VehicleParams,    # [cite: 271, 272]
        numerical_state_jacobian,                           # [cite: 273]
        scenario_sensor_matrix, build_sensor_window_array, # [cite: 274]
        VehicleSimPreparedData, load_vehicle_sim_prepared_data, rmse_fn, # [cite: 275]
    ) = _load_sim_imports()                                 # [cite: 276]
    
    print(f"[MC-sim] Loading data: {data_path}")            # [cite: 277]
    clean_data = load_vehicle_sim_prepared_data(data_path) # [cite: 278]
    print(f"[MC-sim] Loading model: {model_path}")          # [cite: 279]
    bundle = ForceModelBundle.load(model_path, device=device) # [cite: 280, 281]
    predictor = ForcePredictor(bundle, device=device)       # [cite: 282]
    params = VehicleParams.from_prepared_data(clean_data)   # [cite: 283]
    
    print(f"[MC-sim] {len(clean_data.time_s)} steps | m={params.m_kg:.0f} kg | lf={params.lf_m:.2f} m | lr={params.lr_m:.2f} m") # [cite: 284, 285]
    Q_base = np.diag([0.003**2, 0.04**2, 80.0**2, 0.03**2])  # [cite: 286, 287]
    R_base = np.diag([0.02**2, 0.20**2])                     # [cite: 287]
    rng = np.random.default_rng(seed)                       # [cite: 288]
    results: list[TrialResult] = []                         # [cite: 289]
    
    t0 = time.perf_counter()                                # [cite: 291]
    for trial in range(n_trials):                           # [cite: 292]
        r = run_sim_trial(
            clean_data, predictor, params, profile, Q_base, R_base, ekf_perturb, rng,
            numerical_state_jacobian, scenario_sensor_matrix, build_sensor_window_array, rmse_fn
        )                                                   # [cite: 293, 294, 295, 296, 297, 298]
        results.append(r)                                   # [cite: 299]
        if verbose and ((trial + 1) % max(1, n_trials // 10) == 0 or trial == 0): # [cite: 300]
            eta = (time.perf_counter() - t0) / (trial + 1) * (n_trials - trial - 1) # [cite: 301]
            print(f" Trial {trial+1:>4d}/{n_trials} | beta={r.beta_rmse:.5f} rad | yr={r.yaw_rate_rmse:.5f} rad/s | ETA {eta:.0f}s") # [cite: 302, 303, 304, 305]
            
    return results                                          # [cite: 306]


# ==============================================================================
# DYNAMIC REAL-WORLD CONFIG LOADER (REAL TUM BRANCH)
# ==============================================================================
def _load_real_imports():
    """Dynamically interfaces with localized real sensor modules.""" 
    try:
        from real_sensor_common import (
            load_project_configs, load_real_sensor_csv,     # [cite: 313, 314, 316]
            RealSensorForceModelBundle, RealSensorForcePredictor, # [cite: 317, 318]
            build_sensor_windows, model_output_to_wheel_forces, # [cite: 319, 320]
            model_output_to_beta_measurement, run_real_sensor_force_ekf, # [cite: 321, 322]
            rmse, SENSOR_COLUMNS                             # [cite: 323, 324]
        )
    except ImportError as exc:
        sys.exit(
            f"[MC-real] Cannot import real_sensor_common: {exc}\n" # [cite: 325, 327]
            "Run from codes_allFilters_architectures/ or add it to PYTHONPATH." # [cite: 328]
        )
    return (
        load_project_configs, load_real_sensor_csv,         # [cite: 330, 332]
        RealSensorForceModelBundle, RealSensorForcePredictor, # [cite: 333]
        build_sensor_windows, model_output_to_wheel_forces, # [cite: 334]
        model_output_to_beta_measurement, run_real_sensor_force_ekf, # [cite: 335]
        rmse, SENSOR_COLUMNS,                               # [cite: 336]
    )


def _add_real_noise(dataset, profile: RealSensorNoiseProfile, vehicle, rng: np.random.Generator):
    """Corrupts hardware signals and re-derives dependent tracking fields.""" 
    from real_sensor_common import build_twintrack_wheel_force_targets, RealSensorDataset # [cite: 348, 350, 351]
    
    n = len(dataset.sensor_matrix)                          # [cite: 352]
    sigmas = profile.as_array()                             # [cite: 353]
    noise = rng.normal(0.0, 1.0, size=(n, 10)) * sigmas[None, :] # [cite: 354]
    noisy_matrix = dataset.sensor_matrix.astype(float) + noise # [cite: 355]
    noisy_matrix[:, 0] = np.maximum(noisy_matrix[:, 0], 0.1)  # Safeguard forward tracking velocity [cite: 356]
    
    vx = noisy_matrix[:, 0]                                 # [cite: 357, 359]
    vy = noisy_matrix[:, 1]                                 # [cite: 358, 360]
    yaw_rate = noisy_matrix[:, 2]                           # [cite: 361]
    ay = noisy_matrix[:, 4]                                 # [cite: 362]
    steer = noisy_matrix[:, 5]                              # [cite: 363]
    t_rl = noisy_matrix[:, 6]                               # [cite: 364]
    t_rr = noisy_matrix[:, 7]                               # [cite: 365]
    
    beta_ref = np.arctan2(vy, np.maximum(np.abs(vx), 0.5))  # [cite: 366]
    yaw_acc = np.gradient(yaw_rate, vehicle.sample_time_s)  # [cite: 367]
    
    total_lat = vehicle.mass_kg * ay                        # [cite: 368]
    front_body_lat = (vehicle.iz_kgm2 * yaw_acc + vehicle.lr_m * total_lat) / (vehicle.lf_m + vehicle.lr_m) # [cite: 369]
    rear_lat = total_lat - front_body_lat                    # [cite: 370]
    front_tire_lat = front_body_lat / np.clip(np.cos(steer), 0.7, None) # [cite: 371]
    
    wf_targets = build_twintrack_wheel_force_targets(
        front_tire_lat=front_tire_lat, rear_lat=rear_lat, ay_mps2=ay,
        torque_rl_nm=t_rl, torque_rr_nm=t_rr, vehicle=vehicle
    )                                                       # [cite: 372, 373, 374, 375, 376, 377, 378, 379]
    
    axle_targets = np.column_stack([
        wf_targets[:, 0] + wf_targets[:, 1],                # [cite: 380, 381]
        wf_targets[:, 2] + wf_targets[:, 3],                # [cite: 382]
    ]).astype(np.float32)                                   # [cite: 383]
    
    return RealSensorDataset(
        sensor_matrix=noisy_matrix.astype(np.float32),      # [cite: 384, 385]
        beta_ref_rad=beta_ref.astype(np.float32),           # [cite: 386]
        yaw_rate_rps=yaw_rate.astype(np.float32),           # [cite: 387]
        yaw_acc_rps2=yaw_acc.astype(np.float32),            # [cite: 388]
        ay_mps2=ay.astype(np.float32),                      # [cite: 389]
        wheel_force_targets_n=wf_targets.astype(np.float32), # [cite: 390, 392]
        axle_force_targets_n=axle_targets,                  # [cite: 393]
        time_s=dataset.time_s                               # [cite: 394]
    )


def _build_real_ekf_matrices(ekf_cfg):
    Q = np.diag([
        ekf_cfg.q_beta**2,                                  # [cite: 396, 397, 398]
        ekf_cfg.q_yaw_rate**2,                              # [cite: 399]
        ekf_cfg.q_delta_fy_bias**2,                         # [cite: 400]
        ekf_cfg.q_ay_bias**2,                               # [cite: 401]
    ])
    beta_std0 = max(0.05, ekf_cfg.beta_std0)               # [cite: 402]
    r_beta = max(0.05, 2.0 * beta_std0)                     # [cite: 403]
    R = np.diag([ekf_cfg.r_yaw_rate**2, ekf_cfg.r_ay**2, r_beta**2]) # [cite: 404]
    return Q, R                                             # [cite: 405]


def run_real_trial(
    clean_dataset, predictor, vehicle, ekf_cfg,
    profile: RealSensorNoiseProfile,
    Q_base: np.ndarray, R_base: np.ndarray,
    ekf_perturb: EKFCovariancePerturbation,
    rng: np.random.Generator,
    build_sensor_windows_fn, model_output_to_wheel_forces_fn, model_output_to_beta_measurement_fn,
    run_ekf_fn, rmse_fn
) -> TrialResult:                                           # [cite: 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 417, 418, 419, 420, 421, 422]
    noisy = _add_real_noise(clean_dataset, profile, vehicle, rng) # [cite: 423, 424]
    Q, R = perturb_ekf_matrices(Q_base, R_base, ekf_perturb, rng) # [cite: 425, 426]
    
    from real_sensor_common import EkfSettings               # [cite: 429]
    perturbed_ekf = EkfSettings(
        beta_std0=ekf_cfg.beta_std0,                        # [cite: 430, 431]
        yaw_rate_std0=ekf_cfg.yaw_rate_std0,                # [cite: 432]
        delta_fy_bias_std0=ekf_cfg.delta_fy_bias_std0,      # [cite: 433]
        ay_bias_std0=ekf_cfg.ay_bias_std0,                  # [cite: 434, 435]
        q_beta=float(np.sqrt(Q[0, 0])),                     # [cite: 436]
        q_yaw_rate=float(np.sqrt(Q[1, 1])),                 # [cite: 437]
        q_delta_fy_bias=float(np.sqrt(Q[2, 2])),            # [cite: 438]
        q_ay_bias=float(np.sqrt(Q[3, 3])),                  # [cite: 439]
        r_yaw_rate=float(np.sqrt(R[0, 0])),                 # [cite: 441]
        r_ay=float(np.sqrt(R[1, 1]))                        # [cite: 442]
    )
    
    ws = predictor.bundle.window_size                       # [cite: 444, 445]
    windows = build_sensor_windows_fn(noisy.sensor_matrix, ws) # [cite: 446]
    pred_struct = predictor.predict(windows)                 # [cite: 447]
    wf_pred = model_output_to_wheel_forces_fn(
        pred_struct, noisy.sensor_matrix, predictor.bundle, vehicle # [cite: 448, 450]
    )                                                       # [cite: 449]
    beta_meas = model_output_to_beta_measurement_fn(pred_struct) # [cite: 451]
    
    ekf_out = run_ekf_fn(noisy, wf_pred, vehicle, perturbed_ekf, beta_meas) # [cite: 452, 453]
    
    beta_rmse = rmse_fn(clean_dataset.beta_ref_rad, ekf_out["beta_estimates"]) # [cite: 454, 455]
    yaw_rate_rmse = rmse_fn(clean_dataset.yaw_rate_rps, ekf_out["estimates"][:, 1]) # [cite: 455]
    force_rmse = float(np.sqrt(np.mean(
        (clean_dataset.wheel_force_targets_n - wf_pred) ** 2 # [cite: 455, 457]
    )))                                                     # [cite: 455, 456]
    
    return TrialResult(
        beta_rmse=beta_rmse,                                # [cite: 458, 460]
        yaw_rate_rmse=yaw_rate_rmse,                        # [cite: 461]
        force_rmse=force_rmse                               # [cite: 462]
    )                                                       # [cite: 459]


def run_real_monte_carlo(
    csv_path: str, params_file: str, vehicle_config: str, model_path: str,
    profile: RealSensorNoiseProfile,
    ekf_perturb: EKFCovariancePerturbation,
    n_trials: int, device: str, seed: int, verbose: bool = True
) -> list[TrialResult]:                                     # [cite: 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474]
    (
        load_project_configs, load_real_sensor_csv,         # [cite: 475, 476]
        RealSensorForceModelBundle, RealSensorForcePredictor, # [cite: 477]
        build_sensor_windows_fn, model_output_to_wheel_forces_fn, # [cite: 478]
        model_output_to_beta_measurement_fn, run_ekf_fn,    # [cite: 479]
        rmse_fn, SENSOR_COLUMNS,                             # [cite: 480]
    ) = _load_real_imports()                                # [cite: 481]
    
    print(f"[MC-real] Loading configs: {params_file} , {vehicle_config}") # [cite: 482]
    nn_cfg, vehicle, ekf_cfg = load_project_configs(params_file, vehicle_config) # [cite: 483]
    print(f"[MC-real] Loading data: {csv_path}")            # [cite: 484]
    clean_dataset = load_real_sensor_csv(csv_path, vehicle) # [cite: 485]
    print(f"[MC-real] {len(clean_dataset.time_s)} steps | sample_time={vehicle.sample_time_s:.4f}s | mass={vehicle.mass_kg:.0f} kg") # [cite: 486, 487, 488]
    print(f"[MC-real] Loading model: {model_path}")          # [cite: 489]
    
    bundle = RealSensorForceModelBundle.load(model_path, device=device) # [cite: 490, 491]
    predictor = RealSensorForcePredictor(bundle, device=device) # [cite: 492]
    Q_base, R_base = _build_real_ekf_matrices(ekf_cfg)      # [cite: 493]
    rng = np.random.default_rng(seed)                       # [cite: 494]
    results: list[TrialResult] = []                         # [cite: 495]
    
    t0 = time.perf_counter()                                # [cite: 496]
    for trial in range(n_trials):                           # [cite: 497]
        r = run_real_trial(
            clean_dataset, predictor, vehicle, ekf_cfg, profile, Q_base, R_base, ekf_perturb, rng,
            build_sensor_windows_fn, model_output_to_wheel_forces_fn, model_output_to_beta_measurement_fn,
            run_ekf_fn, rmse_fn
        )                                                   # [cite: 498, 499, 500, 501, 502, 503]
        results.append(r)                                   # [cite: 504]
        if verbose and ((trial + 1) % max(1, n_trials // 10) == 0 or trial == 0): # [cite: 506]
            eta = (time.perf_counter() - t0) / (trial + 1) * (n_trials - trial - 1) # [cite: 507]
            print(f" Trial {trial+1:>4d}/{n_trials} | beta={r.beta_rmse:.5f} rad | yr={r.yaw_rate_rmse:.5f} rad/s | ETA {eta:.0f}s") # [cite: 508, 509, 510, 511]
            
    return results                                          # [cite: 512]


# ==============================================================================
# POST-PROCESSING METRICS LOGGERS AND VISUALIZATION ENGINE
# ==============================================================================
def summarise(results: list[TrialResult]) -> dict:
    fields = [("beta_rmse", "rad"), ("yaw_rate_rmse", "rad/s"), ("force_rmse", "N")] # [cite: 515, 516]
    out = {}                                                # [cite: 517]
    for f, unit in fields:                                  # [cite: 518]
        vals = np.array([getattr(r, f) for r in results])   # [cite: 519]
        out[f] = {                                          # [cite: 520]
            "unit": unit,                                   # [cite: 521]
            "mean": float(np.mean(vals)),                   # [cite: 522]
            "std": float(np.std(vals)),                     # [cite: 523]
            "p5": float(np.percentile(vals, 5)),            # [cite: 524, 525]
            "p50": float(np.percentile(vals, 50)),          # [cite: 525]
            "p95": float(np.percentile(vals, 95)),          # [cite: 526]
            "max": float(np.max(vals)),                     # [cite: 527]
        }                                                   # [cite: 528]
    return out                                              # [cite: 529]


def print_summary(stats: dict, noise_dict: dict, n_trials: int, branch: str) -> None:
    print("\n" + "="*66)                                     # [cite: 530]
    print(f" Monte Carlo Summary [{branch}] ({n_trials} trials)") # [cite: 531]
    print("="*66)                                           # [cite: 532]
    print(" Injected sensor noise (1σ):")                  # [cite: 533]
    for k, v in noise_dict.items():                        # [cite: 534]
        print(f"  {k:<22s} : {v:.5f}")                      # [cite: 535, 536]
    print("-"*66)                                           # [cite: 537]
    print(f" {'Metric':<22} {'Mean':>9} {'Std':>9} {'P5':>9} {'P95':>9}") # [cite: 538]
    print("-"*66)                                           # [cite: 539]
    for metric, s in stats.items():                         # [cite: 540]
        print(f" {metric:<22} : {s['mean']:>9.5f} ({s['std']:>9.5f}) | {s['p5']:>9.5f} {s['p95']:>9.5f} [{s['unit']}]") # [cite: 541, 542, 543]
    print("="*66 + "\n")                                    # [cite: 544]


def _vals(results: list[TrialResult], field: str) -> np.ndarray:
    return np.array([getattr(r, field) for r in results])   # [cite: 545, 546]


# ------------------------------------------------------------------------------
# UPDATED VISUALIZATION MATRIX ENGINE (2x3 HISTOGRAMS + BOXPLOTS MATRIX)
# ------------------------------------------------------------------------------
def plot_distributions(
    results: list[TrialResult],
    stats: dict,
    out_path: Path,
    title_suffix: str = "",
) -> None:                                                  # [cite: 547, 548, 549, 550, 551, 552]
    metrics = [                                             # [cite: 553]
        ("beta_rmse", "Beta RMSE [rad]"),                    # [cite: 555]
        ("yaw_rate_rmse", "Yaw-rate RMSE [rad/s]"),         # [cite: 556]
        ("force_rmse", "Wheel-force RMSE [N]"),             # [cite: 557]
    ]                                                       # [cite: 554]
    
    # Instantiate layout matrix grid mapping histograms on Row 0 and Box plots on Row 1
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), gridspec_kw={'height_ratios': [1.2, 0.8]}) # 
    fig.suptitle(f"Monte Carlo Error Distributions & Robustness Whiskers{title_suffix}", fontsize=12) # [cite: 559]
    
    for idx, (field, label) in enumerate(metrics):          # [cite: 560]
        v = _vals(results, field)                           # [cite: 561]
        s = stats[field]                                    # [cite: 562]
        
        ax_hist = axes[0, idx]
        ax_box  = axes[1, idx]
        
        # Row 0 Panel Block: High-Density Histograms (Original logic intact)
        ax_hist.hist(v, bins=30, color="steelblue", edgecolor="white", alpha=0.85) # [cite: 563]
        ax_hist.axvline(s["mean"], color="crimson", lw=2, label=f"Mean={s['mean']:.5f}") # [cite: 564]
        ax_hist.axvline(s["p5"], color="darkorange", lw=1.5, ls="--", label=f"P5={s['p5']:.5f}") # [cite: 564]
        ax_hist.axvline(s["p95"], color="darkorange", lw=1.5, ls=":", label=f"P95={s['p95']:.5f}") # [cite: 564]
        ax_hist.set_title(f"Distribution Profile: {field.upper()}", fontsize=9, fontweight='bold')
        ax_hist.set_ylabel("Count", fontsize=10)             # [cite: 565]
        ax_hist.legend(fontsize=8)                          # [cite: 566]
        ax_hist.grid(True, alpha=0.3)                       # [cite: 567]
        
        # Row 1 Panel Block: Aligned Horizontal Box Plots (New Request)
        box = ax_box.boxplot(v, vert=False, patch_artist=True, widths=0.45, showmeans=True,
                             meanprops={"marker": "D", "markeredgecolor": "crimson", "markerfacecolor": "crimson", "markersize": 4})
        
        # Apply structured color coordination across components
        box['boxes'][0].set(facecolor='steelblue', color='#2c3e50', alpha=0.6)
        box['medians'][0].set(color='#2c3e50', linewidth=2)
        
        ax_box.set_xlabel(label, fontsize=10)                # [cite: 564]
        ax_box.set_yticklabels([])
        ax_box.grid(True, linestyle=':', alpha=0.4)
        
    fig.tight_layout()                                      # [cite: 568]
    fig.savefig(out_path, dpi=200)                          # Enhanced to crisp 200 DPI [cite: 569]
    plt.close(fig)                                          # [cite: 570]
    print(f" Saved distribution plot Matrix → {out_path}")   # [cite: 571]


def plot_convergence(results: list[TrialResult], out_path: Path) -> None:
    metrics = [                                             # [cite: 572]
        ("beta_rmse", "Beta RMSE [rad]"),                    # [cite: 574]
        ("yaw_rate_rmse", "Yaw-rate RMSE [rad/s]"),         # [cite: 575]
        ("force_rmse", "Wheel-force RMSE [N]"),             # [cite: 576]
    ]                                                       # [cite: 573]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))         # [cite: 577]
    fig.suptitle("Monte Carlo Convergence (running mean)", fontsize=12) # [cite: 578]
    
    for ax, (field, label) in zip(axes, metrics):           # [cite: 579]
        v = _vals(results, field)                           # [cite: 580]
        rmean = np.cumsum(v) / np.arange(1, len(v) + 1)     # [cite: 581]
        ax.plot(rmean, color="steelblue", lw=1.5)            # [cite: 582]
        ax.axhline(rmean[-1], color="crimson", ls="--", lw=1, label=f"Final={rmean[-1]:.5f}") # [cite: 583, 584]
        ax.set_xlabel("Trials completed")                   # [cite: 585]
        ax.set_ylabel(label, fontsize=10)                    # [cite: 586]
        ax.legend(fontsize=9)                               # [cite: 587]
        ax.grid(True, alpha=0.3)                            # [cite: 588]
        
    fig.tight_layout()                                      # [cite: 589]
    fig.savefig(out_path, dpi=150)                          # [cite: 590]
    plt.close(fig)                                          # [cite: 591]
    print(f" Saved convergence plot → {out_path}")          # [cite: 592]


def plot_noise_sweep(sweep_results: dict[float, list[TrialResult]], out_path: Path) -> None:
    metrics = [                                             # [cite: 593, 594]
        ("beta_rmse", "Beta RMSE [rad]"),                    # [cite: 599]
        ("yaw_rate_rmse", "Yaw-rate RMSE [rad/s]"),         # [cite: 600]
        ("force_rmse", "Wheel-force RMSE [N]"),             # [cite: 601]
    ]                                                       # [cite: 597]
    scales = sorted(sweep_results.keys())                   # [cite: 602]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))         # [cite: 603]
    fig.suptitle("Noise Sensitivity Sweep (P50 vs P95)", fontsize=12) # [cite: 604]
    
    for ax, (field, label) in zip(axes, metrics):           # [cite: 605]
        p50 = [float(np.median(_vals(sweep_results[s], field))) for s in scales] # [cite: 606]
        p95 = [float(np.percentile(_vals(sweep_results[s], field), 95)) for s in scales] # [cite: 607]
        x = np.arange(len(scales))                          # [cite: 608]
        w = 0.35                                            # [cite: 609]
        
        ax.bar(x - w/2, p50, w, label="P50", color="steelblue", alpha=0.85) # [cite: 611]
        ax.bar(x + w/2, p95, w, label="P95", color="tomato", alpha=0.85) # [cite: 612]
        ax.set_xticks(x)                                    # [cite: 613]
        ax.set_xticklabels([f"x{s}" for s in scales])       # [cite: 614]
        ax.set_xlabel("Noise scale factor")                 # [cite: 615]
        ax.set_ylabel(label, fontsize=10)                    # [cite: 616]
        ax.legend(fontsize=9)                               # [cite: 617]
        ax.grid(True, alpha=0.3, axis="y")                  # [cite: 618]
        
    fig.tight_layout()                                      # [cite: 619]
    fig.savefig(out_path, dpi=150)                          # [cite: 620]
    plt.close(fig)                                          # [cite: 621]
    print(f" Saved sweep plot → {out_path}")                # [cite: 622]


# ==============================================================================
# TERMINAL INTERFACE INTERMEDIATE SPECIFICATIONS
# ==============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(                            # [cite: 624, 625]
        description="Monte Carlo error analysis for the Hybrid Transformer-EKF pipeline." # [cite: 627]
    )                                                       # [cite: 626]
    p.add_argument("--dataset", choices=["sim", "real"], default="sim", # [cite: 628]
                   help="'sim' = prepared .npz | 'real' = TUM 10-channel CSV") # [cite: 629]
    
    # Sim branch overrides                                  # [cite: 630]
    p.add_argument("--data", default="vehicle_sim_data.npz", # [cite: 631]
                   help="[sim] Prepared scenario.npz (from sim_prepare_data.py)") # [cite: 632]
    
    # Real branch overrides                                 # [cite: 633]
    p.add_argument("--csv", default=None, help="[real] TUM sensor CSV (e.g. data_to_run.csv)") # [cite: 634, 635]
    p.add_argument("--params-file", default="../params/parameters.toml", help="[real] parameters.toml path") # [cite: 636, 637]
    p.add_argument("--vehicle-config", default="real_sensor_vehicle_config.toml", help="[real] vehicle config.toml path") # [cite: 638, 639]
    
    # Shared settings                                       # [cite: 640]
    p.add_argument("--model", default="force_transformer_model.pt") # [cite: 641]
    p.add_argument("--trials", type=int, default=200)       # [cite: 642]
    p.add_argument("--seed", type=int, default=42)          # [cite: 643]
    p.add_argument("--device", default="cpu")               # [cite: 644]
    p.add_argument("--out-dir", default="mc_results")       # [cite: 645]
    
    # Override constraints parameters                       # [cite: 646, 647]
    p.add_argument("--noise-vx", type=float, default=0.02)  # [cite: 648]
    p.add_argument("--noise-delta", type=float, default=0.002) # [cite: 649]
    p.add_argument("--noise-ay", type=float, default=0.05)   # [cite: 650]
    p.add_argument("--noise-yaw-rate", type=float, default=0.005)
    p.add_argument("--noise-yaw-acc", type=float, default=0.01)
    
    p.add_argument("--noise-vy", type=float, default=0.01)  # [cite: 651, 652, 653]
    p.add_argument("--noise-ax", type=float, default=0.05)  # [cite: 654, 655]
    p.add_argument("--noise-torque", type=float, default=2.0, help="Applied to both TwheelRL and TwheelRR") # [cite: 656, 657, 658]
    p.add_argument("--noise-brake", type=float, default=0.05, help="Applied to both pBrakeF and pBrakeR") # [cite: 659, 660]
    
    p.add_argument("--ekf-cov-spread", type=float, default=0.15, # [cite: 661, 663]
                   help="Log-normal spread for EKF Q/R perturbation per trial (0=no perturbation)") # [cite: 664]
    p.add_argument("--noise-sweep", nargs="+", type=float, default=None, metavar="FACTOR", # [cite: 665, 666, 667]
                   help="Run MC at these noise-scale factors, e.g. --noise-sweep 0.5 1.0 2.0") # [cite: 668]
    return p.parse_args()                                   # [cite: 668]


# ==============================================================================
# MAIN CORE PROGRAM EXECUTION PASS ENTRYPOINT
# ==============================================================================
def main() -> None:
    args = parse_args()                                     # [cite: 671, 672]
    out_dir = Path(args.out_dir)                            # [cite: 673]
    out_dir.mkdir(parents=True, exist_ok=True)              # [cite: 674]
    ekf_perturb = EKFCovariancePerturbation(cov_spread=args.ekf_cov_spread) # [cite: 675]
    
    if args.dataset == "sim":                               # [cite: 676]
        base_profile = SimSensorNoiseProfile(
            vx_mps=args.noise_vx,                           # [cite: 677, 678]
            delta_rad=args.noise_delta,                     # [cite: 679]
            ay_mps2=args.noise_ay,                          # [cite: 680]
            yaw_rate_rps=args.noise_yaw_rate,               # [cite: 681]
            yaw_acc_rps2=args.noise_yaw_acc,                # [cite: 682]
        )                                                   # [cite: 683]
        label = Path(args.data).stem                        # [cite: 684]
        branch = "sim"                                      # [cite: 685]
        
        def run_mc(profile, seed):                          # [cite: 686]
            return run_sim_monte_carlo(                     # [cite: 687]
                args.data, args.model, profile, ekf_perturb, # [cite: 689]
                args.trials, args.device, seed               # [cite: 690]
            )                                               # [cite: 688]
    else:                                                   # [cite: 691]
        if args.csv is None:                                # [cite: 692]
            sys.exit("[MC] --csv is required for --dataset real") # [cite: 693]
        base_profile = RealSensorNoiseProfile(
            vx_mps=args.noise_vx,                           # [cite: 694, 695]
            vy_mps=args.noise_vy,                           # [cite: 696]
            dpsi_radps=args.noise_yaw_rate,                 # [cite: 697]
            ax_mps2=args.noise_ax,                          # [cite: 698]
            ay_mps2=args.noise_ay,                          # [cite: 699]
            deltawheel_rad=args.noise_delta,                # [cite: 700]
            TwheelRL_Nm=args.noise_torque,                  # [cite: 702]
            TwheelRR_Nm=args.noise_torque,                  # [cite: 703]
            pBrakeF_bar=args.noise_brake,                   # [cite: 704]
            pBrakeR_bar=args.noise_brake,                   # [cite: 705]
        )                                                   # [cite: 701]
        label = Path(args.csv).stem                         # [cite: 706]
        branch = "real"                                     # [cite: 707]
        
        def run_mc(profile, seed):                          # [cite: 708]
            return run_real_monte_carlo(                    # [cite: 709]
                args.csv, args.params_file, args.vehicle_config, # [cite: 711]
                args.model, profile, ekf_perturb,           # [cite: 712]
                args.trials, args.device, seed              # [cite: 713]
            )                                               # [cite: 710]

    # Evaluate singular standalone baseline threshold pass     # [cite: 714]
    if args.noise_sweep is None:                            # [cite: 715]
        print(f"\n[MC] Running {args.trials} trials (scale x1.0, branch={branch})") # [cite: 716]
        results = run_mc(base_profile, args.seed)           # [cite: 717]
        stats = summarise(results)                          # [cite: 718]
        print_summary(stats, base_profile.as_dict(), args.trials, branch) # [cite: 719]
        
        json_path = out_dir / f"mc_summary_{label}.json"     # [cite: 720]
        with open(json_path, "w") as f:                     # [cite: 721]
            json.dump({
                "branch": branch, "n_trials": args.trials,  # [cite: 722]
                "noise_profile": base_profile.as_dict(), "stats": stats # [cite: 723, 724]
            }, f, indent=2)
        print(f"[MC] Saved JSON -> {json_path}")             # [cite: 725]
        
        # Render the newly optimized 2x3 panel visualization matrix layout
        plot_distributions(results, stats, out_dir / f"mc_distributions_{label}.png", title_suffix=f" - {branch}/{label}") # [cite: 726, 727, 728]
        plot_convergence(results, out_dir / f"mc_convergence_{label}.png") # [cite: 729]
        
    # Evaluate a multi-step sensitivity amplification sweep step # [cite: 730]
    else:                                                   # [cite: 731]
        sweep_results: dict[float, list[TrialResult]] = {}  # [cite: 732]
        all_stats: dict[float, dict] = {}                   # [cite: 733]
        
        for scale in sorted(args.noise_sweep):              # [cite: 734]
            scaled = base_profile.scale(scale)              # [cite: 735]
            print(f"\n[MC] Sweep: noise x {scale:.2f} | {args.trials} trials | branch={branch}") # [cite: 736]
            res = run_mc(scaled, args.seed)                 # [cite: 737]
            sweep_results[scale] = res                      # [cite: 738]
            s = summarise(res)                              # [cite: 739]
            all_stats[scale] = s                            # [cite: 740]
            print_summary(s, scaled.as_dict(), args.trials, branch) # [cite: 741]
            
            plot_distributions(
                res, s, out_dir / f"mc_distributions_{label}_x{scale:.1f}.png", title_suffix=f" - noise x{scale:.1f}" # [cite: 742, 744, 745, 746]
            )                                               # [cite: 743]
            
        json_path = out_dir / f"mc_sweep_{label}.json"      # [cite: 747]
        with open(json_path, "w") as f:                     # [cite: 748]
            json.dump({                                     # [cite: 749]
                "branch": branch, "n_trials": args.trials,  # [cite: 750]
                "scales": {str(k): {"noise": base_profile.scale(k).as_dict(), "stats": v} for k, v in all_stats.items()} # [cite: 751, 752, 753]
            }, f, indent=2)                                 # [cite: 754]
        print(f"\n[MC] Saved sweep JSON -> {json_path}")     # [cite: 755]
        plot_noise_sweep(sweep_results, out_dir / f"mc_sweep_{label}.png") # [cite: 756]
        
    print(f"\n[MC] All outputs saved to: {out_dir.resolve()}") # [cite: 757]


if __name__ == "__main__":                                  # [cite: 758]
    main()                                                  # [cite: 759]
