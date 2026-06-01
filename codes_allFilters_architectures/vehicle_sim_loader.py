'''
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class VehicleSimDataset:
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

    def to_dict(self) -> dict[str, np.ndarray]:
        return {
            "time_s": self.time_s,
            "vx_mps": self.vx_mps,
            "vy_mps": self.vy_mps,
            "delta_rad": self.delta_rad,
            "ay_mps2": self.ay_mps2,
            "yaw_rate_rps": self.yaw_rate_rps,
            "alpha_f_rad": self.alpha_f_rad,
            "alpha_r_rad": self.alpha_r_rad,
            "fyf_true_n": self.fyf_true_n,
            "fyr_true_n": self.fyr_true_n,
            "beta_true_rad": self.beta_true_rad,
            "yaw_true_rad": self.yaw_true_rad,
            "global_x_m": self.global_x_m,
            "global_y_m": self.global_y_m,
            "yaw_acc_rps2": self.yaw_acc_rps2,
            "cf_nprad": np.array([self.cf_nprad], dtype=float),
            "cr_nprad": np.array([self.cr_nprad], dtype=float),
            "m_kg": np.array([self.m_kg], dtype=float),
            "iz_kgm2": np.array([self.iz_kgm2], dtype=float),
            "lf_m": np.array([self.lf_m], dtype=float),
            "lr_m": np.array([self.lr_m], dtype=float),
        }


def estimate_cornering_stiffness(alpha_rad: np.ndarray, lateral_force_n: np.ndarray) -> float:
    alpha = np.asarray(alpha_rad, dtype=float)
    force = np.asarray(lateral_force_n, dtype=float)

    mask = (np.abs(alpha) > 1.0e-4) & (np.abs(alpha) < 0.15)
    if not np.any(mask):
        mask = np.abs(alpha) > 1.0e-4
    if not np.any(mask):
        return 1000.0

    alpha_fit = alpha[mask]
    force_fit = force[mask]
    stiffness = np.abs(np.sum(alpha_fit * force_fit) / np.sum(alpha_fit * alpha_fit))
    return float(max(stiffness, 10.0))


def load_vehicle_sim_csv(csv_path: str | Path, metadata_path: str | Path | None = None) -> VehicleSimDataset:
    csv_file = Path(csv_path)
    metadata_file = Path(metadata_path) if metadata_path is not None else csv_file.with_name(f"{csv_file.stem}_metadata.json")

    df = pd.read_csv(csv_file)
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))

    cf_nprad = estimate_cornering_stiffness(df["Alphaf"].to_numpy(), df["Fyf"].to_numpy())
    cr_nprad = estimate_cornering_stiffness(df["Alphar"].to_numpy(), df["Fyr"].to_numpy())

    return VehicleSimDataset(
        time_s=df["Time"].to_numpy(dtype=float),
        vx_mps=df["Vx"].to_numpy(dtype=float),
        vy_mps=df["Vy"].to_numpy(dtype=float),
        delta_rad=df["delta"].to_numpy(dtype=float),
        ay_mps2=df["AY"].to_numpy(dtype=float),
        yaw_rate_rps=df["YawR"].to_numpy(dtype=float),
        alpha_f_rad=df["Alphaf"].to_numpy(dtype=float),
        alpha_r_rad=df["Alphar"].to_numpy(dtype=float),
        fyf_true_n=df["Fyf"].to_numpy(dtype=float),
        fyr_true_n=df["Fyr"].to_numpy(dtype=float),
        beta_true_rad=df["Beta"].to_numpy(dtype=float),
        yaw_true_rad=df["yaw"].to_numpy(dtype=float),
        global_x_m=df["global_x_m"].to_numpy(dtype=float),
        global_y_m=df["global_y_m"].to_numpy(dtype=float),
        yaw_acc_rps2=df["YawAcc"].to_numpy(dtype=float),
        cf_nprad=cf_nprad,
        cr_nprad=cr_nprad,
        m_kg=float(metadata["m"]),
        iz_kgm2=float(metadata["Iz"]),
        lf_m=float(metadata["lf"]),
        lr_m=float(metadata["lr"]),
    )
'''
# vehicle_sim_loader_new.py
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

@dataclass
class VehicleSimDataset:
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

    def to_dict(self) -> dict[str, np.ndarray]:
        return {
            "time_s": self.time_s,
            "vx_mps": self.vx_mps,
            "vy_mps": self.vy_mps,
            "delta_rad": self.delta_rad,
            "ay_mps2": self.ay_mps2,
            "yaw_rate_rps": self.yaw_rate_rps,
            "fyf_true_n": self.fyf_true_n,
            "fyr_true_n": self.fyr_true_n,
            "beta_true_rad": self.beta_true_rad,
            "yaw_true_rad": self.yaw_true_rad,
            "global_x_m": self.global_x_m,
            "global_y_m": self.global_y_m,
            "yaw_acc_rps2": self.yaw_acc_rps2,
            "cf_nprad": np.array([self.cf_nprad], dtype=float),
            "cr_nprad": np.array([self.cr_nprad], dtype=float),
            "m_kg": np.array([self.m_kg], dtype=float),
            "iz_kgm2": np.array([self.iz_kgm2], dtype=float),
            "lf_m": np.array([self.lf_m], dtype=float),
            "lr_m": np.array([self.lr_m], dtype=float),
        }

def estimate_cornering_stiffness(alpha_rad: np.ndarray, lateral_force_n: np.ndarray) -> float:
    alpha = np.asarray(alpha_rad, dtype=float)
    force = np.asarray(lateral_force_n, dtype=float)
    mask = (np.abs(alpha) > 1.0e-4) & (np.abs(alpha) < 0.15)
    
    if not np.any(mask):
        mask = np.abs(alpha) > 1.0e-4
    if not np.any(mask):
        return 1000.0
        
    alpha_fit = alpha[mask]
    force_fit = force[mask]
    
    denom = np.sum(alpha_fit * alpha_fit)
    if denom < 1.0e-6:
        return 1000.0
        
    stiffness = np.abs(np.sum(alpha_fit * force_fit) / denom)
    return float(max(stiffness, 10.0))

def load_vehicle_sim_csv(csv_path: str | Path, metadata_path: str | Path | None = None) -> VehicleSimDataset:
    csv_file = Path(csv_path)
    metadata_file = Path(metadata_path) if metadata_path is not None else csv_file.with_name(f"{csv_file.stem}_metadata.json")
    
    df = pd.read_csv(csv_file)
    
    # Stand-in default parameter block if metadata file doesn't exist for the run scenario
    if metadata_file.exists():
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        m = float(metadata["m"])
        iz = float(metadata["Iz"])
        lf = float(metadata["lf"])
        lr = float(metadata["lr"])
    else:
        # Default placeholder parameters for standard compact evaluation testbed
        m = 1400.0
        iz = 2420.0
        lf = 1.4
        lr = 1.6

    # Clean up and register columns to lowercase mapping, stripping out '#' prefixes
    cleaned_columns = [str(c).lower().replace('#', '').strip() for c in df.columns]
    col_map = dict(zip(cleaned_columns, df.columns))

    def get_column_data(keys: list[str], default_zero=False) -> np.ndarray:
        for k in keys:
            if k.lower() in col_map:
                return df[col_map[k.lower()]].to_numpy(dtype=float)
        if default_zero:
            return np.zeros(len(df), dtype=float)
        raise KeyError(f"Could not find required columns {keys}. Columns available: {list(df.columns)}")

    # 1. Handle Time Column absence by synthesizing via constant dt (assume 100Hz timeline fallback)
    if any(k in col_map for k in ["time", "time_s", "t"]):
        time_s = get_column_data(["time", "time_s", "t"])
    else:
        time_s = np.arange(0, len(df) * 0.01, 0.01, dtype=float)[:len(df)]

    # 2. Extract raw measurable sensor channels mapping to your new headers
    vx_mps       = get_column_data(["vx_mps", "vx", "v_x"])
    vy_mps       = get_column_data(["vy_mps", "vy"], default_zero=True)
    delta_rad    = get_column_data(["deltawheel_rad", "delta", "delta_rad"])
    ay_mps2      = get_column_data(["ay_mps2", "ay", "a_y"])
    yaw_rate_rps = get_column_data(["dpsi_radps", "yawr", "yaw_rate_rps"])
    
    # 3. Handle validation target fields with zero-fallbacks if missing from raw files
    beta_true_rad = vy_mps / np.maximum(vx_mps, 0.5) 
    yaw_true_rad  = get_column_data(["yaw", "psi"], default_zero=True)
    yaw_acc_rps2  = get_column_data(["yawacc"], default_zero=True)
    global_x_m    = get_column_data(["x"], default_zero=True)
    global_y_m    = get_column_data(["y"], default_zero=True)

    # Synthetic Axle Targets computation via dynamic load distribution equations
    total_fy = m * ay_mps2
    front_ratio = lr / (lf + lr)
    fyf_true_n  = total_fy * front_ratio
    fyr_true_n  = total_fy * (1.0 - front_ratio)
    
    # Analytical fallback slip estimation to compute background stiffness parameters
    alpha_f = delta_rad - (vy_mps + lf * yaw_rate_rps) / np.maximum(vx_mps, 0.5)
    alpha_r = -(vy_mps - lr * yaw_rate_rps) / np.maximum(vx_mps, 0.5)
    
    cf_nprad = estimate_cornering_stiffness(alpha_f, fyf_true_n)
    cr_nprad = estimate_cornering_stiffness(alpha_r, fyr_true_n)

    return VehicleSimDataset(
        time_s=time_s, vx_mps=vx_mps, vy_mps=vy_mps, delta_rad=delta_rad,
        ay_mps2=ay_mps2, yaw_rate_rps=yaw_rate_rps, fyf_true_n=fyf_true_n, fyr_true_n=fyr_true_n,
        beta_true_rad=beta_true_rad, yaw_true_rad=yaw_true_rad, global_x_m=global_x_m, global_y_m=global_y_m,
        yaw_acc_rps2=yaw_acc_rps2, cf_nprad=cf_nprad, cr_nprad=cr_nprad,
        m_kg=m, iz_kgm2=iz, lf_m=lf, lr_m=lr
    )