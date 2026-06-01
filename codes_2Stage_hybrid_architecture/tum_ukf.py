from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class TUMDataset:
    time_s: np.ndarray
    vx_mps: np.ndarray
    ay_mps2: np.ndarray
    yaw_rate_rps: np.ndarray
    beta_ref_rad: np.ndarray
    delta_rad: np.ndarray
    m_kg: float
    iz_kgm2: float
    lf_m: float
    lr_m: float
    cf_nprad: float
    cr_nprad: float


def rmse(truth: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(truth) - np.asarray(estimate)) ** 2)))


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {str(col).lower().lstrip("#").strip(): col for col in df.columns}
    for candidate in candidates:
        key = candidate.lower().lstrip("#").strip()
        if key in normalized:
            return normalized[key]
    return None


def _load_metadata(csv_path: Path) -> dict[str, float]:
    sidecar = csv_path.with_name(f"{csv_path.stem}_metadata.json")
    if sidecar.exists():
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        return {k: float(v) for k, v in payload.items() if k in {"m", "Iz", "lf", "lr", "cf", "cr", "Ts"}}
    return {}


def load_tum_csv(csv_path: str | Path, sample_time_s: float = 0.008) -> TUMDataset:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    metadata = _load_metadata(csv_path)

    time_col = _find_column(df, ["time_s", "time", "t", "Time"])
    vx_col = _find_column(df, ["vx_mps", "vx"])
    ay_col = _find_column(df, ["ay_mps2", "ay", "AY"])
    yaw_col = _find_column(df, ["dpsi_radps", "yaw_rate_rps", "YawR"])
    beta_col = _find_column(df, ["beta_true_rad", "beta", "Beta"])
    delta_col = _find_column(df, ["deltawheel_rad", "delta", "Delta"])
    m_kg = metadata.get("m", 1650.0)
    iz_kgm2 = metadata.get("Iz", 3234.0)
    lf_m = metadata.get("lf", 1.4)
    lr_m = metadata.get("lr", 1.65)
    cf_nprad = metadata.get("cf", 50000.0)
    cr_nprad = metadata.get("cr", 65000.0)

    if vx_col is None or ay_col is None or yaw_col is None:
        raise KeyError("Missing required columns. Need vx, ay and yaw-rate columns in the CSV.")

    vx = df[vx_col].to_numpy(dtype=float)
    ay = df[ay_col].to_numpy(dtype=float)
    yaw_rate = df[yaw_col].to_numpy(dtype=float)
    delta = df[delta_col].to_numpy(dtype=float) if delta_col is not None else np.zeros_like(vx)
    if time_col is not None:
        time_s = df[time_col].to_numpy(dtype=float)
    else:
        time_s = np.arange(len(df), dtype=float) * sample_time_s

    if beta_col is not None:
        beta_ref = df[beta_col].to_numpy(dtype=float)
    else:
        beta_ref = np.arctan2(np.zeros_like(vx), np.maximum(np.abs(vx), 0.5))

    return TUMDataset(
        time_s=time_s,
        vx_mps=vx,
        ay_mps2=ay,
        yaw_rate_rps=yaw_rate,
        beta_ref_rad=beta_ref.astype(np.float32),
        delta_rad=delta,
        m_kg=m_kg,
        iz_kgm2=iz_kgm2,
        lf_m=lf_m,
        lr_m=lr_m,
        cf_nprad=cf_nprad,
        cr_nprad=cr_nprad,
    )


def tire_forces(beta: float, yaw_rate: float, vx_mps: float, delta_rad: float, cf_nprad: float, cr_nprad: float, lf_m: float, lr_m: float) -> tuple[float, float]:
    vx_safe = max(abs(vx_mps), 0.5)
    alpha_f = delta_rad - beta - (lf_m * yaw_rate / vx_safe)
    alpha_r = -beta + (lr_m * yaw_rate / vx_safe)
    fyf = cf_nprad * alpha_f
    fyr = cr_nprad * alpha_r
    return fyf, fyr


def state_fx(state: np.ndarray, dt: float, vx_mps: float, delta_rad: float, params: TUMDataset) -> np.ndarray:
    beta, yaw_rate = state
    fyf, fyr = tire_forces(beta, yaw_rate, vx_mps, delta_rad, params.cf_nprad, params.cr_nprad, params.lf_m, params.lr_m)
    vx_safe = max(abs(vx_mps), 0.5)
    beta_dot = (fyf + fyr) / (params.m_kg * vx_safe) - yaw_rate
    yaw_dot = (params.lf_m * fyf - params.lr_m * fyr) / params.iz_kgm2
    return np.array([beta + dt * beta_dot, yaw_rate + dt * yaw_dot], dtype=float)


def state_hx(state: np.ndarray, vx_mps: float, delta_rad: float, params: TUMDataset) -> np.ndarray:
    beta, yaw_rate = state
    fyf, fyr = tire_forces(beta, yaw_rate, vx_mps, delta_rad, params.cf_nprad, params.cr_nprad, params.lf_m, params.lr_m)
    ay_pred = (fyf + fyr) / params.m_kg
    return np.array([float(yaw_rate), ay_pred], dtype=float)


def run_tum_ukf(csv_path: str, plot_file: str, sample_time_s: float = 0.008) -> None:
    data = load_tum_csv(csv_path, sample_time_s=sample_time_s)

    points = MerweScaledSigmaPoints(n=2, alpha=0.3, beta=2.0, kappa=0.0)
    ukf = UnscentedKalmanFilter(
        dim_x=2,
        dim_z=2,
        dt=sample_time_s,
        fx=lambda state, dt, vx_mps, delta_rad, params: state_fx(state, dt, vx_mps, delta_rad, params),
        hx=lambda state, vx_mps, delta_rad, params: state_hx(state, vx_mps, delta_rad, params),
        points=points,
    )
    ukf.x = np.array([0.0, data.yaw_rate_rps[0]], dtype=float)
    ukf.P = np.diag([0.03**2, 0.05**2])
    ukf.Q = np.diag([0.002**2, 0.02**2])
    ukf.R = np.diag([0.03**2, 0.20**2])

    estimates = [ukf.x.copy()]
    for i in range(1, len(data.time_s)):
        dt = float(data.time_s[i] - data.time_s[i - 1]) if len(data.time_s) > 1 else sample_time_s
        vx = float(data.vx_mps[i])
        delta = float(data.delta_rad[i])
        ukf.predict(dt=dt, vx_mps=vx, delta_rad=delta, params=data)
        ukf.update(np.array([float(data.yaw_rate_rps[i]), float(data.ay_mps2[i])], dtype=float), vx_mps=vx, delta_rad=delta, params=data)
        estimates.append(ukf.x.copy())

    estimates = np.asarray(estimates)
    beta_rmse = rmse(data.beta_ref_rad, estimates[:, 0])
    yaw_rmse = rmse(data.yaw_rate_rps, estimates[:, 1])

    print(f"TUM UKF beta RMSE: {beta_rmse:.6f} rad")
    print(f"TUM UKF yaw-rate RMSE: {yaw_rmse:.6f} rad/s")

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    axes[0].plot(data.time_s, data.beta_ref_rad, "k", linewidth=2, label="beta ref")
    axes[0].plot(data.time_s, estimates[:, 0], label="beta est")
    axes[0].set_ylabel("beta (rad)")
    axes[0].set_title("TUM UKF: beta")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(data.time_s, data.yaw_rate_rps, "k", linewidth=2, label="yaw rate meas")
    axes[1].plot(data.time_s, estimates[:, 1], label="yaw rate est")
    axes[1].set_ylabel("yaw rate (rad/s)")
    axes[1].set_title("TUM UKF: yaw rate")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    ay_est = np.array([state_hx(estimates[i], float(data.vx_mps[i]), float(data.delta_rad[i]), data)[1] for i in range(len(data.time_s))])
    axes[2].plot(data.time_s, data.ay_mps2, "k", linewidth=2, label="ay meas")
    axes[2].plot(data.time_s, ay_est, label="ay est")
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylabel("ay (m/s^2)")
    axes[2].set_title("TUM UKF: lateral acceleration")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(plot_file, dpi=180)
    plt.close(fig)
    print(f"Saved plot to {plot_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run no-transformer UKF on the TUM dataset or a converted simulation CSV.")
    parser.add_argument("--csv", default=str(Path("..") / "testingdata" / "data_to_run.csv"))
    parser.add_argument("--plot", default="tum_ukf.png")
    parser.add_argument("--sample-time", type=float, default=0.008)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_tum_ukf(args.csv, args.plot, sample_time_s=args.sample_time)
