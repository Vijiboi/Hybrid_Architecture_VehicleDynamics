from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class PreparedData:
    imu_time: np.ndarray
    accel_long: np.ndarray
    yaw_rate: np.ndarray
    gps_time: np.ndarray
    gps_x: np.ndarray
    gps_y: np.ndarray
    gt_time: np.ndarray
    gt_x: np.ndarray
    gt_y: np.ndarray
    yaw0: float
    speed0: float

    @property
    def has_ground_truth(self) -> bool:
        return len(self.gt_time) > 0 and len(self.gt_x) > 0 and len(self.gt_y) > 0


def load_prepared_data(data_file: str) -> PreparedData:
    data = np.load(data_file)
    return PreparedData(
        imu_time=data["imu_time"],
        accel_long=data["accel_long"],
        yaw_rate=data["yaw_rate"],
        gps_time=data["gps_time"],
        gps_x=data["gps_x"],
        gps_y=data["gps_y"],
        gt_time=data["gt_time"],
        gt_x=data["gt_x"],
        gt_y=data["gt_y"],
        yaw0=float(data["yaw0"][0]),
        speed0=float(data["speed0"][0]),
    )


def wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def fx(x: np.ndarray, dt: float, u: np.ndarray) -> np.ndarray:
    px, py, yaw, speed = x
    accel, yaw_rate = u
    return np.array(
        [
            px + speed * np.cos(yaw) * dt + 0.5 * accel * np.cos(yaw) * dt * dt,
            py + speed * np.sin(yaw) * dt + 0.5 * accel * np.sin(yaw) * dt * dt,
            wrap_angle(yaw + yaw_rate * dt),
            speed + accel * dt,
        ],
        dtype=float,
    )


def hx(x: np.ndarray) -> np.ndarray:
    return np.array([x[0], x[1]], dtype=float)


def compute_rmse(
    est_time: np.ndarray,
    est_x: np.ndarray,
    est_y: np.ndarray,
    gt_time: np.ndarray,
    gt_x: np.ndarray,
    gt_y: np.ndarray,
) -> float:
    gt_x_interp = np.interp(est_time, gt_time, gt_x)
    gt_y_interp = np.interp(est_time, gt_time, gt_y)
    error = np.hypot(est_x - gt_x_interp, est_y - gt_y_interp)
    return float(np.sqrt(np.mean(error**2)))


def save_path_plot(
    data: PreparedData,
    estimates: np.ndarray,
    plot_file: str,
    title: str,
    label: str,
    color: str,
) -> None:
    plt.figure(figsize=(10, 8))
    if data.has_ground_truth:
        plt.plot(data.gt_y, data.gt_x, "k", linewidth=2, label="Ground truth")
    plt.scatter(data.gps_y, data.gps_x, s=4, alpha=0.25, label="GPS")
    plt.plot(estimates[:, 1], estimates[:, 0], color=color, label=label)
    plt.xlabel("Local Y / East (m)")
    plt.ylabel("Local X / North (m)")
    plt.title(title)
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_file, dpi=180)
    plt.close()


def default_covariance_plot_file(plot_file: str) -> str:
    path = Path(plot_file)
    return str(path.with_name(f"{path.stem}_covariance{path.suffix or '.png'}"))


def find_gps_outages(gps_time: np.ndarray, min_gap_sec: float = 1.0) -> list[tuple[float, float]]:
    if len(gps_time) < 2:
        return []

    outages: list[tuple[float, float]] = []
    for start, end in zip(gps_time[:-1], gps_time[1:]):
        if (end - start) > min_gap_sec:
            outages.append((float(start), float(end)))
    return outages


def save_covariance_plot(
    estimate_time: np.ndarray,
    covariance_history: np.ndarray,
    gps_time: np.ndarray,
    plot_file: str,
    title: str,
    min_gap_sec: float = 1.0,
) -> None:
    time_axis = estimate_time - estimate_time[0]
    diag = np.clip(np.diagonal(covariance_history, axis1=1, axis2=2), a_min=0.0, a_max=None)

    sigma_x = np.sqrt(diag[:, 0])
    sigma_y = np.sqrt(diag[:, 1])
    sigma_yaw = np.sqrt(diag[:, 2])
    sigma_speed = np.sqrt(diag[:, 3])
    sigma_pos = np.sqrt(diag[:, 0] + diag[:, 1])

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    outages = find_gps_outages(gps_time, min_gap_sec=min_gap_sec)

    for ax in axes:
        for outage_start, outage_end in outages:
            ax.axvspan(outage_start - estimate_time[0], outage_end - estimate_time[0], color="tab:red", alpha=0.12)
        ax.grid(True, alpha=0.3)

    axes[0].plot(time_axis, sigma_pos, label="Position sigma (combined)", color="tab:blue", linewidth=2)
    axes[0].plot(time_axis, sigma_x, label="Sigma x", color="tab:cyan", alpha=0.8)
    axes[0].plot(time_axis, sigma_y, label="Sigma y", color="tab:green", alpha=0.8)
    axes[0].set_ylabel("Position sigma (m)")
    axes[0].legend()

    axes[1].plot(time_axis, sigma_yaw, color="tab:orange", linewidth=2)
    axes[1].set_ylabel("Yaw sigma (rad)")

    axes[2].plot(time_axis, sigma_speed, color="tab:purple", linewidth=2)
    axes[2].set_ylabel("Speed sigma (m/s)")
    axes[2].set_xlabel("Time from start (s)")

    if outages:
        longest = max(outages, key=lambda span: span[1] - span[0])
        longest_gap = longest[1] - longest[0]
        fig.suptitle(f"{title}\nRed shading marks GPS outages > {min_gap_sec:.1f} s; longest gap = {longest_gap:.1f} s")
    else:
        fig.suptitle(f"{title}\nNo GPS outages longer than {min_gap_sec:.1f} s")

    plt.tight_layout()
    plt.savefig(plot_file, dpi=180)
    plt.close()
