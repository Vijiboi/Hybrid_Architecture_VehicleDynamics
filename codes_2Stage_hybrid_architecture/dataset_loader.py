import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


EARTH_RADIUS_M = 6_378_137.0


@dataclass
class PreparedDataset:
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

    def to_dict(self) -> dict[str, np.ndarray]:
        return {
            "imu_time": self.imu_time,
            "accel_long": self.accel_long,
            "yaw_rate": self.yaw_rate,
            "gps_time": self.gps_time,
            "gps_x": self.gps_x,
            "gps_y": self.gps_y,
            "gt_time": self.gt_time,
            "gt_x": self.gt_x,
            "gt_y": self.gt_y,
            "yaw0": np.array([self.yaw0], dtype=float),
            "speed0": np.array([self.speed0], dtype=float),
        }


def convert_latlon_to_local_xy(
    lat_values: np.ndarray, lon_values: np.ndarray, angle_unit: str
) -> tuple[np.ndarray, np.ndarray]:
    lat = np.asarray(lat_values, dtype=float)
    lon = np.asarray(lon_values, dtype=float)

    if angle_unit == "deg":
        lat = np.deg2rad(lat)
        lon = np.deg2rad(lon)
    elif angle_unit != "rad":
        raise ValueError(f"Unsupported angle unit: {angle_unit}")

    lat0 = float(lat[0])
    lon0 = float(lon[0])
    x_local = EARTH_RADIUS_M * (lat - lat0)
    y_local = EARTH_RADIUS_M * np.cos(lat0) * (lon - lon0)
    return x_local, y_local


def estimate_initial_heading_and_speed(
    time_values: np.ndarray, x_values: np.ndarray, y_values: np.ndarray
) -> tuple[float, float]:
    x0 = float(x_values[0])
    y0 = float(y_values[0])

    for i in range(1, len(time_values)):
        dx = float(x_values[i] - x0)
        dy = float(y_values[i] - y0)
        distance = float(np.hypot(dx, dy))
        dt = float(time_values[i] - time_values[0])
        if distance >= 2.0 and dt > 0.0:
            return float(np.arctan2(dy, dx)), float(distance / dt)

    return 0.0, 0.0


def _subset_time_window(
    frame: pd.DataFrame, time_col: str, t_start: float, t_end: float
) -> pd.DataFrame:
    return frame[(frame[time_col] >= t_start) & (frame[time_col] <= t_end)].reset_index(drop=True)


def _prepare_ground_truth_arrays(
    ground_truth: pd.DataFrame | None, time_col: str | None, x_col: str | None, y_col: str | None, t_start: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if ground_truth is None or time_col is None or x_col is None or y_col is None:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)

    gt_time = (ground_truth[time_col].to_numpy(dtype=float) - t_start).astype(float) / 1e6
    gt_x = ground_truth[x_col].to_numpy(dtype=float)
    gt_y = ground_truth[y_col].to_numpy(dtype=float)
    gt_x = gt_x - gt_x[0]
    gt_y = gt_y - gt_y[0]
    return gt_time, gt_x, gt_y


def _finalize_dataset(
    imu_time_us: np.ndarray,
    accel_long: np.ndarray,
    yaw_rate: np.ndarray,
    gps_time_us: np.ndarray,
    gps_x: np.ndarray,
    gps_y: np.ndarray,
    gt_time: np.ndarray,
    gt_x: np.ndarray,
    gt_y: np.ndarray,
    t_start: float,
) -> PreparedDataset:
    imu_time = (np.asarray(imu_time_us, dtype=float) - t_start) / 1e6
    gps_time = (np.asarray(gps_time_us, dtype=float) - t_start) / 1e6
    yaw0, speed0 = estimate_initial_heading_and_speed(gps_time, gps_x, gps_y)

    return PreparedDataset(
        imu_time=imu_time,
        accel_long=np.asarray(accel_long, dtype=float),
        yaw_rate=np.asarray(yaw_rate, dtype=float),
        gps_time=gps_time,
        gps_x=np.asarray(gps_x, dtype=float),
        gps_y=np.asarray(gps_y, dtype=float),
        gt_time=gt_time,
        gt_x=gt_x,
        gt_y=gt_y,
        yaw0=float(yaw0),
        speed0=float(speed0),
    )


def load_nclt_dataset(
    base_dir: str | Path = ".",
    duration_sec: float | None = None,
    imu_stride: int = 1,
    gps_file: str = "gps.csv",
    imu_file: str = "ms25.csv",
    gt_file: str = "groundtruth_2012-01-22.csv",
) -> PreparedDataset:
    base_path = Path(base_dir)

    gps_cols = [
        "utime",
        "fix_mode",
        "num_sats",
        "lat_rad",
        "lon_rad",
        "alt_m",
        "unused_1",
        "unused_2",
    ]
    imu_cols = [
        "utime",
        "mag_x",
        "mag_y",
        "mag_z",
        "accel_x",
        "accel_y",
        "accel_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
    ]
    gt_cols = ["utime", "x", "y", "z", "roll", "pitch", "yaw"]

    gps = pd.read_csv(base_path / gps_file, header=None, names=gps_cols)
    imu = pd.read_csv(base_path / imu_file, header=None, names=imu_cols)
    gt = pd.read_csv(
        base_path / gt_file,
        header=None,
        names=gt_cols,
        skipinitialspace=True,
        na_values=["-nan", "nan"],
        low_memory=False,
    )
    gt = gt.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)

    t_start = max(float(gps["utime"].iloc[0]), float(imu["utime"].iloc[0]), float(gt["utime"].iloc[0]))
    t_end = min(float(gps["utime"].iloc[-1]), float(imu["utime"].iloc[-1]), float(gt["utime"].iloc[-1]))

    if duration_sec is not None:
        t_end = min(t_end, t_start + duration_sec * 1e6)

    gps = _subset_time_window(gps, "utime", t_start, t_end)
    imu = _subset_time_window(imu, "utime", t_start, t_end)
    gt = _subset_time_window(gt, "utime", t_start, t_end)

    if imu_stride > 1:
        imu = imu.iloc[::imu_stride].reset_index(drop=True)

    gps_x, gps_y = convert_latlon_to_local_xy(
        gps["lat_rad"].to_numpy(dtype=float),
        gps["lon_rad"].to_numpy(dtype=float),
        angle_unit="rad",
    )
    gt_time, gt_x, gt_y = _prepare_ground_truth_arrays(gt, "utime", "x", "y", t_start)

    return _finalize_dataset(
        imu_time_us=imu["utime"].to_numpy(dtype=float),
        accel_long=imu["accel_x"].to_numpy(dtype=float),
        yaw_rate=imu["gyro_z"].to_numpy(dtype=float),
        gps_time_us=gps["utime"].to_numpy(dtype=float),
        gps_x=gps_x,
        gps_y=gps_y,
        gt_time=gt_time,
        gt_x=gt_x,
        gt_y=gt_y,
        t_start=t_start,
    )


def _read_csv_from_spec(base_dir: Path, spec: dict) -> pd.DataFrame:
    path = base_dir / spec["path"]
    header = spec.get("header", "infer")
    if header == "none":
        header = None

    frame = pd.read_csv(
        path,
        header=header,
        names=spec.get("names"),
        delimiter=spec.get("delimiter", ","),
        skiprows=spec.get("skiprows", 0),
    )
    return frame


def _prepare_time_column(frame: pd.DataFrame, time_col: str, time_scale_to_us: float) -> pd.DataFrame:
    frame = frame.copy()
    frame[time_col] = pd.to_numeric(frame[time_col], errors="coerce") * float(time_scale_to_us)
    return frame.dropna(subset=[time_col]).reset_index(drop=True)


def _prepare_xy_from_gps(frame: pd.DataFrame, gps_spec: dict) -> tuple[np.ndarray, np.ndarray]:
    coordinate_mode = gps_spec["coordinate_mode"]

    if coordinate_mode == "xy":
        x_values = pd.to_numeric(frame[gps_spec["x_col"]], errors="coerce").to_numpy(dtype=float)
        y_values = pd.to_numeric(frame[gps_spec["y_col"]], errors="coerce").to_numpy(dtype=float)
        x_values = x_values - x_values[0]
        y_values = y_values - y_values[0]
        return x_values, y_values

    if coordinate_mode == "latlon_rad":
        lat = pd.to_numeric(frame[gps_spec["lat_col"]], errors="coerce").to_numpy(dtype=float)
        lon = pd.to_numeric(frame[gps_spec["lon_col"]], errors="coerce").to_numpy(dtype=float)
        return convert_latlon_to_local_xy(lat, lon, angle_unit="rad")

    if coordinate_mode == "latlon_deg":
        lat = pd.to_numeric(frame[gps_spec["lat_col"]], errors="coerce").to_numpy(dtype=float)
        lon = pd.to_numeric(frame[gps_spec["lon_col"]], errors="coerce").to_numpy(dtype=float)
        return convert_latlon_to_local_xy(lat, lon, angle_unit="deg")

    raise ValueError(f"Unsupported coordinate_mode: {coordinate_mode}")


def load_generic_csv_dataset(
    config_path: str | Path,
    base_dir: str | Path = ".",
    duration_sec: float | None = None,
    imu_stride: int = 1,
) -> PreparedDataset:
    base_path = Path(base_dir)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))

    imu_spec = config["imu"]
    gps_spec = config["gps"]
    gt_spec = config.get("ground_truth")

    imu = _read_csv_from_spec(base_path, imu_spec)
    gps = _read_csv_from_spec(base_path, gps_spec)
    gt = _read_csv_from_spec(base_path, gt_spec) if gt_spec else None

    imu = _prepare_time_column(imu, imu_spec["time_col"], imu_spec.get("time_scale_to_us", 1.0))
    gps = _prepare_time_column(gps, gps_spec["time_col"], gps_spec.get("time_scale_to_us", 1.0))
    if gt is not None:
        gt = _prepare_time_column(gt, gt_spec["time_col"], gt_spec.get("time_scale_to_us", 1.0))

    for col in [imu_spec["accel_long_col"], imu_spec["yaw_rate_col"]]:
        imu[col] = pd.to_numeric(imu[col], errors="coerce")
    imu = imu.dropna(subset=[imu_spec["accel_long_col"], imu_spec["yaw_rate_col"]]).reset_index(drop=True)

    if gt is not None:
        gt[gt_spec["x_col"]] = pd.to_numeric(gt[gt_spec["x_col"]], errors="coerce")
        gt[gt_spec["y_col"]] = pd.to_numeric(gt[gt_spec["y_col"]], errors="coerce")
        gt = gt.dropna(subset=[gt_spec["x_col"], gt_spec["y_col"]]).reset_index(drop=True)

    time_starts = [float(imu[imu_spec["time_col"]].iloc[0]), float(gps[gps_spec["time_col"]].iloc[0])]
    time_ends = [float(imu[imu_spec["time_col"]].iloc[-1]), float(gps[gps_spec["time_col"]].iloc[-1])]
    if gt is not None:
        time_starts.append(float(gt[gt_spec["time_col"]].iloc[0]))
        time_ends.append(float(gt[gt_spec["time_col"]].iloc[-1]))

    t_start = max(time_starts)
    t_end = min(time_ends)
    if duration_sec is not None:
        t_end = min(t_end, t_start + duration_sec * 1e6)

    imu = _subset_time_window(imu, imu_spec["time_col"], t_start, t_end)
    gps = _subset_time_window(gps, gps_spec["time_col"], t_start, t_end)
    if gt is not None:
        gt = _subset_time_window(gt, gt_spec["time_col"], t_start, t_end)

    if imu_stride > 1:
        imu = imu.iloc[::imu_stride].reset_index(drop=True)

    gps_x, gps_y = _prepare_xy_from_gps(gps, gps_spec)
    gt_time, gt_x, gt_y = _prepare_ground_truth_arrays(
        gt,
        gt_spec["time_col"] if gt_spec else None,
        gt_spec["x_col"] if gt_spec else None,
        gt_spec["y_col"] if gt_spec else None,
        t_start,
    )

    return _finalize_dataset(
        imu_time_us=imu[imu_spec["time_col"]].to_numpy(dtype=float),
        accel_long=imu[imu_spec["accel_long_col"]].to_numpy(dtype=float),
        yaw_rate=imu[imu_spec["yaw_rate_col"]].to_numpy(dtype=float),
        gps_time_us=gps[gps_spec["time_col"]].to_numpy(dtype=float),
        gps_x=gps_x,
        gps_y=gps_y,
        gt_time=gt_time,
        gt_x=gt_x,
        gt_y=gt_y,
        t_start=t_start,
    )
