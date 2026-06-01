from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SENSOR_COLUMNS = (
    "vx_mps",
    "vy_mps",
    "dpsi_radps",
    "ax_mps2",
    "ay_mps2",
    "deltawheel_rad",
    "TwheelRL_Nm",
    "TwheelRR_Nm",
    "pBrakeF_bar",
    "pBrakeR_bar",
)


@dataclass
class VehiclePhysics:
    sample_time_s: float
    mass_kg: float
    iz_kgm2: float
    lf_m: float
    lr_m: float
    track_front_m: float
    track_rear_m: float
    cg_height_m: float
    gravity_mps2: float
    force_clip_n: float
    left_right_split_clip: float
    rear_torque_split_gain: float


@dataclass
class EkfSettings:
    beta_std0: float
    yaw_rate_std0: float
    delta_fy_bias_std0: float
    ay_bias_std0: float
    q_beta: float
    q_yaw_rate: float
    q_delta_fy_bias: float
    q_ay_bias: float
    r_yaw_rate: float
    r_ay: float


@dataclass
class RealSensorNNSettings:
    batch_size: int
    epochs: int
    learning_rate: float
    input_timesteps: int
    input_shape: int
    output_shape: int
    val_split: float
    test_split: float


@dataclass
class RealSensorDataset:
    sensor_matrix: np.ndarray
    beta_ref_rad: np.ndarray
    yaw_rate_rps: np.ndarray
    yaw_acc_rps2: np.ndarray
    ay_mps2: np.ndarray
    wheel_force_targets_n: np.ndarray
    axle_force_targets_n: np.ndarray
    time_s: np.ndarray


def rmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference_arr = np.asarray(reference, dtype=float)
    estimate_arr = np.asarray(estimate, dtype=float)
    return float(np.sqrt(np.mean((reference_arr - estimate_arr) ** 2)))


def load_project_configs(
    nn_params_path: str | Path,
    vehicle_config_path: str | Path,
) -> tuple[RealSensorNNSettings, VehiclePhysics, EkfSettings]:
    with open(nn_params_path, "rb") as f:
        nn_cfg = tomllib.load(f)
    with open(vehicle_config_path, "rb") as f:
        vehicle_cfg = tomllib.load(f)

    nn = RealSensorNNSettings(
        batch_size=int(nn_cfg["NeuralNetwork_Settings"]["batch_size"]),
        epochs=int(nn_cfg["NeuralNetwork_Settings"]["epochs"]),
        learning_rate=float(nn_cfg["NeuralNetwork_Settings"]["learning_rate"]),
        input_timesteps=int(nn_cfg["NeuralNetwork_Settings"]["input_timesteps"]),
        input_shape=int(nn_cfg["NeuralNetwork_Settings"]["input_shape"]),
        output_shape=int(nn_cfg["NeuralNetwork_Settings"]["output_shape"]),
        val_split=float(nn_cfg["NeuralNetwork_Settings"]["val_split"]),
        test_split=float(nn_cfg["NeuralNetwork_Settings"]["test_split"]),
    )
    vehicle = VehiclePhysics(
        sample_time_s=float(vehicle_cfg["vehicle"]["sample_time_s"]),
        mass_kg=float(vehicle_cfg["vehicle"]["mass_kg"]),
        iz_kgm2=float(vehicle_cfg["vehicle"]["iz_kgm2"]),
        lf_m=float(vehicle_cfg["vehicle"]["lf_m"]),
        lr_m=float(vehicle_cfg["vehicle"]["lr_m"]),
        track_front_m=float(vehicle_cfg["vehicle"]["track_front_m"]),
        track_rear_m=float(vehicle_cfg["vehicle"]["track_rear_m"]),
        cg_height_m=float(vehicle_cfg["vehicle"]["cg_height_m"]),
        gravity_mps2=float(vehicle_cfg["vehicle"]["gravity_mps2"]),
        force_clip_n=float(vehicle_cfg["force_model"]["force_clip_n"]),
        left_right_split_clip=float(vehicle_cfg["force_model"]["left_right_split_clip"]),
        rear_torque_split_gain=float(vehicle_cfg["force_model"]["rear_torque_split_gain"]),
    )
    ekf = EkfSettings(
        beta_std0=float(vehicle_cfg["ekf"]["beta_std0"]),
        yaw_rate_std0=float(vehicle_cfg["ekf"]["yaw_rate_std0"]),
        delta_fy_bias_std0=float(vehicle_cfg["ekf"]["delta_fy_bias_std0"]),
        ay_bias_std0=float(vehicle_cfg["ekf"]["ay_bias_std0"]),
        q_beta=float(vehicle_cfg["ekf"]["q_beta"]),
        q_yaw_rate=float(vehicle_cfg["ekf"]["q_yaw_rate"]),
        q_delta_fy_bias=float(vehicle_cfg["ekf"]["q_delta_fy_bias"]),
        q_ay_bias=float(vehicle_cfg["ekf"]["q_ay_bias"]),
        r_yaw_rate=float(vehicle_cfg["ekf"]["r_yaw_rate"]),
        r_ay=float(vehicle_cfg["ekf"]["r_ay"]),
    )
    return nn, vehicle, ekf


def load_real_sensor_csv(csv_path: str | Path, vehicle: VehiclePhysics) -> RealSensorDataset:
    df = pd.read_csv(csv_path)
    df = df.rename(columns={df.columns[0]: "vx_mps"})
    sensor_matrix = df.loc[:, SENSOR_COLUMNS].to_numpy(dtype=np.float32)

    vx = sensor_matrix[:, 0].astype(float)
    vy = sensor_matrix[:, 1].astype(float)
    yaw_rate = sensor_matrix[:, 2].astype(float)
    ay = sensor_matrix[:, 4].astype(float)
    steer = sensor_matrix[:, 5].astype(float)
    torque_rl = sensor_matrix[:, 6].astype(float)
    torque_rr = sensor_matrix[:, 7].astype(float)

    beta_ref = np.arctan2(vy, np.maximum(np.abs(vx), 0.5))
    yaw_acc = np.gradient(yaw_rate, vehicle.sample_time_s)
    time_s = np.arange(len(sensor_matrix), dtype=float) * vehicle.sample_time_s

    total_lat = vehicle.mass_kg * ay
    front_body_lat = (vehicle.iz_kgm2 * yaw_acc + vehicle.lr_m * total_lat) / (vehicle.lf_m + vehicle.lr_m)
    rear_lat = total_lat - front_body_lat
    front_tire_lat = front_body_lat / np.clip(np.cos(steer), 0.7, None)

    wheel_force_targets = build_twintrack_wheel_force_targets(
        front_tire_lat=front_tire_lat,
        rear_lat=rear_lat,
        ay_mps2=ay,
        torque_rl_nm=torque_rl,
        torque_rr_nm=torque_rr,
        vehicle=vehicle,
    )
    axle_targets = np.column_stack(
        [
            wheel_force_targets[:, 0] + wheel_force_targets[:, 1],
            wheel_force_targets[:, 2] + wheel_force_targets[:, 3],
        ]
    ).astype(np.float32)

    return RealSensorDataset(
        sensor_matrix=sensor_matrix,
        beta_ref_rad=beta_ref.astype(np.float32),
        yaw_rate_rps=yaw_rate.astype(np.float32),
        yaw_acc_rps2=yaw_acc.astype(np.float32),
        ay_mps2=ay.astype(np.float32),
        wheel_force_targets_n=wheel_force_targets.astype(np.float32),
        axle_force_targets_n=axle_targets,
        time_s=time_s.astype(np.float32),
    )


def slice_real_sensor_dataset(dataset: RealSensorDataset, max_samples: int | None) -> RealSensorDataset:
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset.time_s):
        return dataset
    return RealSensorDataset(
        sensor_matrix=dataset.sensor_matrix[:max_samples].copy(),
        beta_ref_rad=dataset.beta_ref_rad[:max_samples].copy(),
        yaw_rate_rps=dataset.yaw_rate_rps[:max_samples].copy(),
        yaw_acc_rps2=dataset.yaw_acc_rps2[:max_samples].copy(),
        ay_mps2=dataset.ay_mps2[:max_samples].copy(),
        wheel_force_targets_n=dataset.wheel_force_targets_n[:max_samples].copy(),
        axle_force_targets_n=dataset.axle_force_targets_n[:max_samples].copy(),
        time_s=dataset.time_s[:max_samples].copy(),
    )


def estimate_split_targets(
    ay_mps2: np.ndarray,
    torque_rl_nm: np.ndarray,
    torque_rr_nm: np.ndarray,
    vehicle: VehiclePhysics,
) -> tuple[np.ndarray, np.ndarray]:
    wheelbase = vehicle.lf_m + vehicle.lr_m
    front_axle_load = vehicle.mass_kg * vehicle.gravity_mps2 * vehicle.lr_m / wheelbase
    rear_axle_load = vehicle.mass_kg * vehicle.gravity_mps2 * vehicle.lf_m / wheelbase

    front_transfer = vehicle.mass_kg * ay_mps2 * vehicle.cg_height_m * (vehicle.lr_m / wheelbase) / vehicle.track_front_m
    rear_transfer = vehicle.mass_kg * ay_mps2 * vehicle.cg_height_m * (vehicle.lf_m / wheelbase) / vehicle.track_rear_m

    front_split = np.clip(front_transfer / np.maximum(front_axle_load, 1.0), -vehicle.left_right_split_clip, vehicle.left_right_split_clip)
    rear_split = np.clip(rear_transfer / np.maximum(rear_axle_load, 1.0), -vehicle.left_right_split_clip, vehicle.left_right_split_clip)

    torque_scale = np.maximum(np.max(np.abs(torque_rr_nm - torque_rl_nm)), 1.0)
    rear_split = np.clip(
        rear_split + vehicle.rear_torque_split_gain * np.tanh((torque_rr_nm - torque_rl_nm) / torque_scale),
        -vehicle.left_right_split_clip,
        vehicle.left_right_split_clip,
    )
    return front_split.astype(np.float32), rear_split.astype(np.float32)


def build_twintrack_wheel_force_targets(
    front_tire_lat: np.ndarray,
    rear_lat: np.ndarray,
    ay_mps2: np.ndarray,
    torque_rl_nm: np.ndarray,
    torque_rr_nm: np.ndarray,
    vehicle: VehiclePhysics,
) -> np.ndarray:
    front_split, rear_split = estimate_split_targets(ay_mps2, torque_rl_nm, torque_rr_nm, vehicle)

    fy_fl = 0.5 * front_tire_lat * (1.0 - front_split)
    fy_fr = 0.5 * front_tire_lat * (1.0 + front_split)
    fy_rl = 0.5 * rear_lat * (1.0 - rear_split)
    fy_rr = 0.5 * rear_lat * (1.0 + rear_split)

    wheel_forces = np.column_stack([fy_fl, fy_fr, fy_rl, fy_rr])
    return np.clip(wheel_forces, -vehicle.force_clip_n, vehicle.force_clip_n)


def build_structured_force_targets(dataset: RealSensorDataset, vehicle: VehiclePhysics) -> np.ndarray:
    steer = dataset.sensor_matrix[:, 5].astype(float)
    front_tire_total = dataset.axle_force_targets_n[:, 0].astype(float)
    rear_total = dataset.axle_force_targets_n[:, 1].astype(float)
    front_body_total = front_tire_total * np.cos(steer)
    front_split, rear_split = estimate_split_targets(
        ay_mps2=dataset.ay_mps2.astype(float),
        torque_rl_nm=dataset.sensor_matrix[:, 6].astype(float),
        torque_rr_nm=dataset.sensor_matrix[:, 7].astype(float),
        vehicle=vehicle,
    )
    return np.column_stack([front_body_total, rear_total, front_split, rear_split]).astype(np.float32)


def build_structured_force_and_beta_targets(dataset: RealSensorDataset, vehicle: VehiclePhysics) -> np.ndarray:
    structured = build_structured_force_targets(dataset, vehicle)
    return np.column_stack([structured, dataset.beta_ref_rad.astype(np.float32)]).astype(np.float32)


def structured_forces_to_wheel_forces(
    structured_force_pred: np.ndarray,
    sensor_matrix: np.ndarray,
    vehicle: VehiclePhysics,
) -> np.ndarray:
    front_body_total = structured_force_pred[:, 0].astype(float)
    rear_total = structured_force_pred[:, 1].astype(float)
    front_split = np.clip(structured_force_pred[:, 2].astype(float), -vehicle.left_right_split_clip, vehicle.left_right_split_clip)
    rear_split = np.clip(structured_force_pred[:, 3].astype(float), -vehicle.left_right_split_clip, vehicle.left_right_split_clip)
    steer = sensor_matrix[:, 5].astype(float)
    cos_steer = np.clip(np.cos(steer), 0.7, None)
    front_tire_total = front_body_total / cos_steer

    fy_fl = 0.5 * front_tire_total * (1.0 - front_split)
    fy_fr = 0.5 * front_tire_total * (1.0 + front_split)
    fy_rl = 0.5 * rear_total * (1.0 - rear_split)
    fy_rr = 0.5 * rear_total * (1.0 + rear_split)

    wheel_forces = np.column_stack([fy_fl, fy_fr, fy_rl, fy_rr])
    return np.clip(wheel_forces, -vehicle.force_clip_n, vehicle.force_clip_n).astype(np.float32)


class SequenceStandardScaler:
    def __init__(self, mean: np.ndarray, scale: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.scale = np.where(scale > 1.0e-8, scale, 1.0).astype(np.float32)

    @classmethod
    def fit(cls, values: np.ndarray) -> "SequenceStandardScaler":
        reshaped = values.reshape(-1, values.shape[-1])
        return cls(np.mean(reshaped, axis=0), np.std(reshaped, axis=0))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean[None, None, :]) / self.scale[None, None, :]

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * self.scale[None, :] + self.mean[None, :]


class VectorStandardScaler:
    def __init__(self, mean: np.ndarray, scale: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.scale = np.where(scale > 1.0e-8, scale, 1.0).astype(np.float32)

    @classmethod
    def fit(cls, values: np.ndarray) -> "VectorStandardScaler":
        return cls(np.mean(values, axis=0), np.std(values, axis=0))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean[None, :]) / self.scale[None, :]

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * self.scale[None, :] + self.mean[None, :]


def build_sensor_windows(sensor_matrix: np.ndarray, window_size: int) -> np.ndarray:
    X = np.zeros((len(sensor_matrix), window_size, sensor_matrix.shape[1]), dtype=np.float32)
    for i in range(len(sensor_matrix)):
        start = max(0, i - window_size + 1)
        window = sensor_matrix[start : i + 1]
        if len(window) < window_size:
            pad = np.repeat(window[[0]], window_size - len(window), axis=0)
            window = np.vstack([pad, window])
        X[i] = window
    return X


class TinyRealSensorTransformer(nn.Module):
    def __init__(self, input_dim: int, window_size: int, output_dim: int = 4, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 128, dropout: float = 0.1):
        super().__init__()
        self.window_size = window_size
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, window_size, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input_proj(x) + self.pos_embed[:, : x.shape[1], :]
        encoded = self.encoder(hidden)
        return self.head(encoded[:, -1, :])


@dataclass
class RealSensorForceModelBundle:
    window_size: int
    input_scaler: SequenceStandardScaler
    target_scaler: VectorStandardScaler
    model_state: dict
    architecture: dict[str, float | int | str]

    def save(self, path: str) -> None:
        torch.save(
            {
                "window_size": self.window_size,
                "input_mean": self.input_scaler.mean,
                "input_scale": self.input_scaler.scale,
                "target_mean": self.target_scaler.mean,
                "target_scale": self.target_scaler.scale,
                "model_state": self.model_state,
                "architecture": self.architecture,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "RealSensorForceModelBundle":
        payload = torch.load(path, map_location=device, weights_only=False)
        return cls(
            window_size=int(payload["window_size"]),
            input_scaler=SequenceStandardScaler(payload["input_mean"], payload["input_scale"]),
            target_scaler=VectorStandardScaler(payload["target_mean"], payload["target_scale"]),
            model_state=payload["model_state"],
            architecture=dict(payload["architecture"]),
        )


class RealSensorForcePredictor:
    def __init__(self, bundle: RealSensorForceModelBundle, device: str = "cpu"):
        self.bundle = bundle
        arch = bundle.architecture
        self.device = torch.device(device)
        self.model = TinyRealSensorTransformer(
            input_dim=int(arch["input_dim"]),
            window_size=bundle.window_size,
            output_dim=int(arch["output_dim"]),
            d_model=int(arch["d_model"]),
            nhead=int(arch["nhead"]),
            num_layers=int(arch["num_layers"]),
            dim_feedforward=int(arch["dim_feedforward"]),
            dropout=float(arch["dropout"]),
        ).to(self.device)
        self.model.load_state_dict(bundle.model_state)
        self.model.eval()

    def predict(self, sensor_windows: np.ndarray) -> np.ndarray:
        X_scaled = self.bundle.input_scaler.transform(sensor_windows.astype(np.float32))
        with torch.no_grad():
            output_scaled = self.model(torch.from_numpy(X_scaled).to(self.device)).cpu().numpy()
        return self.bundle.target_scaler.inverse_transform(output_scaled).astype(np.float32)


def model_output_to_wheel_forces(
    model_output: np.ndarray,
    sensor_matrix: np.ndarray,
    bundle: RealSensorForceModelBundle,
    vehicle: VehiclePhysics,
) -> np.ndarray:
    target_mode = str(bundle.architecture.get("target_mode", "direct_wheel_forces"))
    if target_mode.startswith("structured_body_force_and_split"):
        return structured_forces_to_wheel_forces(model_output[:, :4], sensor_matrix, vehicle)
    return np.clip(model_output[:, :4], -vehicle.force_clip_n, vehicle.force_clip_n).astype(np.float32)


def model_output_to_beta_measurement(model_output: np.ndarray) -> np.ndarray | None:
    if model_output.shape[1] < 5:
        return None
    return model_output[:, 4].astype(np.float32)


def train_force_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    nn_cfg: RealSensorNNSettings,
    target_mode: str = "direct_wheel_forces",
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[RealSensorForceModelBundle, dict[str, list[float]]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_scaler = SequenceStandardScaler.fit(X_train)
    target_scaler = VectorStandardScaler.fit(y_train)
    X_train_scaled = input_scaler.transform(X_train).astype(np.float32)
    X_val_scaled = input_scaler.transform(X_val).astype(np.float32)
    y_train_scaled = target_scaler.transform(y_train).astype(np.float32)
    y_val_scaled = target_scaler.transform(y_val).astype(np.float32)

    model = TinyRealSensorTransformer(input_dim=nn_cfg.input_shape, window_size=nn_cfg.input_timesteps, output_dim=y_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate or nn_cfg.learning_rate, weight_decay=1.0e-4)
    criterion = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_scaled), torch.from_numpy(y_train_scaled)),
        batch_size=batch_size or nn_cfg.batch_size,
        shuffle=True,
    )
    X_val_t = torch.from_numpy(X_val_scaled).to(device)
    y_val_t = torch.from_numpy(y_val_scaled).to(device)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None

    for epoch in range(epochs or nn_cfg.epochs):
        model.train()
        batch_losses: list[float] = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(X_val_t), y_val_t).item())
        train_loss = float(np.mean(batch_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == (epochs or nn_cfg.epochs) - 1:
            print(f"Epoch {epoch + 1:4d}/{epochs or nn_cfg.epochs}: train_loss = {train_loss:.6f}, val_loss = {val_loss:.6f}")

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    bundle = RealSensorForceModelBundle(
        window_size=nn_cfg.input_timesteps,
        input_scaler=input_scaler,
        target_scaler=target_scaler,
        model_state=best_state,
        architecture={
            "input_dim": nn_cfg.input_shape,
            "output_dim": int(y_train.shape[1]),
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 128,
            "dropout": 0.1,
            "target_mode": target_mode,
        },
    )
    return bundle, history


def clip_ekf_state(state: np.ndarray) -> np.ndarray:
    clipped = state.copy()
    clipped[0] = float(np.clip(clipped[0], -6.0, 6.0))
    clipped[1] = float(np.clip(clipped[1], -1.8, 1.8))
    clipped[2] = float(np.clip(clipped[2], -4000.0, 4000.0))
    clipped[3] = float(np.clip(clipped[3], -4.0, 4.0))
    return clipped


def numerical_state_jacobian(func, state: np.ndarray, epsilon: float = 1.0e-5) -> np.ndarray:
    base = func(state)
    jacobian = np.zeros((len(base), len(state)), dtype=float)
    for idx in range(len(state)):
        perturbed = state.copy()
        perturbed[idx] += epsilon
        jacobian[:, idx] = (func(perturbed) - base) / epsilon
    return jacobian


def front_rear_body_forces_from_wheels(wheel_forces_n: np.ndarray, steering_wheel_rad: float) -> tuple[float, float]:
    """Collapse four wheel forces into bicycle-model front/rear body forces."""
    fy_fl, fy_fr, fy_rl, fy_rr = [float(v) for v in wheel_forces_n]
    front_body = (fy_fl + fy_fr) * np.cos(float(steering_wheel_rad))
    rear_body = fy_rl + fy_rr
    return front_body, rear_body


def vy_to_beta(vy_mps: np.ndarray, vx_mps: np.ndarray) -> np.ndarray:
    vy_arr = np.asarray(vy_mps, dtype=float)
    vx_arr = np.asarray(vx_mps, dtype=float)
    return np.arctan2(vy_arr, np.maximum(np.abs(vx_arr), 0.5))


def build_pseudo_vy_measurement(sensor_matrix: np.ndarray, sample_time_s: float) -> np.ndarray:
    vx = sensor_matrix[:, 0].astype(float)
    yaw_rate = sensor_matrix[:, 2].astype(float)
    ay = sensor_matrix[:, 4].astype(float)
    vy_pseudo = np.zeros(len(sensor_matrix), dtype=float)
    for i in range(1, len(sensor_matrix)):
        vy_dot = ay[i - 1] - vx[i - 1] * yaw_rate[i - 1]
        vy_pseudo[i] = vy_pseudo[i - 1] + sample_time_s * vy_dot
    return vy_pseudo


def run_real_sensor_force_ekf(
    dataset: RealSensorDataset,
    wheel_force_pred: np.ndarray,
    vehicle: VehiclePhysics,
    ekf_cfg: EkfSettings,
    beta_meas: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    vx_series = dataset.sensor_matrix[:, 0].astype(float)
    vy_series = dataset.sensor_matrix[:, 1].astype(float)
    vx_safe0 = max(abs(vx_series[0]), 1.0)
    beta0 = float(np.arctan2(vy_series[0], vx_safe0))
    if beta_meas is None:
        beta_meas = vy_to_beta(vy_series, vx_series).astype(float)
    beta_std0 = max(0.05, ekf_cfg.beta_std0)
    q_beta = max(0.01, ekf_cfg.q_beta)
    r_beta = max(0.05, 2.0 * beta_std0)

    x = np.array([beta0, dataset.yaw_rate_rps[0], 0.0, 0.0], dtype=float)
    P = np.diag([beta_std0**2, ekf_cfg.yaw_rate_std0**2, ekf_cfg.delta_fy_bias_std0**2, ekf_cfg.ay_bias_std0**2])
    Q = np.diag([q_beta**2, ekf_cfg.q_yaw_rate**2, ekf_cfg.q_delta_fy_bias**2, ekf_cfg.q_ay_bias**2])
    R = np.diag([ekf_cfg.r_yaw_rate**2, ekf_cfg.r_ay**2, r_beta**2])

    estimates = [x.copy()]
    innovations = [np.zeros(3, dtype=float)]

    for i in range(1, len(dataset.time_s)):
        dt = float(vehicle.sample_time_s)
        vx_mps = float(dataset.sensor_matrix[i, 0])
        steer = float(dataset.sensor_matrix[i, 5])
        front_body, rear_body = front_rear_body_forces_from_wheels(wheel_force_pred[i], steer)

        def fx_local(state: np.ndarray) -> np.ndarray:
            beta_rad, yaw_rate, delta_fy_bias, _ = state
            _ = beta_rad
            total_lat = front_body + rear_body + float(delta_fy_bias)
            vx_safe = max(abs(vx_mps), 0.5)
            beta_dot = total_lat / (vehicle.mass_kg * vx_safe) - float(yaw_rate)
            yaw_dot = (vehicle.lf_m * front_body - vehicle.lr_m * rear_body) / vehicle.iz_kgm2
            return clip_ekf_state(state + dt * np.array([beta_dot, yaw_dot, 0.0, 0.0], dtype=float))

        def hx_local(state: np.ndarray) -> np.ndarray:
            beta_rad, yaw_rate, delta_fy_bias, ay_bias = state
            _ = beta_rad
            total_lat = front_body + rear_body + float(delta_fy_bias)
            ay_pred = total_lat / vehicle.mass_kg + float(ay_bias)
            return np.array([float(yaw_rate), ay_pred, float(beta_rad)], dtype=float)

        x_pred = fx_local(x)
        F = numerical_state_jacobian(fx_local, x)
        P_pred = F @ P @ F.T + Q

        z = np.array(
            [
                float(dataset.yaw_rate_rps[i]),
                float(dataset.ay_mps2[i]),
                float(beta_meas[i]),
            ],
            dtype=float,
        )
        innovation = z - hx_local(x_pred)
        H = numerical_state_jacobian(hx_local, x_pred)
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        x = clip_ekf_state(x_pred + K @ innovation)
        P = (np.eye(4) - K @ H) @ P_pred
        P = 0.5 * (P + P.T)

        estimates.append(x.copy())
        innovations.append(innovation.copy())

    est = np.asarray(estimates)
    inn = np.asarray(innovations)
    return {
        "estimates": est,
        "beta_estimates": est[:, 0].copy(),
        "innovations": inn,
        "beta_rmse": rmse(dataset.beta_ref_rad, est[:, 0]),
        "yaw_rmse": rmse(dataset.yaw_rate_rps, est[:, 1]),
        "force_rmse": float(np.sqrt(np.mean((dataset.wheel_force_targets_n - wheel_force_pred) ** 2))),
    }


def save_real_sensor_summary_plot(
    dataset: RealSensorDataset,
    ekf_result: dict[str, np.ndarray | float],
    wheel_force_pred: np.ndarray,
    plot_file: str,
    title: str,
) -> None:
    estimates = np.asarray(ekf_result["estimates"])
    beta_est = np.asarray(ekf_result["beta_estimates"])
    fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True)

    axes[0].plot(dataset.time_s, dataset.beta_ref_rad, "k", linewidth=2, label="beta proxy (atan2(vy,vx))")
    axes[0].plot(dataset.time_s, beta_est, label="beta est from vy")
    axes[0].set_ylabel("beta (rad)")
    axes[0].set_title(f"{title}: beta")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(dataset.time_s, dataset.yaw_rate_rps, "k", linewidth=2, label="yaw rate meas")
    axes[1].plot(dataset.time_s, estimates[:, 1], label="yaw rate est")
    axes[1].set_ylabel("yaw rate (rad/s)")
    axes[1].set_title(f"{title}: yaw rate")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(dataset.time_s, dataset.wheel_force_targets_n[:, 0] + dataset.wheel_force_targets_n[:, 1], "k", linewidth=2, label="front axle target")
    axes[2].plot(dataset.time_s, wheel_force_pred[:, 0] + wheel_force_pred[:, 1], label="front axle pred")
    axes[2].plot(dataset.time_s, dataset.wheel_force_targets_n[:, 2] + dataset.wheel_force_targets_n[:, 3], "k--", linewidth=2, label="rear axle target")
    axes[2].plot(dataset.time_s, wheel_force_pred[:, 2] + wheel_force_pred[:, 3], label="rear axle pred")
    axes[2].set_ylabel("axle force (N)")
    axes[2].set_title(f"{title}: axle force sums")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(ncol=2)

    axes[3].plot(dataset.time_s, np.abs(beta_est - dataset.beta_ref_rad), label="|beta error|")
    axes[3].plot(dataset.time_s, np.abs(estimates[:, 1] - dataset.yaw_rate_rps), label="|yaw rate error|")
    axes[3].set_xlabel("time (s)")
    axes[3].set_ylabel("error")
    axes[3].set_title(f"{title}: error graph")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    fig.tight_layout()
    fig.savefig(plot_file, dpi=180)
    plt.close(fig)
