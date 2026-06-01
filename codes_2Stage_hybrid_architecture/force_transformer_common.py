'''
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from vehicle_filter_common import VehicleSimPreparedData
from vehicle_filter_common import reconstruct_path
from vehicle_sim_loader import load_vehicle_sim_csv

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SENSOR_FEATURE_NAMES = ("vx_mps", "delta_rad", "ay_mps2", "yaw_rate_rps", "yaw_acc_rps2")


@dataclass
class ForceModelConfig:
    window_size: int = 8
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    force_clip_n: float = 10000.0


@dataclass
class VehicleParams:
    m_kg: float
    iz_kgm2: float
    lf_m: float
    lr_m: float

    #change
    b_m: float = 1.5 
    #change_end

    @classmethod
    def from_prepared_data(cls, data: VehicleSimPreparedData) -> "VehicleParams":
        return cls(
            m_kg=float(data.m_kg),
            iz_kgm2=float(data.iz_kgm2),
            lf_m=float(data.lf_m),
            lr_m=float(data.lr_m),
            b_m=1.5,  #change here
        )


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


def collect_scenario_files(folder: Path) -> list[Path]:
    def scenario_key(path: Path) -> tuple[int, str]:
        prefix = path.stem.split("_", 1)[0]
        return (int(prefix), path.name) if prefix.isdigit() else (10_000, path.name)

    return sorted((path for path in folder.glob("*.csv") if path.name != "scenario_summary.csv"), key=scenario_key)


def select_files_by_name(all_files: list[Path], requested_names: list[str] | None) -> list[Path]:
    if not requested_names:
        return all_files
    lookup = {path.name: path for path in all_files}
    selected: list[Path] = []
    missing: list[str] = []
    for name in requested_names:
        if name in lookup:
            selected.append(lookup[name])
        else:
            missing.append(name)
    if missing:
        raise ValueError(f"Requested files not found: {missing}")
    return selected


def scenario_sensor_matrix(data: VehicleSimPreparedData) -> np.ndarray:
    return np.column_stack(
        [
            data.vx_mps,
            data.delta_rad,
            data.ay_mps2,
            data.yaw_rate_rps,
            data.yaw_acc_rps2,
        ]
    ).astype(np.float32)


def build_sensor_window_array(sensor_matrix: np.ndarray, index: int, window_size: int) -> np.ndarray:
    start = max(0, index - window_size + 1)
    window = sensor_matrix[start : index + 1]
    if len(window) < window_size:
        pad = np.repeat(window[[0]], window_size - len(window), axis=0)
        window = np.vstack([pad, window])
    return window.astype(np.float32)


def build_force_training_arrays(data: VehicleSimPreparedData, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    sensors = scenario_sensor_matrix(data)
    X = np.zeros((len(data.time_s), window_size, sensors.shape[1]), dtype=np.float32)
    y = build_wheel_force_targets(data).astype(np.float32)
    for i in range(len(data.time_s)):
        X[i] = build_sensor_window_array(sensors, i, window_size)
    return X, y


def build_wheel_force_targets(data: VehicleSimPreparedData) -> np.ndarray:
    
    ay_norm = np.clip(data.ay_mps2 / 9.81, -1.0, 1.0)
    front_split = np.clip(0.18 * ay_norm, -0.20, 0.20)
    rear_split = np.clip(0.12 * ay_norm, -0.15, 0.15)

    fy_fl = 0.5 * data.fyf_true_n * (1.0 - front_split)
    fy_fr = 0.5 * data.fyf_true_n * (1.0 + front_split)
    fy_rl = 0.5 * data.fyr_true_n * (1.0 - rear_split)
    fy_rr = 0.5 * data.fyr_true_n * (1.0 + rear_split)
    return np.column_stack([fy_fl, fy_fr, fy_rl, fy_rr]).astype(np.float32)



class TinyForceTransformer(nn.Module):
    def __init__(self, config: ForceModelConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(len(SENSOR_FEATURE_NAMES), config.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.window_size, config.d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input_proj(x) + self.pos_embed[:, : x.shape[1], :]
        encoded = self.encoder(hidden)
        pooled = encoded[:, -1, :]
        return self.head(pooled)


@dataclass
class ForceModelBundle:
    config: ForceModelConfig
    input_scaler: SequenceStandardScaler
    target_scaler: VectorStandardScaler
    model_state: dict

    def save(self, model_path: str) -> None:
        torch.save(
            {
                "config": self.config.__dict__,
                "input_mean": self.input_scaler.mean,
                "input_scale": self.input_scaler.scale,
                "target_mean": self.target_scaler.mean,
                "target_scale": self.target_scaler.scale,
                "model_state": self.model_state,
            },
            model_path,
        )

    @classmethod
    def load(cls, model_path: str, device: str = "cpu") -> "ForceModelBundle":
        payload = torch.load(model_path, map_location=device, weights_only=False)
        config = ForceModelConfig(**payload["config"])
        return cls(
            config=config,
            input_scaler=SequenceStandardScaler(payload["input_mean"], payload["input_scale"]),
            target_scaler=VectorStandardScaler(payload["target_mean"], payload["target_scale"]),
            model_state=payload["model_state"],
        )


class ForcePredictor:
    def __init__(self, bundle: ForceModelBundle, device: str = "cpu"):
        self.bundle = bundle
        self.device = torch.device(device)
        self.model = TinyForceTransformer(bundle.config).to(self.device)
        self.model.load_state_dict(bundle.model_state)
        self.model.eval()

    def predict_forces(self, sensor_windows: np.ndarray) -> np.ndarray:
        X_scaled = self.bundle.input_scaler.transform(sensor_windows.astype(np.float32))
        with torch.no_grad():
            inputs = torch.from_numpy(X_scaled).to(self.device)
            prediction_scaled = self.model(inputs).cpu().numpy()
        forces = self.bundle.target_scaler.inverse_transform(prediction_scaled).astype(np.float32)
        return np.clip(forces, -self.bundle.config.force_clip_n, self.bundle.config.force_clip_n)


def fit_force_transformer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: ForceModelConfig,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[ForceModelBundle, dict[str, list[float]]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_scaler = SequenceStandardScaler.fit(X_train)
    target_scaler = VectorStandardScaler.fit(y_train)
    X_train_scaled = input_scaler.transform(X_train).astype(np.float32)
    X_val_scaled = input_scaler.transform(X_val).astype(np.float32)
    y_train_scaled = target_scaler.transform(y_train).astype(np.float32)
    y_val_scaled = target_scaler.transform(y_val).astype(np.float32)

    model = TinyForceTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_scaled), torch.from_numpy(y_train_scaled)),
        batch_size=batch_size,
        shuffle=True,
    )
    X_val_tensor = torch.from_numpy(X_val_scaled).to(device)
    y_val_tensor = torch.from_numpy(y_val_scaled).to(device)

    best_val = float("inf")
    best_state = None
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        batch_losses: list[float] = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_tensor)
            val_loss = float(criterion(val_pred, y_val_tensor).item())

        train_loss = float(np.mean(batch_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch + 1:4d}/{epochs}: train_loss = {train_loss:.6f}, val_loss = {val_loss:.6f}")

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    bundle = ForceModelBundle(config=config, input_scaler=input_scaler, target_scaler=target_scaler, model_state=best_state)
    return bundle, history


def predict_forces_for_prepared_data(predictor: ForcePredictor, data: VehicleSimPreparedData) -> np.ndarray:
    sensors = scenario_sensor_matrix(data)
    windows = np.stack([build_sensor_window_array(sensors, i, predictor.bundle.config.window_size) for i in range(len(data.time_s))], axis=0)
    return predictor.predict_forces(windows)


def force_rmse(force_true: np.ndarray, force_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((force_true - force_pred) ** 2)))


def wheel_to_axle_force_sums(wheel_forces: np.ndarray) -> np.ndarray:
    front = wheel_forces[:, 0] + wheel_forces[:, 1]
    rear = wheel_forces[:, 2] + wheel_forces[:, 3]
    return np.column_stack([front, rear]).astype(np.float32)


def axle_force_rmse_from_wheels(data: VehicleSimPreparedData, wheel_force_pred: np.ndarray) -> float:
    truth = np.column_stack([data.fyf_true_n, data.fyr_true_n]).astype(np.float32)
    pred = wheel_to_axle_force_sums(wheel_force_pred)
    return force_rmse(truth, pred)


def clip_vehicle_state(state: np.ndarray) -> np.ndarray:
    clipped = state.copy()
    clipped[0] = float(np.clip(clipped[0], -0.35, 0.35))
    clipped[1] = float(np.clip(clipped[1], -1.5, 1.5))
    return clipped


def force_driven_dynamics(state: np.ndarray, vx_mps: float, fyf_n: float, fyr_n: float, params: VehicleParams) -> np.ndarray:
    vx_safe = max(abs(float(vx_mps)), 0.5)
    beta_dot = (float(fyf_n) + float(fyr_n)) / (params.m_kg * vx_safe) - float(state[1])
    yaw_rate_dot = (params.lf_m * float(fyf_n) - params.lr_m * float(fyr_n)) / params.iz_kgm2
    return np.array([beta_dot, yaw_rate_dot], dtype=float)


def force_driven_fx(state: np.ndarray, dt: float, vx_mps: float, fyf_n: float, fyr_n: float, params: VehicleParams) -> np.ndarray:
    return state + dt * force_driven_dynamics(state, vx_mps, fyf_n, fyr_n, params)


def numerical_state_jacobian(func, state: np.ndarray, epsilon: float = 1.0e-5) -> np.ndarray:
    base = func(state)
    jacobian = np.zeros((len(base), len(state)), dtype=float)
    for idx in range(len(state)):
        perturbed = state.copy()
        perturbed[idx] += epsilon
        jacobian[:, idx] = (func(perturbed) - base) / epsilon
    return jacobian


def save_force_hybrid_plot(
    data: VehicleSimPreparedData,
    beta_est_rad: np.ndarray,
    yaw_rate_est_rps: np.ndarray,
    force_pred_n: np.ndarray,
    plot_file: str,
    title: str,
) -> None:
    path_x_est, path_y_est, _ = reconstruct_path(
        data.time_s, data.vx_mps, beta_est_rad, yaw_rate_est_rps, data.yaw_true_rad[0]
    )

    fig, axes = plt.subplots(4, 1, figsize=(10, 15))

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

    pred_axle = wheel_to_axle_force_sums(force_pred_n)
    axes[2].plot(data.time_s, data.fyf_true_n, "k", linewidth=2, label="Front axle true")
    axes[2].plot(data.time_s, pred_axle[:, 0], label="Front axle NN sum")
    axes[2].plot(data.time_s, data.fyr_true_n, "k--", linewidth=2, label="Rear axle true")
    axes[2].plot(data.time_s, pred_axle[:, 1], label="Rear axle NN sum")
    axes[2].set_title(f"{title}: axle-force comparison")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Force (N)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(ncol=2)

    axes[3].plot(data.global_y_m, data.global_x_m, "k", linewidth=2, label="True path")
    axes[3].plot(path_y_est, path_x_est, label="Estimated path")
    axes[3].set_title(f"{title}: reconstructed path")
    axes[3].set_xlabel("Global Y (m)")
    axes[3].set_ylabel("Global X (m)")
    axes[3].axis("equal")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    fig.tight_layout()
    fig.savefig(plot_file, dpi=180)
    plt.close(fig)


def load_csv_datasets(csv_files: list[Path]) -> list:
    return [load_vehicle_sim_csv(path) for path in csv_files]
'''

# force_transformer_common_new.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import matplotlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from vehicle_filter_common import VehicleSimPreparedData, reconstruct_path
from vehicle_sim_loader import load_vehicle_sim_csv

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Real-world IMU/Odometer sensor streams accessible in a production vehicle environment
SENSOR_FEATURE_NAMES = ("vx_mps", "delta_rad", "ay_mps2")

@dataclass
class ForceModelConfig:
    window_size: int = 8
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    force_clip_n: float = 10000.0

@dataclass
class VehicleParams:
    m_kg: float
    iz_kgm2: float
    lf_m: float
    lr_m: float

    @classmethod
    def from_prepared_data(cls, data: VehicleSimPreparedData) -> "VehicleParams":
        return cls(
            m_kg=float(data.m_kg),
            iz_kgm2=float(data.iz_kgm2),
            lf_m=float(data.lf_m),
            lr_m=float(data.lr_m),
        )

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

def collect_scenario_files(folder: Path) -> list[Path]:
    def scenario_key(path: Path) -> tuple[int, str]:
        prefix = path.stem.split("_", 1)[0]
        # Robustly handles files prefixed with indices like 'data_to_train_0' or '1_runtime_dataset'
        digits = "".join(filter(str.isdigit, prefix))
        return (int(digits), path.name) if digits else (10_000, path.name)
    return sorted((path for path in folder.glob("*.csv") if path.name != "scenario_summary.csv"), key=scenario_key)

def select_files_by_name(all_files: list[Path], requested_names: list[str] | None) -> list[Path]:
    if not requested_names:
        return all_files
    lookup = {path.name: path for path in all_files}
    selected: list[Path] = []
    missing: list[str] = []
    for name in requested_names:
        if name in lookup:
            selected.append(lookup[name])
        else:
            missing.append(name)
    if missing:
        raise ValueError(f"Requested files not found: {missing}")
    return selected

def scenario_sensor_matrix(data: VehicleSimPreparedData) -> np.ndarray:
    # Explicit mapping to strictly measurable kinematics
    return np.column_stack([
        data.vx_mps,
        data.delta_rad,
        data.ay_mps2
    ]).astype(np.float32)

def build_sensor_window_array(sensor_matrix: np.ndarray, index: int, window_size: int) -> np.ndarray:
    start = max(0, index - window_size + 1)
    window = sensor_matrix[start: index + 1]
    if len(window) < window_size:
        pad = np.repeat(window[[0]], window_size - len(window), axis=0)
        window = np.vstack([pad, window])
    return window.astype(np.float32)

def build_force_training_arrays(data: VehicleSimPreparedData, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    sensors = scenario_sensor_matrix(data)
    X = np.zeros((len(data.time_s), window_size, sensors.shape[1]), dtype=np.float32)
    y = build_wheel_force_targets(data).astype(np.float32)
    for i in range(len(data.time_s)):
        X[i] = build_sensor_window_array(sensors, i, window_size)
    return X, y

def build_wheel_force_targets(data: VehicleSimPreparedData) -> np.ndarray:
    """
    Translates standard bicycle model axle forces into four distinct wheel-level 
    frictional forces utilizing an analytical transient lateral load-transfer estimation.
    """
    ay_norm = np.clip(data.ay_mps2 / 9.81, -1.0, 1.0)
    
    # Transient load transfer heuristic mapping based on dynamic parameters
    front_split = np.clip(0.18 * ay_norm, -0.20, 0.20)
    rear_split = np.clip(0.12 * ay_norm, -0.15, 0.15)
    
    fy_fl = 0.5 * data.fyf_true_n * (1.0 - front_split)
    fy_fr = 0.5 * data.fyf_true_n * (1.0 + front_split)
    fy_rl = 0.5 * data.fyr_true_n * (1.0 - rear_split)
    fy_rr = 0.5 * data.fyr_true_n * (1.0 + rear_split)
    
    return np.column_stack([fy_fl, fy_fr, fy_rl, fy_rr]).astype(np.float32)

class TinyForceTransformer(nn.Module):
    def __init__(self, config: ForceModelConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(len(SENSOR_FEATURE_NAMES), config.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.window_size, config.d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 4) # Output: [F_y_fl, F_y_fr, F_y_rl, F_y_rr]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input_proj(x) + self.pos_embed[:, :x.shape[1], :]
        encoded = self.encoder(hidden)
        pooled = encoded[:, -1, :]  # Causal sequence pooling (extracting the last step)
        return self.head(pooled)

@dataclass
class ForceModelBundle:
    config: ForceModelConfig
    input_scaler: SequenceStandardScaler
    target_scaler: VectorStandardScaler
    model_state: dict

    def save(self, model_path: str) -> None:
        torch.save({
            "config": self.config.__dict__,
            "input_mean": self.input_scaler.mean,
            "input_scale": self.input_scaler.scale,
            "target_mean": self.target_scaler.mean,
            "target_scale": self.target_scaler.scale,
            "model_state": self.model_state,
        }, model_path)

    @classmethod
    def load(cls, model_path: str, device: str = "cpu") -> "ForceModelBundle":
        payload = torch.load(model_path, map_location=device, weights_only=False)
        config = ForceModelConfig(**payload["config"])
        return cls(
            config=config,
            input_scaler=SequenceStandardScaler(payload["input_mean"], payload["input_scale"]),
            target_scaler=VectorStandardScaler(payload["target_mean"], payload["target_scale"]),
            model_state=payload["model_state"],
        )

class ForcePredictor:
    def __init__(self, bundle: ForceModelBundle, device: str = "cpu"):
        self.bundle = bundle
        self.device = torch.device(device)
        self.model = TinyForceTransformer(bundle.config).to(self.device)
        self.model.load_state_dict(bundle.model_state)
        self.model.eval()

    def predict_forces(self, sensor_windows: np.ndarray) -> np.ndarray:
        X_scaled = self.bundle.input_scaler.transform(sensor_windows.astype(np.float32))
        with torch.no_grad():
            inputs = torch.from_numpy(X_scaled).to(self.device)
            prediction_scaled = self.model(inputs).cpu().numpy()
        forces = self.bundle.target_scaler.inverse_transform(prediction_scaled).astype(np.float32)
        return np.clip(forces, -self.bundle.config.force_clip_n, self.bundle.config.force_clip_n)

def fit_force_transformer(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    config: ForceModelConfig, epochs: int, batch_size: int,
    learning_rate: float, weight_decay: float, seed: int = 0, device: str = "cpu"
) -> tuple[ForceModelBundle, dict[str, list[float]]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    input_scaler = SequenceStandardScaler.fit(X_train)
    target_scaler = VectorStandardScaler.fit(y_train)
    
    X_train_scaled = input_scaler.transform(X_train).astype(np.float32)
    X_val_scaled = input_scaler.transform(X_val).astype(np.float32)
    y_train_scaled = target_scaler.transform(y_train).astype(np.float32)
    y_val_scaled = target_scaler.transform(y_val).astype(np.float32)
    
    model = TinyForceTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_scaled), torch.from_numpy(y_train_scaled)),
        batch_size=batch_size, shuffle=True
    )
    
    X_val_tensor = torch.from_numpy(X_val_scaled).to(device)
    y_val_tensor = torch.from_numpy(y_val_scaled).to(device)
    
    best_val = float("inf")
    best_state = None
    history = {"train_loss": [], "val_loss": []}
    
    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(float(loss.item()))
            
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_tensor)
            val_loss = float(criterion(val_pred, y_val_tensor).item())
            
        train_loss = float(np.mean(batch_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch + 1:4d}/{epochs}: train_loss = {train_loss:.6f}, val_loss = {val_loss:.6f}")
            
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        
    bundle = ForceModelBundle(config=config, input_scaler=input_scaler, target_scaler=target_scaler, model_state=best_state)
    return bundle, history

def predict_forces_for_prepared_data(predictor: ForcePredictor, data: VehicleSimPreparedData) -> np.ndarray:
    sensors = scenario_sensor_matrix(data)
    windows = np.stack([build_sensor_window_array(sensors, i, predictor.bundle.config.window_size) for i in range(len(data.time_s))], axis=0)
    return predictor.predict_forces(windows)

def force_rmse(force_true: np.ndarray, force_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((force_true - force_pred) ** 2)))

def wheel_to_axle_force_sums(wheel_forces: np.ndarray) -> np.ndarray:
    front = wheel_forces[:, 0] + wheel_forces[:, 1]
    rear = wheel_forces[:, 2] + wheel_forces[:, 3]
    return np.column_stack([front, rear]).astype(np.float32)

def axle_force_rmse_from_wheels(data: VehicleSimPreparedData, wheel_force_pred: np.ndarray) -> float:
    truth = np.column_stack([data.fyf_true_n, data.fyr_true_n]).astype(np.float32)
    pred = wheel_to_axle_force_sums(wheel_force_pred)
    return force_rmse(truth, pred)

def clip_vehicle_state(state: np.ndarray) -> np.ndarray:
    clipped = state.copy()
    clipped[0] = float(np.clip(clipped[0], -0.35, 0.35))
    clipped[1] = float(np.clip(clipped[1], -1.5, 1.5))
    return clipped

def force_driven_dynamics(state: np.ndarray, vx_mps: float, fyf_n: float, fyr_n: float, params: VehicleParams) -> np.ndarray:
    vx_safe = max(abs(float(vx_mps)), 0.5)
    beta_dot = (float(fyf_n) + float(fyr_n)) / (params.m_kg * vx_safe) - float(state[1])
    yaw_rate_dot = (params.lf_m * float(fyf_n) - params.lr_m * float(fyr_n)) / params.iz_kgm2
    return np.array([beta_dot, yaw_rate_dot], dtype=float)

def force_driven_fx(state: np.ndarray, dt: float, vx_mps: float, fyf_n: float, fyr_n: float, params: VehicleParams) -> np.ndarray:
    return state + dt * force_driven_dynamics(state, vx_mps, fyf_n, fyr_n, params)

def numerical_state_jacobian(func, state: np.ndarray, epsilon: float = 1.0e-5) -> np.ndarray:
    base = func(state)
    jacobian = np.zeros((len(base), len(state)), dtype=float)
    for idx in range(len(state)):
        perturbed = state.copy()
        perturbed[idx] += epsilon
        jacobian[:, idx] = (func(perturbed) - base) / epsilon
    return jacobian

def save_force_hybrid_plot(data: VehicleSimPreparedData, beta_est_rad: np.ndarray, yaw_rate_est_rps: np.ndarray, force_pred_n: np.ndarray, plot_file: str, title: str) -> None:
    path_x_est, path_y_est, _ = reconstruct_path(data.time_s, data.vx_mps, beta_est_rad, yaw_rate_est_rps, data.yaw_true_rad[0])
    fig, axes = plt.subplots(4, 1, figsize=(10, 15))
    
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

    pred_axle = wheel_to_axle_force_sums(force_pred_n)
    axes[2].plot(data.time_s, data.fyf_true_n, "k", linewidth=2, label="Front axle true")
    axes[2].plot(data.time_s, pred_axle[:, 0], label="Front axle NN sum")
    axes[2].plot(data.time_s, data.fyr_true_n, "k", linewidth=2, label="Rear axle true")
    axes[2].plot(data.time_s, pred_axle[:, 1], label="Rear axle NN sum")
    axes[2].set_title(f"{title}: axle-force comparison")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Force (N)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(ncol=2)

    axes[3].plot(data.global_y_m, data.global_x_m, "k", linewidth=2, label="True path")
    axes[3].plot(path_y_est, path_x_est, label="Estimated path")
    axes[3].set_title(f"{title}: reconstructed path")
    axes[3].set_xlabel("Global Y (m)")
    axes[3].set_ylabel("Global X (m)")
    axes[3].axis("equal")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    fig.tight_layout()
    fig.savefig(plot_file, dpi=180)
    plt.close(fig)

def load_csv_datasets(csv_files: list[Path]) -> list:
    return [load_vehicle_sim_csv(path) for path in csv_files]