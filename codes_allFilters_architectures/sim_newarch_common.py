from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from vehicle_sim_loader import VehicleSimDataset
from vehicle_sim_loader import load_vehicle_sim_csv

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class SimNNSettings:
    batch_size: int = 512
    epochs: int = 20
    learning_rate: float = 4.0e-4
    val_split: float = 0.25
    input_timesteps: int = 5
    input_dim: int = 5
    output_dim: int = 3
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1


@dataclass
class SimVehicleParams:
    sample_time_s: float
    mass_kg: float
    iz_kgm2: float
    lf_m: float
    lr_m: float
    force_clip_n: float = 10000.0

    @classmethod
    def from_dataset(cls, data: VehicleSimDataset) -> "SimVehicleParams":
        time_step = float(np.median(np.diff(data.time_s))) if len(data.time_s) > 1 else 0.01
        return cls(
            sample_time_s=time_step,
            mass_kg=float(data.m_kg),
            iz_kgm2=float(data.iz_kgm2),
            lf_m=float(data.lf_m),
            lr_m=float(data.lr_m),
        )


def rmse(truth: np.ndarray, estimate: np.ndarray) -> float:
    truth_arr = np.asarray(truth, dtype=float)
    estimate_arr = np.asarray(estimate, dtype=float)
    return float(np.sqrt(np.mean((truth_arr - estimate_arr) ** 2)))


def load_sim_dataset(csv_path: str | Path) -> VehicleSimDataset:
    return load_vehicle_sim_csv(csv_path)


def scenario_sensor_matrix(data: VehicleSimDataset) -> np.ndarray:
    return np.column_stack(
        [
            data.vx_mps,
            data.delta_rad,
            data.ay_mps2,
            data.yaw_rate_rps,
            data.yaw_acc_rps2,
        ]
    ).astype(np.float32)


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


def build_targets(data: VehicleSimDataset) -> np.ndarray:
    return np.column_stack(
        [
            data.fyf_true_n.astype(np.float32),
            data.fyr_true_n.astype(np.float32),
            data.beta_true_rad.astype(np.float32),
        ]
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


class TinySimTransformer(nn.Module):
    def __init__(self, config: SimNNSettings):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.input_dim, config.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.input_timesteps, config.d_model))
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
            nn.Linear(config.d_model, config.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input_proj(x) + self.pos_embed[:, : x.shape[1], :]
        encoded = self.encoder(hidden)
        return self.head(encoded[:, -1, :])


@dataclass
class SimForceModelBundle:
    config: SimNNSettings
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
    def load(cls, model_path: str, device: str = "cpu") -> "SimForceModelBundle":
        payload = torch.load(model_path, map_location=device, weights_only=False)
        config = SimNNSettings(**payload["config"])
        return cls(
            config=config,
            input_scaler=SequenceStandardScaler(payload["input_mean"], payload["input_scale"]),
            target_scaler=VectorStandardScaler(payload["target_mean"], payload["target_scale"]),
            model_state=payload["model_state"],
        )


class SimForcePredictor:
    def __init__(self, bundle: SimForceModelBundle, device: str = "cpu"):
        self.bundle = bundle
        self.device = torch.device(device)
        self.model = TinySimTransformer(bundle.config).to(self.device)
        self.model.load_state_dict(bundle.model_state)
        self.model.eval()

    def predict(self, sensor_windows: np.ndarray) -> np.ndarray:
        X_scaled = self.bundle.input_scaler.transform(sensor_windows.astype(np.float32))
        with torch.no_grad():
            output_scaled = self.model(torch.from_numpy(X_scaled).to(self.device)).cpu().numpy()
        return self.bundle.target_scaler.inverse_transform(output_scaled).astype(np.float32)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: SimNNSettings,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[SimForceModelBundle, dict[str, list[float]]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_scaler = SequenceStandardScaler.fit(X_train)
    target_scaler = VectorStandardScaler.fit(y_train)
    X_train_scaled = input_scaler.transform(X_train).astype(np.float32)
    X_val_scaled = input_scaler.transform(X_val).astype(np.float32)
    y_train_scaled = target_scaler.transform(y_train).astype(np.float32)
    y_val_scaled = target_scaler.transform(y_val).astype(np.float32)

    model = TinySimTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate or config.learning_rate, weight_decay=1.0e-4)
    criterion = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_scaled), torch.from_numpy(y_train_scaled)),
        batch_size=batch_size or config.batch_size,
        shuffle=True,
    )
    X_val_t = torch.from_numpy(X_val_scaled).to(device)
    y_val_t = torch.from_numpy(y_val_scaled).to(device)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None

    for epoch in range(epochs or config.epochs):
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
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == (epochs or config.epochs) - 1:
            print(f"Epoch {epoch + 1:4d}/{epochs or config.epochs}: train_loss = {train_loss:.6f}, val_loss = {val_loss:.6f}")

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    bundle = SimForceModelBundle(
        config=config,
        input_scaler=input_scaler,
        target_scaler=target_scaler,
        model_state=best_state,
    )
    return bundle, history


def run_sim_ekf(
    data: VehicleSimDataset,
    predicted_outputs: np.ndarray,
    params: SimVehicleParams,
) -> dict[str, np.ndarray | float]:
    vx_series = data.vx_mps.astype(float)
    beta0 = float(data.beta_true_rad[0])
    x = np.array([beta0, float(data.yaw_rate_rps[0]), 0.0, 0.0], dtype=float)
    P = np.diag([0.02**2, 0.05**2, 350.0**2, 0.15**2])
    Q = np.diag([0.004**2, 0.03**2, 60.0**2, 0.02**2])
    R = np.diag([0.02**2, 0.10**2, 0.03**2])

    estimates = [x.copy()]

    for i in range(1, len(data.time_s)):
        dt = float(data.time_s[i] - data.time_s[i - 1])
        vx_mps = float(data.vx_mps[i])
        front_force = float(predicted_outputs[i, 0])
        rear_force = float(predicted_outputs[i, 1])
        beta_meas = float(predicted_outputs[i, 2])

        def fx_local(state: np.ndarray) -> np.ndarray:
            beta_rad, yaw_rate, force_bias, _ay_bias = state
            vx_safe = max(abs(vx_mps), 0.5)
            total_force = front_force + rear_force + float(force_bias)
            beta_dot = total_force / (params.mass_kg * vx_safe) - float(yaw_rate)
            yaw_dot = (params.lf_m * front_force - params.lr_m * rear_force) / params.iz_kgm2
            return np.array([beta_rad + dt * beta_dot, yaw_rate + dt * yaw_dot, force_bias, _ay_bias], dtype=float)

        def hx_local(state: np.ndarray) -> np.ndarray:
            beta_rad, yaw_rate, force_bias, ay_bias = state
            total_force = front_force + rear_force + float(force_bias)
            ay_pred = total_force / params.mass_kg + float(ay_bias)
            return np.array([float(yaw_rate), ay_pred, float(beta_rad)], dtype=float)

        x_pred = fx_local(x)
        F = numerical_state_jacobian(fx_local, x)
        P_pred = F @ P @ F.T + Q

        z = np.array([float(data.yaw_rate_rps[i]), float(data.ay_mps2[i]), beta_meas], dtype=float)
        innovation = z - hx_local(x_pred)
        H = numerical_state_jacobian(hx_local, x_pred)
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        x = x_pred + K @ innovation
        x[0] = float(np.clip(x[0], -0.6, 0.6))
        x[1] = float(np.clip(x[1], -2.0, 2.0))
        x[2] = float(np.clip(x[2], -5000.0, 5000.0))
        x[3] = float(np.clip(x[3], -5.0, 5.0))
        P = (np.eye(4) - K @ H) @ P_pred
        P = 0.5 * (P + P.T)

        estimates.append(x.copy())

    est = np.asarray(estimates)
    front_truth = data.fyf_true_n.astype(float)
    rear_truth = data.fyr_true_n.astype(float)
    force_pred = predicted_outputs[:, :2].astype(float)

    return {
        "estimates": est,
        "beta_rmse": rmse(data.beta_true_rad, est[:, 0]),
        "yaw_rmse": rmse(data.yaw_rate_rps, est[:, 1]),
        "force_rmse": float(np.sqrt(np.mean((np.column_stack([front_truth, rear_truth]) - force_pred) ** 2))),
    }


def numerical_state_jacobian(func, state: np.ndarray, epsilon: float = 1.0e-5) -> np.ndarray:
    base = func(state)
    jacobian = np.zeros((len(base), len(state)), dtype=float)
    for idx in range(len(state)):
        perturbed = state.copy()
        perturbed[idx] += epsilon
        jacobian[:, idx] = (func(perturbed) - base) / epsilon
    return jacobian


def save_summary_plot(
    data: VehicleSimDataset,
    result: dict[str, np.ndarray | float],
    predicted_outputs: np.ndarray,
    plot_file: str,
    title: str,
) -> None:
    estimates = np.asarray(result["estimates"])
    fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True)

    axes[0].plot(data.time_s, data.beta_true_rad, "k", linewidth=2, label="beta true")
    axes[0].plot(data.time_s, estimates[:, 0], label="beta est")
    axes[0].set_ylabel("beta (rad)")
    axes[0].set_title(f"{title}: beta")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(data.time_s, data.yaw_rate_rps, "k", linewidth=2, label="yaw rate true")
    axes[1].plot(data.time_s, estimates[:, 1], label="yaw rate est")
    axes[1].set_ylabel("yaw rate (rad/s)")
    axes[1].set_title(f"{title}: yaw rate")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(data.time_s, data.fyf_true_n, "k", linewidth=2, label="front force true")
    axes[2].plot(data.time_s, predicted_outputs[:, 0], label="front force pred")
    axes[2].plot(data.time_s, data.fyr_true_n, "k--", linewidth=2, label="rear force true")
    axes[2].plot(data.time_s, predicted_outputs[:, 1], label="rear force pred")
    axes[2].set_ylabel("force (N)")
    axes[2].set_title(f"{title}: axle force")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(ncol=2)

    axes[3].plot(data.time_s, np.abs(estimates[:, 0] - data.beta_true_rad), label="|beta error|")
    axes[3].plot(data.time_s, np.abs(estimates[:, 1] - data.yaw_rate_rps), label="|yaw rate error|")
    axes[3].set_xlabel("time (s)")
    axes[3].set_ylabel("error")
    axes[3].set_title(f"{title}: error graph")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    fig.tight_layout()
    fig.savefig(plot_file, dpi=180)
    plt.close(fig)
