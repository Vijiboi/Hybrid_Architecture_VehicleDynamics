from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vehicle_filter_common import VehicleSimPreparedData
from vehicle_filter_common import safe_speed


MEASURED_WINDOW_FEATURES = ("vx_mps", "delta_rad", "ay_mps2", "yaw_rate_rps", "yaw_acc_rps2")


@dataclass
class HybridVehicleParams:
    cf_nprad: float
    cr_nprad: float
    m_kg: float
    iz_kgm2: float
    lf_m: float
    lr_m: float

    @classmethod
    def from_prepared_data(cls, data: VehicleSimPreparedData) -> "HybridVehicleParams":
        return cls(
            cf_nprad=float(data.cf_nprad),
            cr_nprad=float(data.cr_nprad),
            m_kg=float(data.m_kg),
            iz_kgm2=float(data.iz_kgm2),
            lf_m=float(data.lf_m),
            lr_m=float(data.lr_m),
        )


@dataclass
class ForceCorrectionConfig:
    residual_scale: float = 0.15
    residual_clip_n: float = 800.0
    total_force_clip_n: float = 5000.0
    slip_angle_clip_rad: float = 0.25
    full_correction_speed_mps: float = 12.0


def compute_slip_angles(
    beta_rad: float,
    yaw_rate_rps: float,
    delta_rad: float,
    vx_mps: float,
    lf_m: float,
    lr_m: float,
) -> tuple[float, float]:
    vx_safe = safe_speed(vx_mps)
    alpha_f = float(delta_rad - beta_rad - (lf_m * yaw_rate_rps) / vx_safe)
    alpha_r = float(-beta_rad + (lr_m * yaw_rate_rps) / vx_safe)
    return alpha_f, alpha_r


def compute_linear_tire_forces(
    alpha_f_rad: float,
    alpha_r_rad: float,
    cf_nprad: float,
    cr_nprad: float,
) -> tuple[float, float]:
    return float(cf_nprad * alpha_f_rad), float(cr_nprad * alpha_r_rad)


def build_measurement_window(data: VehicleSimPreparedData, index: int, window_size: int) -> np.ndarray:
    rows: list[list[float]] = []
    start = max(0, index - window_size + 1)
    for j in range(start, index + 1):
        rows.append(
            [
                float(data.vx_mps[j]),
                float(data.delta_rad[j]),
                float(data.ay_mps2[j]),
                float(data.yaw_rate_rps[j]),
                float(data.yaw_acc_rps2[j]),
            ]
        )
    while len(rows) < window_size:
        rows.insert(0, rows[0])
    return np.asarray(rows, dtype=float).reshape(-1)


def build_hybrid_feature_vector(
    measurement_window_flat: np.ndarray,
    alpha_f_rad: float,
    alpha_r_rad: float,
) -> np.ndarray:
    dynamic_terms = np.array([alpha_f_rad, alpha_r_rad, alpha_f_rad - alpha_r_rad, alpha_f_rad + alpha_r_rad], dtype=float)
    return np.concatenate([measurement_window_flat.astype(float), dynamic_terms])


def build_friction_training_matrices(
    data: VehicleSimPreparedData,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    params = HybridVehicleParams.from_prepared_data(data)
    X_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []

    for i in range(len(data.time_s)):
        alpha_f, alpha_r = compute_slip_angles(
            beta_rad=float(data.beta_true_rad[i]),
            yaw_rate_rps=float(data.yaw_rate_rps[i]),
            delta_rad=float(data.delta_rad[i]),
            vx_mps=float(data.vx_mps[i]),
            lf_m=params.lf_m,
            lr_m=params.lr_m,
        )
        linear_fyf, linear_fyr = compute_linear_tire_forces(alpha_f, alpha_r, params.cf_nprad, params.cr_nprad)
        window_flat = build_measurement_window(data, i, window_size)
        X_rows.append(
            build_hybrid_feature_vector(
                window_flat,
                alpha_f_rad=alpha_f,
                alpha_r_rad=alpha_r,
            )
        )
        y_rows.append(
            np.array(
                [
                    float(data.fyf_true_n[i]) - linear_fyf,
                    float(data.fyr_true_n[i]) - linear_fyr,
                ],
                dtype=float,
            )
        )

    return np.vstack(X_rows), np.vstack(y_rows)


class StandardScaler:
    def __init__(self, mean: np.ndarray, scale: np.ndarray):
        self.mean = mean.astype(float)
        self.scale = np.where(scale > 1.0e-8, scale, 1.0).astype(float)

    @classmethod
    def fit(cls, values: np.ndarray) -> "StandardScaler":
        return cls(np.mean(values, axis=0), np.std(values, axis=0))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * self.scale + self.mean


class SimpleMLPRegressor:
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (48, 48),
        output_dim: int = 2,
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        layer_dims = (input_dim,) + hidden_dims + (output_dim,)
        self.weights = [
            rng.normal(0.0, np.sqrt(2.0 / layer_dims[i]), size=(layer_dims[i], layer_dims[i + 1])).astype(float)
            for i in range(len(layer_dims) - 1)
        ]
        self.biases = [np.zeros(layer_dims[i + 1], dtype=float) for i in range(len(layer_dims) - 1)]

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        activations = [X]
        pre_activations: list[np.ndarray] = []
        current = X
        for idx, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = current @ W + b
            pre_activations.append(z)
            if idx < len(self.weights) - 1:
                current = np.tanh(z)
            else:
                current = z
            activations.append(current)
        return current, activations, pre_activations

    def predict(self, X: np.ndarray) -> np.ndarray:
        current = X
        for idx, (W, b) in enumerate(zip(self.weights, self.biases)):
            current = current @ W + b
            if idx < len(self.weights) - 1:
                current = np.tanh(current)
        return current

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 800,
        learning_rate: float = 1.0e-3,
        batch_size: int = 256,
        l2: float = 1.0e-5,
        verbose_every: int = 100,
    ) -> dict[str, list[float]]:
        rng = np.random.default_rng(0)
        adam_m_w = [np.zeros_like(W) for W in self.weights]
        adam_v_w = [np.zeros_like(W) for W in self.weights]
        adam_m_b = [np.zeros_like(b) for b in self.biases]
        adam_v_b = [np.zeros_like(b) for b in self.biases]
        beta1 = 0.9
        beta2 = 0.999
        eps = 1.0e-8
        step = 0

        history = {"loss": []}

        for epoch in range(epochs):
            permutation = rng.permutation(len(X))
            X_epoch = X[permutation]
            y_epoch = y[permutation]

            for start in range(0, len(X_epoch), batch_size):
                step += 1
                end = min(start + batch_size, len(X_epoch))
                xb = X_epoch[start:end]
                yb = y_epoch[start:end]

                pred, activations, pre_activations = self.forward(xb)
                error = pred - yb
                batch_loss = float(np.mean(error**2))

                grad = (2.0 / len(xb)) * error
                grad_w: list[np.ndarray] = []
                grad_b: list[np.ndarray] = []

                for layer in range(len(self.weights) - 1, -1, -1):
                    grad_w_layer = activations[layer].T @ grad + l2 * self.weights[layer]
                    grad_b_layer = np.sum(grad, axis=0)
                    grad_w.insert(0, grad_w_layer)
                    grad_b.insert(0, grad_b_layer)

                    if layer > 0:
                        grad = (grad @ self.weights[layer].T) * (1.0 - np.tanh(pre_activations[layer - 1]) ** 2)

                for layer in range(len(self.weights)):
                    adam_m_w[layer] = beta1 * adam_m_w[layer] + (1.0 - beta1) * grad_w[layer]
                    adam_v_w[layer] = beta2 * adam_v_w[layer] + (1.0 - beta2) * (grad_w[layer] ** 2)
                    adam_m_b[layer] = beta1 * adam_m_b[layer] + (1.0 - beta1) * grad_b[layer]
                    adam_v_b[layer] = beta2 * adam_v_b[layer] + (1.0 - beta2) * (grad_b[layer] ** 2)

                    m_w_hat = adam_m_w[layer] / (1.0 - beta1**step)
                    v_w_hat = adam_v_w[layer] / (1.0 - beta2**step)
                    m_b_hat = adam_m_b[layer] / (1.0 - beta1**step)
                    v_b_hat = adam_v_b[layer] / (1.0 - beta2**step)

                    self.weights[layer] -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + eps)
                    self.biases[layer] -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + eps)

            full_loss = float(np.mean((self.predict(X) - y) ** 2))
            history["loss"].append(full_loss)
            if verbose_every and ((epoch + 1) % verbose_every == 0 or epoch == 0):
                print(f"Epoch {epoch + 1:4d}/{epochs}: loss = {full_loss:.6f}")

        return history


@dataclass
class FrictionResidualModel:
    input_scaler: StandardScaler
    target_scaler: StandardScaler
    network: SimpleMLPRegressor
    window_size: int
    config: ForceCorrectionConfig

    def predict_residual_forces(self, feature_matrix: np.ndarray) -> np.ndarray:
        X_scaled = self.input_scaler.transform(feature_matrix)
        residual_scaled = self.network.predict(X_scaled)
        return self.target_scaler.inverse_transform(residual_scaled)

    def save(self, model_path: str) -> None:
        payload: dict[str, np.ndarray] = {
            "input_mean": self.input_scaler.mean,
            "input_scale": self.input_scaler.scale,
            "target_mean": self.target_scaler.mean,
            "target_scale": self.target_scaler.scale,
            "window_size": np.array([self.window_size], dtype=int),
            "num_layers": np.array([len(self.network.weights)], dtype=int),
            "residual_scale": np.array([self.config.residual_scale], dtype=float),
            "residual_clip_n": np.array([self.config.residual_clip_n], dtype=float),
            "total_force_clip_n": np.array([self.config.total_force_clip_n], dtype=float),
            "slip_angle_clip_rad": np.array([self.config.slip_angle_clip_rad], dtype=float),
            "full_correction_speed_mps": np.array([self.config.full_correction_speed_mps], dtype=float),
        }
        for idx, (W, b) in enumerate(zip(self.network.weights, self.network.biases)):
            payload[f"W_{idx}"] = W
            payload[f"b_{idx}"] = b
        np.savez_compressed(model_path, **payload)

    @classmethod
    def load(cls, model_path: str) -> "FrictionResidualModel":
        data = np.load(model_path)
        num_layers = int(data["num_layers"][0])
        weights = [data[f"W_{idx}"] for idx in range(num_layers)]
        biases = [data[f"b_{idx}"] for idx in range(num_layers)]
        network = SimpleMLPRegressor(input_dim=weights[0].shape[0], hidden_dims=tuple(W.shape[1] for W in weights[:-1]), output_dim=weights[-1].shape[1])
        network.weights = [W.astype(float) for W in weights]
        network.biases = [b.astype(float) for b in biases]
        return cls(
            input_scaler=StandardScaler(data["input_mean"], data["input_scale"]),
            target_scaler=StandardScaler(data["target_mean"], data["target_scale"]),
            network=network,
            window_size=int(data["window_size"][0]),
            config=ForceCorrectionConfig(
                residual_scale=float(data["residual_scale"][0]) if "residual_scale" in data else 0.15,
                residual_clip_n=float(data["residual_clip_n"][0]) if "residual_clip_n" in data else 800.0,
                total_force_clip_n=float(data["total_force_clip_n"][0]) if "total_force_clip_n" in data else 5000.0,
                slip_angle_clip_rad=float(data["slip_angle_clip_rad"][0]) if "slip_angle_clip_rad" in data else 0.25,
                full_correction_speed_mps=float(data["full_correction_speed_mps"][0]) if "full_correction_speed_mps" in data else 12.0,
            ),
        )


def fit_friction_residual_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    window_size: int,
    config: ForceCorrectionConfig | None = None,
    hidden_dims: tuple[int, ...] = (48, 48),
    epochs: int = 800,
    learning_rate: float = 1.0e-3,
    batch_size: int = 256,
    l2: float = 1.0e-5,
    seed: int = 0,
) -> tuple[FrictionResidualModel, dict[str, list[float]]]:
    if config is None:
        config = ForceCorrectionConfig()
    input_scaler = StandardScaler.fit(X_train)
    target_scaler = StandardScaler.fit(y_train)
    X_scaled = input_scaler.transform(X_train)
    y_scaled = target_scaler.transform(y_train)

    network = SimpleMLPRegressor(input_dim=X_train.shape[1], hidden_dims=hidden_dims, output_dim=y_train.shape[1], seed=seed)
    history = network.fit(
        X_scaled,
        y_scaled,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        l2=l2,
    )
    return FrictionResidualModel(input_scaler, target_scaler, network, window_size, config), history


def predict_total_lateral_forces(
    model: FrictionResidualModel,
    measurement_window_flat: np.ndarray,
    state: np.ndarray,
    delta_rad: float,
    vx_mps: float,
    params: HybridVehicleParams,
) -> tuple[float, float, float, float]:
    alpha_f, alpha_r = compute_slip_angles(
        beta_rad=float(state[0]),
        yaw_rate_rps=float(state[1]),
        delta_rad=delta_rad,
        vx_mps=vx_mps,
        lf_m=params.lf_m,
        lr_m=params.lr_m,
    )
    alpha_f = float(np.clip(alpha_f, -model.config.slip_angle_clip_rad, model.config.slip_angle_clip_rad))
    alpha_r = float(np.clip(alpha_r, -model.config.slip_angle_clip_rad, model.config.slip_angle_clip_rad))
    linear_fyf, linear_fyr = compute_linear_tire_forces(alpha_f, alpha_r, params.cf_nprad, params.cr_nprad)
    feature_vector = build_hybrid_feature_vector(measurement_window_flat, alpha_f, alpha_r)
    residual_fy = model.predict_residual_forces(feature_vector[None, :])[0]
    residual_fy = np.clip(residual_fy, -model.config.residual_clip_n, model.config.residual_clip_n)
    speed_gain = float(np.clip(abs(vx_mps) / model.config.full_correction_speed_mps, 0.0, 1.0))
    blend = model.config.residual_scale * speed_gain
    fyf_total = float(np.clip(linear_fyf + blend * float(residual_fy[0]), -model.config.total_force_clip_n, model.config.total_force_clip_n))
    fyr_total = float(np.clip(linear_fyr + blend * float(residual_fy[1]), -model.config.total_force_clip_n, model.config.total_force_clip_n))
    return fyf_total, fyr_total, alpha_f, alpha_r


def hybrid_vehicle_state_derivative(
    state: np.ndarray,
    measurement_window_flat: np.ndarray,
    delta_rad: float,
    vx_mps: float,
    params: HybridVehicleParams,
    model: FrictionResidualModel,
) -> np.ndarray:
    fyf_total, fyr_total, _, _ = predict_total_lateral_forces(model, measurement_window_flat, state, delta_rad, vx_mps, params)
    vx_safe = safe_speed(vx_mps)
    beta_dot = (fyf_total + fyr_total) / (params.m_kg * vx_safe) - float(state[1])
    yaw_rate_dot = (params.lf_m * fyf_total - params.lr_m * fyr_total) / params.iz_kgm2
    return np.array([beta_dot, yaw_rate_dot], dtype=float)


def hybrid_vehicle_fx(
    state: np.ndarray,
    dt: float,
    measurement_window_flat: np.ndarray,
    delta_rad: float,
    vx_mps: float,
    params: HybridVehicleParams,
    model: FrictionResidualModel,
) -> np.ndarray:
    derivative = hybrid_vehicle_state_derivative(state, measurement_window_flat, delta_rad, vx_mps, params, model)
    return state + dt * derivative


def hybrid_vehicle_hx(
    state: np.ndarray,
    measurement_window_flat: np.ndarray,
    delta_rad: float,
    vx_mps: float,
    params: HybridVehicleParams,
    model: FrictionResidualModel,
) -> np.ndarray:
    fyf_total, fyr_total, _, _ = predict_total_lateral_forces(model, measurement_window_flat, state, delta_rad, vx_mps, params)
    ay_pred = (fyf_total + fyr_total) / params.m_kg
    return np.array([float(state[1]), ay_pred], dtype=float)


def numerical_state_jacobian(
    func,
    state: np.ndarray,
    epsilon: float = 1.0e-5,
) -> np.ndarray:
    base = func(state)
    jacobian = np.zeros((len(base), len(state)), dtype=float)
    for idx in range(len(state)):
        perturbed = state.copy()
        perturbed[idx] += epsilon
        jacobian[:, idx] = (func(perturbed) - base) / epsilon
    return jacobian
