import argparse

import numpy as np

from vehicle_filter_common import load_vehicle_sim_prepared_data
from vehicle_filter_common import reconstruct_path
from vehicle_filter_common import rmse
from vehicle_filter_common import save_vehicle_sim_plot
from vehicle_filter_common import vehicle_fx


def create_gaussian_particles(mean: np.ndarray, std: np.ndarray, count: int) -> np.ndarray:
    particles = np.empty((count, 2), dtype=float)
    particles[:, 0] = mean[0] + np.random.randn(count) * std[0]
    particles[:, 1] = mean[1] + np.random.randn(count) * std[1]
    return particles


def predict(
    particles: np.ndarray,
    dt: float,
    ay_input_mps2: float,
    yaw_acc_input_rps2: float,
    vx_mps: float,
    process_std: tuple[float, float],
) -> None:
    for i in range(len(particles)):
        particles[i] = vehicle_fx(
            particles[i],
            dt,
            ay_input_mps2,
            yaw_acc_input_rps2,
            vx_mps,
        )
    particles[:, 0] += np.random.randn(len(particles)) * process_std[0]
    particles[:, 1] += np.random.randn(len(particles)) * process_std[1]


def update(weights: np.ndarray, particles: np.ndarray, yaw_rate_meas: float, meas_std: float) -> None:
    yaw_error = (yaw_rate_meas - particles[:, 1]) / meas_std
    weights *= np.exp(-0.5 * (yaw_error * yaw_error))
    weights += 1.0e-300
    weights /= np.sum(weights)


def estimate(particles: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.array(
        [
            np.average(particles[:, 0], weights=weights),
            np.average(particles[:, 1], weights=weights),
        ],
        dtype=float,
    )


def neff(weights: np.ndarray) -> float:
    return 1.0 / np.sum(np.square(weights))


def systematic_resample(weights: np.ndarray) -> np.ndarray:
    count = len(weights)
    positions = (np.arange(count) + np.random.random()) / count
    indexes = np.zeros(count, dtype=int)
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1.0

    i = 0
    j = 0
    while i < count:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


def resample_from_index(particles: np.ndarray, weights: np.ndarray, indexes: np.ndarray) -> None:
    particles[:] = particles[indexes]
    weights.fill(1.0 / len(weights))


def run_vehicle_pf(data_file: str, plot_file: str, particle_count: int) -> None:
    data = load_vehicle_sim_prepared_data(data_file)

    particles = create_gaussian_particles(
        mean=np.array([data.beta_true_rad[0], data.yaw_rate_rps[0]], dtype=float),
        std=np.array([0.02, 0.05], dtype=float),
        count=particle_count,
    )
    weights = np.ones(particle_count, dtype=float) / particle_count
    estimates = [estimate(particles, weights)]

    for i in range(1, len(data.time_s)):
        dt = data.time_s[i] - data.time_s[i - 1]
        predict(
            particles,
            dt=dt,
            ay_input_mps2=float(data.ay_mps2[i]),
            yaw_acc_input_rps2=float(data.yaw_acc_rps2[i]),
            vx_mps=float(data.vx_mps[i]),
            process_std=(0.002, 0.03),
        )

        update(
            weights,
            particles,
            yaw_rate_meas=float(data.yaw_rate_rps[i]),
            meas_std=0.03,
        )

        if neff(weights) < particle_count / 2.0:
            indexes = systematic_resample(weights)
            resample_from_index(particles, weights, indexes)

        estimates.append(estimate(particles, weights))

    estimates = np.asarray(estimates)
    beta_rmse = rmse(data.beta_true_rad, estimates[:, 0])
    yaw_rate_rmse = rmse(data.yaw_rate_rps, estimates[:, 1])
    path_x_est, path_y_est, _ = reconstruct_path(data.time_s, data.vx_mps, estimates[:, 0], estimates[:, 1], data.yaw_true_rad[0])
    path_rmse = rmse(np.hypot(data.global_x_m, data.global_y_m), np.hypot(path_x_est, path_y_est))

    print(f"Vehicle PF beta RMSE: {beta_rmse:.6f} rad")
    print(f"Vehicle PF yaw-rate RMSE: {yaw_rate_rmse:.6f} rad/s")
    print(f"Vehicle PF path RMSE: {path_rmse:.6f} m")

    save_vehicle_sim_plot(data, estimates[:, 0], estimates[:, 1], plot_file, "Vehicle PF")
    print(f"Saved plot to {plot_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PF on prepared vehicle simulation data.")
    parser.add_argument("--data", default="vehicle_sim_data.npz", help="Prepared NPZ data file.")
    parser.add_argument("--plot", default="vehicle_pf.png", help="Output plot filename.")
    parser.add_argument("--particles", type=int, default=2000, help="Number of particles.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_vehicle_pf(args.data, args.plot, args.particles)
