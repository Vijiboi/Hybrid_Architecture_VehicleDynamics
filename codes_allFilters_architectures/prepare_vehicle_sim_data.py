import argparse
import numpy as np

from vehicle_sim_loader import load_vehicle_sim_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare one vehicle simulation CSV scenario for EKF/UKF/PF.")
    parser.add_argument("--csv", required=True, help="Scenario CSV file path.")
    parser.add_argument("--metadata", default=None, help="Optional metadata JSON path.")
    parser.add_argument("--output", default="vehicle_sim_data.npz", help="Output NPZ filename.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_vehicle_sim_csv(args.csv, args.metadata)
    np.savez_compressed(args.output, **dataset.to_dict())

    print(f"Saved prepared vehicle simulation data to {args.output}")
    print(f"Samples: {len(dataset.time_s)}")
    print(f"Duration: {dataset.time_s[-1] - dataset.time_s[0]:.3f} s")
    print(f"Estimated Cf: {dataset.cf_nprad:.3f} N/rad")
    print(f"Estimated Cr: {dataset.cr_nprad:.3f} N/rad")


if __name__ == "__main__":
    main()
