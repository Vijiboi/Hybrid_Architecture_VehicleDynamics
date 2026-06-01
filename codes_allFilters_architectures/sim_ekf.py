from vehicle_ekf import parse_args
from vehicle_ekf import run_vehicle_ekf


if __name__ == "__main__":
    args = parse_args()
    run_vehicle_ekf(args.data, args.plot)
