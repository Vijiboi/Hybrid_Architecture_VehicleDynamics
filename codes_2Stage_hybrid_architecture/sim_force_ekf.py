from vehicle_force_ekf import parse_args
from vehicle_force_ekf import run_vehicle_force_ekf


if __name__ == "__main__":
    args = parse_args()
    run_vehicle_force_ekf(args.data, args.model, args.plot, device=args.device)
