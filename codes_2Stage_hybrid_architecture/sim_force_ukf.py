from vehicle_force_ukf import parse_args
from vehicle_force_ukf import run_vehicle_force_ukf


if __name__ == "__main__":
    args = parse_args()
    run_vehicle_force_ukf(args.data, args.model, args.plot, device=args.device)
