from vehicle_ukf import parse_args
from vehicle_ukf import run_vehicle_ukf


if __name__ == "__main__":
    args = parse_args()
    run_vehicle_ukf(args.data, args.plot)
