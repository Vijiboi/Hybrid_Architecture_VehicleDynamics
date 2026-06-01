from vehicle_hybrid_ukf import parse_args
from vehicle_hybrid_ukf import run_vehicle_hybrid_ukf


if __name__ == "__main__":
    args = parse_args()
    run_vehicle_hybrid_ukf(args.data, args.model, args.plot)
