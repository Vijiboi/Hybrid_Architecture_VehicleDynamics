from vehicle_hybrid_ekf import parse_args
from vehicle_hybrid_ekf import run_vehicle_hybrid_ekf


if __name__ == "__main__":
    args = parse_args()
    run_vehicle_hybrid_ekf(args.data, args.model, args.plot)
