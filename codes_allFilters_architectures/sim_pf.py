from vehicle_pf import parse_args
from vehicle_pf import run_vehicle_pf


if __name__ == "__main__":
    args = parse_args()
    run_vehicle_pf(args.data, args.plot, args.particles)
