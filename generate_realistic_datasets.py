import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
from src.data_engine.generator import DatasetGenerator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=["10k", "50k", "200k"], required=True)
    args = parser.parse_args()
    scale_map = {"10k": 10000, "50k": 50000, "200k": 200000}
    n_total = scale_map[args.scale]
    os.makedirs("data/raw", exist_ok=True)
    output_path = f"data/raw/realistic_{args.scale}.npz"
    gen = DatasetGenerator()
    gen.generate(n_total=n_total, output_path=output_path)
    print("Done.")

if __name__ == "__main__":
    main()
