import os
import sys
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def _scale_label(n):
    if n >= 1000:
        return f"{n//1000}k"
    return str(n)

def main():
    parser = argparse.ArgumentParser(description="Digital Twin & AI Pipeline")
    parser.add_argument("--mode", choices=["validate", "generate", "train", "infer"], required=True)
    parser.add_argument("--dataset", choices=["train", "test", "10k", "50k", "200k"], default="train")
    parser.add_argument("--scale", type=int, default=10000)
    parser.add_argument("--arch", choices=["lstm", "large_lstm", "transformer", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()

    if args.mode == "validate":
        from src.digital_twin.validators import run_validations
        success = run_validations()
        sys.exit(0 if success else 1)

    elif args.mode == "generate":
        from src.data_engine.generator import DatasetGenerator
        os.makedirs("data/raw", exist_ok=True)
        gen = DatasetGenerator(biological_noise=0.005)

        scale_label = _scale_label(args.scale)

        if args.dataset == "train":
            output = f"data/raw/train_diverse_{scale_label}.npz"
            gen.generate_train(args.scale, output, label_noise=0.15)
        elif args.dataset == "test":
            output = f"data/raw/test_unseen_{scale_label}.npz"
            gen.generate_test(args.scale, output, label_noise=0.0)
        else:
            output = f"data/raw/realistic_{args.dataset}.npz"
            gen.generate_train(args.scale, output, label_noise=0.15)
        print("Done.")

    elif args.mode == "train":
        from src.training.engine import run_training, run_all_training

        train_label = _scale_label(args.scale)
        train_path = f"data/raw/train_diverse_{train_label}.npz"

        test_scale = int(args.scale * 0.3)
        test_label = _scale_label(test_scale)
        test_path = f"data/raw/test_unseen_{test_label}.npz"

        if not os.path.exists(train_path):
            print(f"[ERROR] Missing train dataset: {train_path}")
            print(f"[INFO] Run: python run.py --mode generate --dataset train --scale {args.scale}")
            sys.exit(1)
        if not os.path.exists(test_path):
            print(f"[ERROR] Missing test dataset: {test_path}")
            print(f"[INFO] Run: python run.py --mode generate --dataset test --scale {test_scale}")
            sys.exit(1)

        if args.arch == "all":
            run_all_training(train_path, test_path, args.epochs, args.batch_size, args.lr, args.device)
        else:
            run_training(args.arch, train_path, test_path, args.epochs, args.batch_size, args.lr, args.device)

    elif args.mode == "infer":
        print("[INFO] Infer mode: load model from models/ and run on new data.")

if __name__ == "__main__":
    main()