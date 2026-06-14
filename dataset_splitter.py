import numpy as np
from pathlib import Path

def split_dataset(input_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Split dataset into train/val/test BEFORE any modification."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    data = np.load(input_path)
    trajectories = data["trajectories"]
    labels = data["labels"]
    mechanisms = data["mechanisms"]
    subsystem_states = data["subsystem_states"]
    summaries = data["summaries"]
    times = data["times"]

    n = len(trajectories)
    np.random.seed(seed)
    indices = np.random.permutation(n)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    output_dir = Path(input_path).parent

    for split_name, split_idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        np.savez(output_dir / f"{split_name}.npz",
                 trajectories=trajectories[split_idx],
                 labels=labels[split_idx],
                 mechanisms=mechanisms[split_idx],
                 subsystem_states=subsystem_states[split_idx],
                 summaries=summaries[split_idx],
                 times=times)
        print(f"Saved {split_name}: {len(split_idx)} trajectories")

    print(f"Split complete: {n_train} train / {n_val} val / {len(test_idx)} test")
