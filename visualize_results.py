import numpy as np
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path

CLASS_NAMES = ['CVR', 'BRT', 'PVR', 'VBR', 'TRF', 'SMR']

def plot_training_curves(results_dir="results", output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)
    architectures = ['lstm', 'large_lstm', 'transformer']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, arch in enumerate(architectures):
        path = os.path.join(results_dir, f"{arch}_results.json")
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping...")
            continue

        with open(path, 'r') as f:
            data = json.load(f)

        history = data['history']
        epochs = [h['epoch'] for h in history]
        train_loss = [h['train_loss'] for h in history]
        val_loss = [h['val_loss'] for h in history]
        val_acc = [h['val_acc'] for h in history]

        ax = axes[idx]
        ax2 = ax.twinx()

        l1, = ax.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=1.5)
        l2, = ax.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=1.5)
        l3, = ax2.plot(epochs, val_acc, 'g--', label='Val Acc', linewidth=1.5)

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax2.set_ylabel('Accuracy', fontsize=11)
        ax.set_title(f"{arch.upper().replace('_', ' ')}\nBest Epoch: {data['best_epoch']}, Test Acc: {data['test_accuracy']:.3f}", fontsize=12)
        ax.legend([l1, l2, l3], ['Train Loss', 'Val Loss', 'Val Acc'], loc='center right', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/training_curves.png")

def plot_confusion_matrices(results_dir="results", output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)
    architectures = ['lstm', 'large_lstm', 'transformer']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, arch in enumerate(architectures):
        path = os.path.join(results_dir, f"{arch}_results.json")
        if not os.path.exists(path):
            continue

        with open(path, 'r') as f:
            data = json.load(f)

        cm = np.array(data['confusion_matrix'])

        ax = axes[idx]
        im = ax.imshow(cm, cmap='Blues', aspect='auto')
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        ax.set_xticklabels(CLASS_NAMES, fontsize=10)
        ax.set_yticklabels(CLASS_NAMES, fontsize=10)
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('True', fontsize=11)
        ax.set_title(f"{arch.upper().replace('_', ' ')}\nAccuracy: {data['test_accuracy']:.3f}", fontsize=12)

        for i in range(6):
            for j in range(6):
                color = 'white' if cm[i, j] > cm.max() * 0.6 else 'black'
                ax.text(j, i, cm[i, j], ha="center", va="center", color=color, fontsize=10, fontweight='bold')

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/confusion_matrices.png")

def plot_comparison_bar(results_dir="results", output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)

    architectures = ['lstm', 'large_lstm', 'transformer']
    accuracies = []
    f1s = []
    params = []

    for arch in architectures:
        path = os.path.join(results_dir, f"{arch}_results.json")
        if not os.path.exists(path):
            continue
        with open(path, 'r') as f:
            data = json.load(f)
        accuracies.append(data['test_accuracy'])
        f1s.append(data['test_f1'])
        params.append(data['parameters'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(architectures))
    width = 0.35

    bars1 = ax1.bar(x - width/2, accuracies, width, label='Accuracy', color='steelblue', edgecolor='black', linewidth=0.5)
    bars2 = ax1.bar(x + width/2, f1s, width, label='F1-Score', color='coral', edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Model Performance on Unseen Test Set', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([a.upper().replace('_', ' ') for a in architectures], fontsize=11)
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3, axis='y')

    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.bar(x, [p/1000 for p in params], color='seagreen', edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Parameters (thousands)', fontsize=12)
    ax2.set_title('Model Complexity', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([a.upper().replace('_', ' ') for a in architectures], fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(params):
        ax2.text(i, v/1000 + max(params)/1000*0.05, f'{v:,}', ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/comparison.png")

def plot_sample_trajectories(dataset_path="data/raw/test_unseen_3k.npz", output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(dataset_path):
        print(f"Warning: {dataset_path} not found, skipping trajectory plot...")
        return

    data = np.load(dataset_path)
    trajs = data['trajectories']
    labels = data['labels']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, cls in enumerate(CLASS_NAMES):
        mask = labels == cls
        cls_trajs = trajs[mask]
        if len(cls_trajs) == 0:
            continue

        ax = axes[idx]
        time = np.arange(49) * 0.5

        atp_mean = cls_trajs[:, :, 4].mean(axis=0)
        atp_std = cls_trajs[:, :, 4].std(axis=0)
        ax.plot(time, atp_mean, 'b-', label='ATP', linewidth=2.5)
        ax.fill_between(time, atp_mean - atp_std, atp_mean + atp_std, alpha=0.2, color='b')

        vir_mean = cls_trajs[:, :, 27].mean(axis=0)
        vir_std = cls_trajs[:, :, 27].std(axis=0)
        ax.plot(time, vir_mean, 'r-', label='Viral Load', linewidth=2.5)
        ax.fill_between(time, vir_mean - vir_std, vir_mean + vir_std, alpha=0.2, color='r')

        ros_mean = cls_trajs[:, :, 22].mean(axis=0)
        ros_std = cls_trajs[:, :, 22].std(axis=0)
        ax.plot(time, ros_mean, 'g-', label='ROS', linewidth=2.5)
        ax.fill_between(time, ros_mean - ros_std, ros_mean + ros_std, alpha=0.2, color='g')

        ax.set_title(f'{cls} (n={len(cls_trajs)})', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (h)', fontsize=11)
        ax.set_ylabel('Normalized Level', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_trajectories.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/sample_trajectories.png")

def generate_all_plots(results_dir="results", dataset_path="data/raw/test_unseen_3k.npz", output_dir="figures"):
    print("Generating visualization report...")
    plot_training_curves(results_dir, output_dir)
    plot_confusion_matrices(results_dir, output_dir)
    plot_comparison_bar(results_dir, output_dir)
    plot_sample_trajectories(dataset_path, output_dir)
    print(f"\nAll figures saved to {output_dir}/")
    print("Files created:")
    print("  - training_curves.png")
    print("  - confusion_matrices.png")
    print("  - comparison.png")
    print("  - sample_trajectories.png")

if __name__ == "__main__":
    generate_all_plots()
