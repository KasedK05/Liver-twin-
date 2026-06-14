import numpy as np
import torch
from ..digital_twin.gpu_simulator import CellSimulatorGPU
from .augmentation import Augmenter

CLASSES = ['CVR', 'BRT', 'PVR', 'VBR', 'TRF', 'SMR']

TARGET_PREVALENCE = {
    'CVR': 0.20,
    'BRT': 0.20,
    'PVR': 0.15,
    'VBR': 0.15,
    'TRF': 0.15,
    'SMR': 0.15
}

class DatasetGenerator:
    def __init__(self, device="cuda"):
        self.sim = CellSimulatorGPU(device=device)
        self.augmenter = Augmenter()

        # Each class has its own targeted scenario pool
        self.scenario_pools = {
            'CVR': [  # Controlled Viral Response — effective antiviral
                {"drug": "entecavir", "dose": 0.8, "virus": "hbv_wildtype"},
                {"drug": "entecavir", "dose": 1.0, "virus": "hbv_wildtype"},
            ],
            'BRT': [  # Baseline / Recovery — no stress
                {"drug": None, "dose": 0.0, "virus": None},
                {"drug": "dexamethasone", "dose": 0.3, "virus": None},
            ],
            'PVR': [  # Prolonged Viral — partial control
                {"drug": "entecavir", "dose": 0.3, "virus": "hbv_wildtype"},
                {"drug": "entecavir", "dose": 0.2, "virus": "hbv_ymdd"},
            ],
            'VBR': [  # Viral Breakthrough — resistant strain
                {"drug": "entecavir", "dose": 0.2, "virus": "hbv_ymdd"},
                {"drug": None, "dose": 0.0, "virus": "hbv_ymdd"},
            ],
            'TRF': [  # Treatment Failure — severe stress but not dead
                {"drug": "rotenone", "dose": 0.3, "virus": "hbv_wildtype"},
                {"drug": "rotenone", "dose": 0.5, "virus": None},
                {"drug": "bortezomib", "dose": 0.5, "virus": None},
            ],
            'SMR': [  # Severe Metabolic Response — mitochondrial collapse
                {"drug": "rotenone", "dose": 1.0, "virus": None},
                {"drug": "rotenone", "dose": 0.8, "virus": "hbv_ymdd"},
            ],
        }

    def generate(self, n_total, output_path, max_attempts=4):
        print(f"=== Generating {n_total} trajectories (6-class stratified) ===")
        target_counts = {k: int(v * n_total) for k, v in TARGET_PREVALENCE.items()}
        # fix rounding so sum equals n_total
        diff = n_total - sum(target_counts.values())
        target_counts['CVR'] += diff

        print(f"Target distribution: {target_counts}")
        collected = {k: [] for k in target_counts}
        batch_size = 1024

        for cls in CLASSES:
            n_needed = target_counts[cls]
            pool = self.scenario_pools[cls]
            print(f"\nGenerating {n_needed} for class {cls}...")
            attempts = 0
            sim_count = 0

            while len(collected[cls]) < n_needed and attempts < max_attempts:
                need = n_needed - len(collected[cls])
                current_bs = min(batch_size, need * 2)
                # round-robin scenarios from this class's pool
                scenarios = [pool[i % len(pool)] for i in range(current_bs)]
                batch = [{"scenario": s, "params": {}} for s in scenarios]

                trajs, _ = self.sim.run(batch, t_max=24.0, dt=0.5, n_steps=49)
                outcomes = self.sim.classify_subsystems(trajs)
                sim_count += current_bs
                attempts += 1

                for i, out in enumerate(outcomes):
                    if out == cls and len(collected[cls]) < n_needed:
                        collected[cls].append(trajs[i])

                print(f"  Attempt {attempts}: {len(collected[cls])}/{n_needed} "
                      f"(simulated {sim_count})")

            # Augment shortfall if needed
            if len(collected[cls]) < n_needed:
                short = n_needed - len(collected[cls])
                sources = collected[cls]
                if len(sources) == 0:
                    # extreme fallback: duplicate last known good trajectory
                    # or use another class's samples (shouldn't happen)
                    print(f"  WARNING: no sources for {cls}, filling with zeros")
                    collected[cls] = [np.zeros((49, 32)) for _ in range(n_needed)]
                else:
                    print(f"  Augmenting {short} for {cls}")
                    for _ in range(short):
                        src = sources[np.random.randint(len(sources))]
                        aug = self.augmenter.generate(src, 1)[0]
                        collected[cls].append(aug)

        # Assemble & shuffle
        final_trajs = []
        final_labels = []
        for cls in CLASSES:
            final_trajs.extend(collected[cls][:target_counts[cls]])
            final_labels.extend([cls] * target_counts[cls])

        final_trajs = np.array(final_trajs)
        final_labels = np.array(final_labels)
        perm = np.random.permutation(len(final_labels))
        final_trajs = final_trajs[perm]
        final_labels = final_labels[perm]

        times = np.arange(49) * 0.5
        np.savez(output_path, trajectories=final_trajs, labels=final_labels, times=times)
        print(f"\nSaved to {output_path}")
        print(f"Shape: {final_trajs.shape}")
        for cls in CLASSES:
            c = int((final_labels == cls).sum())
            print(f" {cls}: {c} ({100*c/len(final_labels):.1f}%)")