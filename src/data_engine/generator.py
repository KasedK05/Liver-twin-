import numpy as np
import torch
from ..digital_twin.gpu_simulator import CellSimulatorGPU
from .augmentation import Augmenter
from .imbalance import TARGET_PREVALENCE

class DatasetGenerator:
    def __init__(self, device="cuda"):
        self.sim = CellSimulatorGPU(device=device)
        self.augmenter = Augmenter()
        self.scenarios = [
            {"drug": None, "dose": 0.0, "virus": None, "weight": 0.3},
            {"drug": "entecavir", "dose": 0.8, "virus": "hbv_wildtype", "weight": 0.5},
            {"drug": "dexamethasone", "dose": 0.6, "virus": "hbv_wildtype", "weight": 0.4},
            {"drug": "NAC", "dose": 0.6, "virus": "hbv_wildtype", "weight": 0.4},
            {"drug": "entecavir", "dose": 0.2, "virus": "hbv_ymdd", "weight": 6.0},
            {"drug": "rotenone", "dose": 0.3, "virus": "hbv_wildtype", "weight": 6.0},
            {"drug": "rotenone", "dose": 0.5, "virus": None, "weight": 5.0},
            {"drug": None, "dose": 0.0, "virus": "hbv_ymdd", "weight": 4.0},
            {"drug": "bortezomib", "dose": 0.5, "virus": None, "weight": 4.0},
            {"drug": "rotenone", "dose": 1.0, "virus": None, "weight": 12.0},
            {"drug": None, "dose": 0.0, "virus": "adenovirus_5", "weight": 10.0},
            {"drug": "rotenone", "dose": 0.8, "virus": "hbv_ymdd", "weight": 12.0},
            {"drug": None, "dose": 0.0, "virus": "hbv_ymdd", "weight": 8.0},
        ]
        self.weights = [s["weight"] for s in self.scenarios]

    def _sample_scenario(self):
        probs = np.array(self.weights) / sum(self.weights)
        idx = np.random.choice(len(self.scenarios), p=probs)
        return self.scenarios[idx]

    def generate(self, n_total, output_path, max_attempts=4):
        print(f"=== Generating {n_total} trajectories (GPU-only) ===")
        target_counts = {k: int(v * n_total) for k, v in TARGET_PREVALENCE.items()}
        print(f"Target distribution: {target_counts}")
        collected = {k: [] for k in target_counts}
        total_generated = 0
        batch_size = 1024

        for attempt in range(max_attempts):
            need = {k: target_counts[k] - len(v) for k, v in collected.items()}
            total_need = sum(max(0, v) for v in need.values())
            if total_need == 0:
                break
            pool_size = min(total_need * 2, 20000)
            n_batches = (pool_size + batch_size - 1) // batch_size
            print(f"  Attempt {attempt+1}: need {total_need}, pool {pool_size} in {n_batches} GPU batches...")

            for b in range(n_batches):
                current_bs = min(batch_size, pool_size - b * batch_size)
                scenario = self._sample_scenario()
                batch = [{"scenario": scenario, "params": {}} for _ in range(current_bs)]
                trajs, times = self.sim.run(batch, t_max=24.0, dt=0.5, n_steps=49)
                outcomes = self.sim.classify_outcomes(trajs)
                total_generated += current_bs

                for i, out in enumerate(outcomes):
                    if need.get(out, 0) > 0:
                        collected[out].append(trajs[i])
                        need[out] -= 1

                current = sum(len(v) for v in collected.values())
                if b % 5 == 0 or b == n_batches - 1:
                    print(f"    Batch {b+1}/{n_batches}: {current}/{n_total} (sim'd: {total_generated})")

                if total_need == 0:
                    break

            current = sum(len(v) for v in collected.values())
            print(f"    Attempt complete. Current: {current}/{n_total}")

        shortfall = {k: target_counts[k] - len(v) for k, v in collected.items()}
        if any(v > 0 for v in shortfall.values()):
            print(f"  Augmenting to fill shortfall of {sum(shortfall.values())} trajectories...")
            for out, needed in shortfall.items():
                if needed <= 0:
                    continue
                if out == "decline":
                    sources = collected["decline"] if len(collected["decline"]) > 0 else collected["death"]
                elif out == "death":
                    sources = collected["death"] if len(collected["death"]) > 0 else collected["decline"]
                elif out == "stable":
                    sources = collected["stable"] if len(collected["stable"]) > 0 else collected["recovery"]
                else:
                    sources = collected[out] if len(collected[out]) > 0 else collected["stable"]

                if len(sources) == 0:
                    continue
                print(f"    Augmenting {out}: need {needed}, have {len(sources)}")
                for _ in range(needed):
                    src = sources[np.random.randint(len(sources))]
                    aug = self.augmenter.generate(src, 1)[0]
                    collected[out].append(aug)

        final_trajs = []
        final_labels = []
        for out in ["recovery", "stable", "decline", "death"]:
            final_trajs.extend(collected[out][:target_counts[out]])
            final_labels.extend([out] * len(collected[out][:target_counts[out]]))
        final_trajs = np.array(final_trajs)
        final_labels = np.array(final_labels)
        perm = np.random.permutation(len(final_labels))
        final_trajs = final_trajs[perm]
        final_labels = final_labels[perm]

        times = np.arange(49) * 0.5
        np.savez(output_path, trajectories=final_trajs, labels=final_labels, times=times)
        print()
        print(f"Saved to {output_path}")
        print(f"Shape: {final_trajs.shape}")
        for out in ["death", "decline", "recovery", "stable"]:
            c = int((final_labels == out).sum())
            print(f"  {out:10s}: {c:5d} ({100*c/len(final_labels):5.1f}%)")
        print(f"Total simulated: {total_generated}")
