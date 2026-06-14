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
    def __init__(self, device="cuda", biological_noise=0.005):
        self.sim = CellSimulatorGPU(device=device, biological_noise=biological_noise)
        self.augmenter = Augmenter()

        # TRAIN: each scenario empirically verified to produce its target class
        self.train_pools = {
            'CVR': [
                {"scenario": {"drug": "entecavir", "dose": 0.8, "virus": "hbv_wildtype", "patient_profile": "healthy"}, "params": {}},
                {"scenario": {"drug": "entecavir", "dose": 1.0, "virus": "hbv_wildtype", "patient_profile": "healthy"}, "params": {}},
                {"scenario": {"drug": "entecavir", "dose": 0.9, "virus": "hbv_wildtype", "patient_profile": "compromised"}, "params": {}},
                {"scenario": {"drug": "entecavir", "dose": 0.8, "virus": "hbv_wildtype", "patient_profile": "elderly"}, "params": {}},
                {"scenario": {"drug": "entecavir", "dose": 1.0, "virus": "hbv_ymdd", "patient_profile": "healthy"}, "params": {}},
                {"scenario": {"drug": "entecavir", "dose": 0.9, "virus": "hbv_wildtype", "patient_profile": "stressed"}, "params": {}},
            ],
            'BRT': [
                {"scenario": {"drug": None, "dose": 0.0, "virus": None, "patient_profile": "healthy"}, "params": {}},
                {"scenario": {"drug": "dexamethasone", "dose": 0.2, "virus": None, "patient_profile": "healthy"}, "params": {}},
                {"scenario": {"drug": None, "dose": 0.0, "virus": None, "patient_profile": "compromised"}, "params": {}},
                {"scenario": {"drug": "dexamethasone", "dose": 0.3, "virus": None, "patient_profile": "elderly"}, "params": {}},
                {"scenario": {"drug": None, "dose": 0.0, "virus": None, "patient_profile": "stressed"}, "params": {}},
            ],
            'PVR': [
                {"scenario": {"drug": "entecavir", "dose": 0.3, "virus": "hbv_wildtype", "patient_profile": "healthy"}, "params": {}},
                {"scenario": {"drug": "entecavir", "dose": 0.5, "virus": "hbv_wildtype", "patient_profile": "compromised"}, "params": {}},
                {"scenario": {"drug": "entecavir", "dose": 0.4, "virus": "hbv_wildtype", "patient_profile": "elderly"}, "params": {}},
                {"scenario": {"drug": "entecavir", "dose": 0.6, "virus": "hbv_ymdd", "patient_profile": "healthy"}, "params": {}},
                {"scenario": {"drug": "entecavir", "dose": 0.7, "virus": "hbv_ymdd", "patient_profile": "stressed"}, "params": {}},
            ],
            'VBR': [
                {"scenario": {"drug": "entecavir", "dose": 0.1, "virus": "hbv_ymdd", "patient_profile": "healthy"}, "params": {}},
                {"scenario": {"drug": None, "dose": 0.0, "virus": "hbv_ymdd", "patient_profile": "compromised"}, "params": {}},
                {"scenario": {"drug": "entecavir", "dose": 0.05, "virus": "hbv_ymdd", "patient_profile": "elderly"}, "params": {}},
                {"scenario": {"drug": None, "dose": 0.0, "virus": "hbv_ymdd", "patient_profile": "stressed"}, "params": {}},
                {"scenario": {"drug": "entecavir", "dose": 0.2, "virus": "hbv_ymdd", "patient_profile": "healthy"}, "params": {}},
            ],
            'TRF': [
                {"scenario": {"drug": "rotenone", "dose": 0.2, "virus": "hbv_wildtype", "patient_profile": "healthy"}, "params": {}},
                {"scenario": {"drug": "rotenone", "dose": 0.15, "virus": "hbv_wildtype", "patient_profile": "compromised"}, "params": {}},
                {"scenario": {"drug": "bortezomib", "dose": 0.3, "virus": "hbv_wildtype", "patient_profile": "elderly"}, "params": {}},
                {"scenario": {"drug": "rotenone", "dose": 0.1, "virus": "hbv_wildtype", "patient_profile": "stressed"}, "params": {}},
                {"scenario": {"drug": "entecavir", "dose": 0.15, "virus": "hbv_wildtype", "patient_profile": "compromised"}, "params": {}},
            ],
            'SMR': [
                {"scenario": {"drug": "rotenone", "dose": 0.8, "virus": None, "patient_profile": "healthy"}, "params": {}},
                {"scenario": {"drug": "rotenone", "dose": 1.0, "virus": None, "patient_profile": "compromised"}, "params": {}},
                {"scenario": {"drug": "rotenone", "dose": 0.9, "virus": None, "patient_profile": "elderly"}, "params": {}},
                {"scenario": {"drug": "rotenone", "dose": 0.6, "virus": None, "patient_profile": "stressed"}, "params": {}},
            ],
        }

        # TEST: unseen doses and profiles, but same drugs
        self.test_pools = {
            'CVR': [
                {"scenario": {"drug": "entecavir", "dose": 0.85, "virus": "hbv_wildtype", "patient_profile": "stressed"}, "params": {}},
                {"scenario": {"drug": "entecavir", "dose": 0.75, "virus": "hbv_wildtype", "patient_profile": "compromised"}, "params": {}},
            ],
            'BRT': [
                {"scenario": {"drug": "dexamethasone", "dose": 0.1, "virus": None, "patient_profile": "healthy"}, "params": {}},
                {"scenario": {"drug": None, "dose": 0.0, "virus": None, "patient_profile": "elderly"}, "params": {}},
            ],
            'PVR': [
                {"scenario": {"drug": "entecavir", "dose": 0.35, "virus": "hbv_wildtype", "patient_profile": "stressed"}, "params": {}},
                {"scenario": {"drug": "entecavir", "dose": 0.55, "virus": "hbv_ymdd", "patient_profile": "compromised"}, "params": {}},
            ],
            'VBR': [
                {"scenario": {"drug": "entecavir", "dose": 0.08, "virus": "hbv_ymdd", "patient_profile": "stressed"}, "params": {}},
                {"scenario": {"drug": None, "dose": 0.0, "virus": "hbv_ymdd", "patient_profile": "elderly"}, "params": {}},
            ],
            'TRF': [
                {"scenario": {"drug": "rotenone", "dose": 0.18, "virus": "hbv_wildtype", "patient_profile": "elderly"}, "params": {}},
                {"scenario": {"drug": "bortezomib", "dose": 0.35, "virus": "hbv_wildtype", "patient_profile": "stressed"}, "params": {}},
            ],
            'SMR': [
                {"scenario": {"drug": "rotenone", "dose": 0.85, "virus": None, "patient_profile": "stressed"}, "params": {}},
                {"scenario": {"drug": "rotenone", "dose": 0.7, "virus": None, "patient_profile": "compromised"}, "params": {}},
            ],
        }


    def _generate_class(self, cls, n_needed, pool, batch_size=8192, label_noise=0.0):
        cls_collected = []
        total_simulated = 0

        for scen_idx, item in enumerate(pool):
            if len(cls_collected) >= n_needed:
                break

            scenario = item["scenario"]
            scenario_simulated = 0
            scenario_matches = 0

            while len(cls_collected) < n_needed and scenario_simulated < 30000:
                need = n_needed - len(cls_collected)
                current_bs = min(batch_size, need)

                batch = [{"scenario": scenario, "params": {}} for _ in range(current_bs)]
                trajs, _ = self.sim.run(batch, t_max=24.0, dt=0.5, n_steps=49)
                outcomes = self.sim.classify_subsystems(trajs, label_noise=label_noise)
                total_simulated += current_bs
                scenario_simulated += current_bs

                matches = 0
                for i, out in enumerate(outcomes):
                    if out == cls and len(cls_collected) < n_needed:
                        cls_collected.append(trajs[i])
                        matches += 1
                scenario_matches += matches

                batches_tried = scenario_simulated // current_bs
                if batches_tried >= 3 and scenario_matches == 0:
                    print(f"    Scenario {scen_idx+1}: SKIP (0 matches after {scenario_simulated})")
                    break

            print(f"    Scenario {scen_idx+1}: {len(cls_collected)}/{n_needed} (+{scenario_matches} matches)")

        if len(cls_collected) < n_needed:
            short = n_needed - len(cls_collected)
            sources = cls_collected
            if len(sources) == 0:
                print(f"    WARNING: no sources for {cls}, filling with zeros")
                cls_collected = [np.zeros((49, 32)) for _ in range(n_needed)]
            else:
                print(f"    Augmenting {short} for {cls}")
                for _ in range(short):
                    src = sources[np.random.randint(len(sources))]
                    aug = self.augmenter.generate(src, 1)[0]
                    cls_collected.append(aug)

        return cls_collected

    def _generate_from_pool(self, n_total, pool, label_noise=0.0):
        target_counts = {k: int(v * n_total) for k, v in TARGET_PREVALENCE.items()}
        diff = n_total - sum(target_counts.values())
        target_counts['CVR'] += diff

        all_collected = []
        all_labels = []

        for cls in CLASSES:
            n_needed = target_counts[cls]
            print(f"  Generating {n_needed} for class {cls}...")
            cls_data = self._generate_class(cls, n_needed, pool[cls], label_noise=label_noise)
            all_collected.extend(cls_data)
            all_labels.extend([cls] * len(cls_data))

        perm = np.random.permutation(len(all_labels))
        all_collected = np.array(all_collected)[perm]
        all_labels = np.array(all_labels)[perm]

        return all_collected, all_labels

    def generate_train(self, n_total, output_path, label_noise=0.15):
        print(f"=== Generating {n_total} TRAINING trajectories (label_noise={label_noise}) ===")
        trajs, labels = self._generate_from_pool(n_total, self.train_pools, label_noise=label_noise)
        times = np.arange(49) * 0.5
        np.savez(output_path, trajectories=trajs, labels=labels, times=times)
        print(f"Saved to {output_path}")
        self._print_stats(labels)

    def generate_test(self, n_total, output_path, label_noise=0.0):
        print(f"=== Generating {n_total} TEST trajectories (label_noise={label_noise}) ===")
        trajs, labels = self._generate_from_pool(n_total, self.test_pools, label_noise=label_noise)
        times = np.arange(49) * 0.5
        np.savez(output_path, trajectories=trajs, labels=labels, times=times)
        print(f"Saved to {output_path}")
        self._print_stats(labels)

    def _print_stats(self, labels):
        for cls in CLASSES:
            c = int((labels == cls).sum())
            print(f" {cls}: {c} ({100*c/len(labels):.1f}%)")
