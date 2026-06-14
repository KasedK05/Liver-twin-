import numpy as np

class BatchEffectsEngine:
    """Simulate lab-to-lab calibration shifts."""

    def __init__(self, n_labs=3):
        self.n_labs = n_labs
        self.shifts = {}

    def apply(self, trajectories, lab_id=0):
        if lab_id not in self.shifts:
            # Generate random but consistent shift for this lab
            np.random.seed(lab_id)
            self.shifts[lab_id] = np.random.normal(0, 0.02, 32)

        shifted = trajectories + self.shifts[lab_id]
        return np.clip(shifted, 0, 1)
