import numpy as np

class MissingnessEngine:
    """Inject missing values into trajectories."""

    def __init__(self, config):
        self.config = config

    def apply(self, trajectories, mode="training"):
        cfg = self.config.get(mode, {})
        if not cfg:
            return trajectories, None

        n, t, f = trajectories.shape
        masked = trajectories.copy()
        mask = np.zeros((n, t), dtype=bool)

        # Time-point dropout
        dropout_rate = cfg.get("timepoint_dropout", 0.0)
        if dropout_rate > 0:
            for i in range(n):
                n_drop = int(t * dropout_rate)
                drop_indices = np.random.choice(t, n_drop, replace=False)
                masked[i, drop_indices, :] = np.nan
                mask[i, drop_indices] = True

        # Variable dropout (entire subsystem missing)
        var_dropout = cfg.get("variable_dropout", 0.0)
        if var_dropout > 0:
            n_drop = int(n * var_dropout)
            drop_indices = np.random.choice(n, n_drop, replace=False)
            for i in drop_indices:
                # Randomly drop one subsystem (4-6 variables)
                subsystems = [(0,4), (4,9), (9,13), (13,17), (17,22), (22,28), (28,32)]
                sub_idx = np.random.randint(7)
                start, end = subsystems[sub_idx]
                masked[i, :, start:end] = np.nan

        # Endpoint censoring
        censor_rate = cfg.get("endpoint_censoring", 0.0)
        if censor_rate > 0:
            n_censor = int(n * censor_rate)
            censor_indices = np.random.choice(n, n_censor, replace=False)
            for i in censor_indices:
                masked[i, 1:-1, :] = np.nan  # Keep only first and last
                mask[i, 1:-1] = True

        return masked, mask
