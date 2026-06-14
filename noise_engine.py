import numpy as np

class NoiseEngine:
    """Applies realistic biological and technical noise."""

    def __init__(self, config):
        self.config = config

    def apply(self, trajectories, mode="training"):
        """Apply noise based on mode: training/validation/test."""
        cfg = self.config.get(mode, {})
        if not cfg:
            return trajectories

        noisy = trajectories.copy().astype(np.float32)

        # Instrument noise
        sigma = cfg.get("instrument_noise_sigma", 0.0)
        if sigma > 0:
            noise = np.random.normal(0, sigma, noisy.shape)
            noisy = np.clip(noisy + noise, 0, 1)

        # Count noise (Poisson) for low-abundance variables
        if cfg.get("count_noise", False):
            low_abundance_vars = [10, 11, 22, 25, 26, 27, 30, 31]  # UPR, aggregates, ROS, inflammasome, IFN, viral, waste, lipofuscin
            for var in low_abundance_vars:
                vals = noisy[:, :, var]
                poisson_noise = np.random.poisson(vals * 100) / 100.0 - vals
                noisy[:, :, var] = np.clip(vals + poisson_noise, 0, 1)

        # Saturation
        if cfg.get("saturation_clip"):
            low, high = cfg["saturation_clip"]
            noisy = np.clip(noisy, low, high)

        return noisy
