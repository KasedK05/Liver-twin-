import numpy as np

class Augmenter:
    def __init__(self, sigma=0.02):
        self.sigma = sigma

    def jitter(self, traj):
        noise = np.random.normal(0, self.sigma, traj.shape)
        return np.clip(traj + noise, 0, 1)

    def scale(self, traj, factor_range=(0.9, 1.1)):
        factor = np.random.uniform(*factor_range)
        return np.clip(traj * factor, 0, 1)

    def time_warp(self, traj, sigma=0.2):
        n_steps = traj.shape[0]
        warp = np.cumsum(np.random.normal(1, sigma, n_steps))
        warp = (warp - warp.min()) / (warp.max() - warp.min()) * (n_steps - 1)
        indices = np.round(warp).astype(int)
        indices = np.clip(indices, 0, n_steps - 1)
        return traj[indices]

    def magnitude_warp(self, traj, sigma=0.2):
        n_steps = traj.shape[0]
        warp = np.random.normal(1, sigma, n_steps)
        return np.clip(traj * warp[:, None], 0, 1)

    def generate(self, traj, n_needed, methods=["jitter", "scale", "time_warp", "magnitude_warp"]):
        augmented = []
        for _ in range(n_needed):
            method = np.random.choice(methods)
            t = traj.copy()
            if method == "jitter":
                t = self.jitter(t)
            elif method == "scale":
                t = self.scale(t)
            elif method == "time_warp":
                t = self.time_warp(t)
            elif method == "magnitude_warp":
                t = self.magnitude_warp(t)
            augmented.append(t)
        return augmented
