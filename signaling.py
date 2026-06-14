import numpy as np
from .base import Subsystem

class SignalingControl(Subsystem):
    def initialize_state(self, params):
        return {
            "mtor": params.get("mtor_0", 0.50),
            "p53": params.get("p53_0", 0.10),
            "growth_factor": params.get("gf_0", 0.40),
            "cytokines": params.get("cyto_0", 0.10),
        }

    def update(self, dt, c, d):
        s = self.state
        mtor_drive = 0.5 * c.get("atp", 0.5) + 0.3 * s["growth_factor"]
        mtor_inhibit = 0.2 * c.get("ros", 0) + d.get("mtor_inhibit", 0.0)
        s["mtor"] += dt * (0.05 * (mtor_drive - s["mtor"]) - mtor_inhibit)
        s["mtor"] = np.clip(s["mtor"], 0.0, 1.0)

        p53_trigger = 1.0 if (c.get("dna_damage_signal", 0) > 0.5 or c.get("ros", 0) > 0.6) else 0.0
        p53_decay = 0.08 * s["p53"]
        s["p53"] += dt * (0.4 * p53_trigger - p53_decay)
        s["p53"] = np.clip(s["p53"], 0.0, 1.0)

        gf_decay = 0.05 * s["growth_factor"]
        s["growth_factor"] += dt * (-gf_decay)
        s["growth_factor"] = np.clip(s["growth_factor"], 0.0, 1.0)

        cyto_drive = 0.3 * c.get("inflammasome_activity", 0) + 0.2 * c.get("viral_load", 0)
        s["cytokines"] += dt * (0.05 * cyto_drive - 0.03 * s["cytokines"] - d.get("cytokine_suppress", 0.0))
        s["cytokines"] = np.clip(s["cytokines"], 0.0, 1.0)

    def get_health_score(self):
        s = self.state
        return 0.3 * s["mtor"] + 0.3 * (1 - s["p53"]) + 0.2 * s["growth_factor"] + 0.2 * (1 - s["cytokines"])
