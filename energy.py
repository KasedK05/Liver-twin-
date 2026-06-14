import numpy as np
from .base import Subsystem

class EnergyMetabolism(Subsystem):
    def initialize_state(self, params):
        return {
            "atp": params.get("atp_0", 0.90),
            "nad": params.get("nad_0", 0.85),
            "mitochondria": params.get("mito_0", 0.88),
            "glycolysis": params.get("glyco_0", 0.30),
            "oxygen": params.get("o2_0", 0.95),
        }

    def update(self, dt, c, d):
        s = self.state
        atp_prod = 0.05 * s["mitochondria"] * s["oxygen"] * (1 + d.get("atp_boost", 0))
        atp_demand = c.get("atp_demand", 0.03)
        atp_decay = 0.02 * s["atp"]
        s["atp"] += dt * (atp_prod - atp_demand - atp_decay)
        s["atp"] = np.clip(s["atp"], 0.0, 1.0)

        nad_shift = 0.01 * (s["glycolysis"] - 0.3) - 0.02 * c.get("ros", 0)
        s["nad"] += dt * nad_shift
        s["nad"] = np.clip(s["nad"], 0.0, 1.0)

        mito_damage = d.get("mito_damage", 0.0) + 0.1 * c.get("ros", 0)
        mito_recovery = 0.03 * (1 - s["mitochondria"]) * s["atp"]
        s["mitochondria"] += dt * (mito_recovery - mito_damage)
        s["mitochondria"] = np.clip(s["mitochondria"], 0.0, 1.0)

        glyco_drive = 0.5 if s["mitochondria"] < 0.4 else 0.0
        s["glycolysis"] += dt * (0.02 * glyco_drive - 0.01 * (s["glycolysis"] - 0.3))
        s["glycolysis"] = np.clip(s["glycolysis"], 0.0, 1.0)

        o2_consump = 0.02 * s["mitochondria"] + 0.01 * s["glycolysis"]
        s["oxygen"] += dt * (0.05 * (0.95 - s["oxygen"]) - o2_consump)
        s["oxygen"] = np.clip(s["oxygen"], 0.0, 1.0)

    def get_health_score(self):
        s = self.state
        return 0.3 * s["atp"] + 0.2 * s["nad"] + 0.2 * s["mitochondria"] + 0.2 * s["oxygen"] + 0.1 * s["glycolysis"]
