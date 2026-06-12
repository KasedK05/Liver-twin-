import numpy as np
from .base import Subsystem

class DefenseStress(Subsystem):
    def initialize_state(self, params):
        return {"ros": params.get("ros_0", 0.10), "antioxidant": params.get("aod_0", 0.60), "dna_repair": params.get("repair_0", 0.20), "inflammasome": params.get("inflam_0", 0.05), "interferon": params.get("ifn_0", 0.05), "viral_load": 0.0}

    def update(self, dt, c, d):
        s = self.state
        s["viral_load"] = d.get("viral_load", s["viral_load"])
        ros_gen = 0.05 * c.get("metabolic_load", 0.5) + d.get("ros_boost", 0.0)
        ros_scav = 0.10 * s["antioxidant"] * s["ros"]
        s["ros"] += dt * (ros_gen - ros_scav)
        s["ros"] = np.clip(s["ros"], 0.0, 1.0)
        aod_boost = d.get("aod_boost", 0.0)
        aod_exhaust = 0.05 * s["ros"]
        s["antioxidant"] += dt * (0.02 * aod_boost - aod_exhaust)
        s["antioxidant"] = np.clip(s["antioxidant"], 0.0, 1.0)
        repair_trigger = 1.0 if s["ros"] > 0.4 else 0.0
        repair_cost = 0.02 * s["dna_repair"]
        s["dna_repair"] += dt * (0.3 * repair_trigger - repair_cost)
        s["dna_repair"] = np.clip(s["dna_repair"], 0.0, 1.0)
        inflam_trigger = 1.0 if (s["ros"] > 0.5 and s["antioxidant"] < 0.3) else 0.0
        s["inflammasome"] += dt * (0.4 * inflam_trigger - 0.1 * s["inflammasome"])
        s["inflammasome"] = np.clip(s["inflammasome"], 0.0, 1.0)
        ifn_trigger = s["viral_load"] * (1 - d.get("ifn_suppress", 0.0))
        ifn_decay = 0.05 * s["interferon"]
        s["interferon"] += dt * (0.2 * ifn_trigger - ifn_decay)
        s["interferon"] = np.clip(s["interferon"], 0.0, 1.0)

    def get_health_score(self):
        s = self.state
        return 0.3 * (1 - s["ros"]) + 0.3 * s["antioxidant"] + 0.2 * s["dna_repair"] + 0.2 * (1 - s["inflammasome"])
