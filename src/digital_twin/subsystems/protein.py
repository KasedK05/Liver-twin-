import numpy as np
from .base import Subsystem

class ProteinHomeostasis(Subsystem):
    def initialize_state(self, params):
        return {"translation": params.get("trans_0", 0.60), "upr": params.get("upr_0", 0.05), "aggregates": params.get("agg_0", 0.05), "secretory": params.get("sec_0", 0.40)}

    def update(self, dt, c, d):
        s = self.state
        trans_capacity = 0.5 * c.get("atp", 0.5)
        trans_inhibit = d.get("trans_inhibit", 0.0)
        s["translation"] += dt * (0.05 * (trans_capacity - s["translation"]) - trans_inhibit)
        s["translation"] = np.clip(s["translation"], 0.0, 1.0)
        upr_trigger = 1.0 if (s["aggregates"] > 0.4 or d.get("proteasome_inhibit", 0.0) > 0.5) else 0.0
        upr_resolve = 0.1 * s["upr"] * c.get("atp", 0.5)
        s["upr"] += dt * (0.3 * upr_trigger - upr_resolve)
        s["upr"] = np.clip(s["upr"], 0.0, 1.0)
        agg_form = 0.05 * (1 - s["upr"]) * s["translation"]
        agg_clear = 0.10 * c.get("autophagy_flux", 0)
        s["aggregates"] += dt * (agg_form - agg_clear)
        s["aggregates"] = np.clip(s["aggregates"], 0.0, 1.0)
        sec_load = 0.03 * s["translation"]
        sec_capacity = 0.02 * c.get("atp", 0.5)
        s["secretory"] += dt * (sec_load - sec_capacity)
        s["secretory"] = np.clip(s["secretory"], 0.0, 1.0)

    def get_health_score(self):
        s = self.state
        return 0.3 * s["translation"] + 0.3 * (1 - s["upr"]) + 0.2 * (1 - s["aggregates"]) + 0.2 * (1 - s["secretory"])
