import numpy as np
from .base import Subsystem

class MembraneTransport(Subsystem):
    def initialize_state(self, params):
        return {"integrity": params.get("mem_0", 0.90), "na_k_gradient": params.get("nak_0", 0.85), "calcium": params.get("ca_0", 0.10), "nutrient_uptake": params.get("nut_0", 0.70), "vesicle_traffic": params.get("ves_0", 0.60)}

    def update(self, dt, c, d):
        s = self.state
        mem_damage = 0.05 * c.get("ros", 0) + 0.03 * c.get("viral_load", 0) + d.get("mem_damage", 0.0)
        mem_repair = 0.04 * c.get("atp", 0.5)
        s["integrity"] += dt * (mem_repair - mem_damage)
        s["integrity"] = np.clip(s["integrity"], 0.0, 1.0)
        pump_capacity = 0.5 * c.get("atp", 0.5)
        leak = 0.03 * (1 - s["integrity"])
        s["na_k_gradient"] += dt * (0.05 * (pump_capacity - s["na_k_gradient"]) - leak)
        s["na_k_gradient"] = np.clip(s["na_k_gradient"], 0.0, 1.0)
        ca_influx = 0.1 * (1 - s["na_k_gradient"]) + d.get("ca_influx", 0.0)
        ca_efflux = 0.08 * c.get("atp", 0.5)
        s["calcium"] += dt * (ca_influx - ca_efflux)
        s["calcium"] = np.clip(s["calcium"], 0.0, 1.0)
        uptake_drive = 0.4 * s["integrity"]
        uptake_inhibit = 0.05 * c.get("cytokine", 0)
        s["nutrient_uptake"] += dt * (0.03 * (uptake_drive - s["nutrient_uptake"]) - uptake_inhibit)
        s["nutrient_uptake"] = np.clip(s["nutrient_uptake"], 0.0, 1.0)
        ves_drive = 0.3 * c.get("atp", 0.5)
        ves_disrupt = 0.04 * c.get("aggregates", 0.0)
        s["vesicle_traffic"] += dt * (0.03 * (ves_drive - s["vesicle_traffic"]) - ves_disrupt)
        s["vesicle_traffic"] = np.clip(s["vesicle_traffic"], 0.0, 1.0)

    def get_health_score(self):
        s = self.state
        return 0.3 * s["integrity"] + 0.2 * s["na_k_gradient"] + 0.2 * (1 - s["calcium"]) + 0.2 * s["nutrient_uptake"] + 0.1 * s["vesicle_traffic"]
