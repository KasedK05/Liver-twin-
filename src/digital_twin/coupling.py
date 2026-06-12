import numpy as np

class CouplingEngine:
    def __init__(self):
        self.vars = {}

    def compute(self, subsystems):
        s = {name: sub.state for name, sub in subsystems.items()}
        self.vars = {
            "atp": s["energy"]["atp"],
            "atp_demand": 0.03 + 0.01 * s["protein"]["translation"] + 0.01 * s["membrane"]["nutrient_uptake"],
            "atp_supply_ratio": s["energy"]["atp"] / max(0.01, 0.03 + 0.01 * s["protein"]["translation"]),
            "ros": s["defense"]["ros"],
            "oxidative_stress_index": max(0, s["defense"]["ros"] - s["defense"]["antioxidant"]),
            "inflammatory_tone": s["defense"]["inflammasome"] + 0.5 * s["signaling"]["cytokine"],
            "viral_load": s["defense"]["viral_load"],
            "dna_damage_signal": 1.0 - s["genetic"]["dna_integrity"],
            "dna_repair_activity": s["defense"]["dna_repair"],
            "autophagy_flux": s["waste"]["autophagy"],
            "metabolic_load": 0.5 + 0.3 * s["energy"]["glycolysis"],
            "mtor": s["signaling"]["mtor"],
            "ampk": 1.0 - s["signaling"]["mtor"],
            "inflammasome_activity": s["defense"]["inflammasome"],
            "aggregates": s["protein"]["aggregates"],
        }
        return self.vars
