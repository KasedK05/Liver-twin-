import numpy as np
from .base import Subsystem

class GeneticRegulation(Subsystem):
    def initialize_state(self, params):
        return {"dna_integrity": params.get("dna_0", 0.95), "transcription": params.get("tx_0", 0.70), "chromatin": params.get("chrom_0", 0.80), "cell_cycle": params.get("cycle_0", 0.50)}

    def update(self, dt, c, d):
        s = self.state
        damage = 0.05 * c.get("ros", 0) + 0.03 * c.get("viral_load", 0) + d.get("dna_damage", 0.0)
        repair = 0.10 * c.get("dna_repair_activity", 0)
        s["dna_integrity"] += dt * (repair - damage)
        s["dna_integrity"] = np.clip(s["dna_integrity"], 0.0, 1.0)
        tx_drive = 0.5 * s["dna_integrity"] + 0.3 * c.get("atp", 0.5)
        tx_suppress = d.get("tx_suppress", 0.0)
        s["transcription"] += dt * (0.05 * (tx_drive - s["transcription"]) - tx_suppress)
        s["transcription"] = np.clip(s["transcription"], 0.0, 1.0)
        chrom_stress = 0.03 * c.get("ros", 0)
        chrom_relax = 0.02 * (1 - s["chromatin"])
        s["chromatin"] += dt * (chrom_relax - chrom_stress)
        s["chromatin"] = np.clip(s["chromatin"], 0.0, 1.0)
        cycle_speed = 0.02 * s["transcription"] * c.get("atp", 0.5)
        arrest = 1.0 if s["dna_integrity"] < 0.3 else 0.0
        s["cell_cycle"] += dt * (cycle_speed * (1 - arrest))
        s["cell_cycle"] = s["cell_cycle"] % 1.0

    def get_health_score(self):
        s = self.state
        return 0.5 * s["dna_integrity"] + 0.3 * s["transcription"] + 0.2 * s["chromatin"]
