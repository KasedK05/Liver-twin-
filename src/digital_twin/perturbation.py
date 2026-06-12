import json
from pathlib import Path

class Perturbation:
    def __init__(self):
        base = Path(__file__).resolve().parent.parent / "features"
        with open(base / "drug_library.json") as f:
            self.drug_lib = json.load(f)
        with open(base / "virus_library.json") as f:
            self.virus_lib = json.load(f)

    def apply(self, drug_name, drug_dose, virus_name, subsystems):
        deltas = {}
        drug = self.drug_lib.get(drug_name, {})
        virus = self.virus_lib.get(virus_name, {})
        dose = drug_dose

        if drug.get("class") == "antiviral":
            deltas["atp_boost"] = 0.1 * dose
            deltas["tx_suppress"] = 0.05 * dose
            deltas["viral_replication_factor"] = max(0, 1 - 0.8 * dose)
        elif drug.get("class") == "anti-inflammatory":
            deltas["aod_boost"] = 0.3 * dose
            deltas["cytokine_suppress"] = 0.4 * dose
            deltas["ifn_suppress"] = 0.1 * dose
        elif drug.get("class") == "antioxidant":
            deltas["aod_boost"] = 0.5 * dose
            deltas["ros_boost"] = -0.2 * dose
        elif drug.get("class") == "mito_toxin":
            deltas["mito_damage"] = 0.4 * dose
            deltas["atp_boost"] = -0.3 * dose
        elif drug.get("class") == "proteasome_inhibitor":
            deltas["proteasome_inhibit"] = 0.6 * dose
            deltas["trans_inhibit"] = 0.1 * dose

        rep = virus.get("replication_rate", 0.0)
        cpe = virus.get("cytopathic_effect", 0.0)
        deltas["ros_boost"] = deltas.get("ros_boost", 0) + 0.2 * rep
        deltas["dna_damage"] = 0.1 * rep
        deltas["mem_damage"] = 0.15 * cpe
        deltas["viral_load"] = rep
        deltas["ifn_suppress"] = deltas.get("ifn_suppress", 0) + 0.3 * virus.get("interferon_antagonism", 0)
        return deltas
