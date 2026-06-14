import json
from pathlib import Path

class Perturbation:
    """Applies drug and virus perturbations to subsystem parameters."""

    def __init__(self, drug_lib_path=None, virus_lib_path=None):
        base = Path(__file__).resolve().parent.parent / "features"

        with open(base / "drug_library.json") as f:
            self.drug_lib = json.load(f)
        with open(base / "virus_library.json") as f:
            self.virus_lib = json.load(f)

    def apply(self, drug_name, drug_dose, virus_name):
        """Return perturbation deltas as dict."""
        deltas = {}
        drug = self.drug_lib.get(drug_name, {})
        virus = self.virus_lib.get(virus_name, {})

        # Initialize all keys to 0
        all_keys = ["atp_boost", "tx_suppress", "viral_replication_factor", "aod_boost",
                    "cytokine_suppress", "ifn_suppress", "ros_boost", "mito_damage",
                    "proteasome_inhibit", "trans_inhibit", "dna_damage", "mem_damage",
                    "viral_load", "ca_influx", "autophagy_block", "mtor_inhibit", "growth_factor"]
        for k in all_keys:
            deltas[k] = 0.0

        # Drug effects
        if drug.get("class") == "antiviral":
            deltas["atp_boost"] = 0.1 * drug_dose
            deltas["tx_suppress"] = 0.05 * drug_dose
            deltas["viral_replication_factor"] = max(0, 1 - 0.8 * drug_dose)
        elif drug.get("class") == "anti-inflammatory":
            deltas["aod_boost"] = 0.3 * drug_dose
            deltas["cytokine_suppress"] = 0.4 * drug_dose
            deltas["ifn_suppress"] = 0.1 * drug_dose
        elif drug.get("class") == "antioxidant":
            deltas["aod_boost"] = 0.5 * drug_dose
            deltas["ros_boost"] = -0.2 * drug_dose
        elif drug.get("class") == "mito_toxin":
            deltas["mito_damage"] = 0.4 * drug_dose
            deltas["atp_boost"] = -0.3 * drug_dose
        elif drug.get("class") == "proteasome_inhibitor":
            deltas["proteasome_inhibit"] = 0.6 * drug_dose
            deltas["trans_inhibit"] = 0.1 * drug_dose

        # Virus effects
        rep = virus.get("replication_rate", 0.0)
        cpe = virus.get("cytopathic_effect", 0.0)
        deltas["ros_boost"] += 0.2 * rep
        deltas["dna_damage"] = 0.1 * rep
        deltas["mem_damage"] = 0.15 * cpe
        deltas["viral_load"] = rep
        deltas["ifn_suppress"] += 0.3 * virus.get("interferon_antagonism", 0)

        return deltas
