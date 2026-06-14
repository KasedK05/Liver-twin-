import numpy as np

class CouplingEngine:
    """Computes cross-subsystem coupling variables from batched state."""

    def compute(self, state_matrix):
        """
        state_matrix: (batch, 32) — current state of all cells
        Returns: dict of coupling variables, each (batch,)
        """
        c = {}
        c["atp"] = state_matrix[:, 4]
        c["atp_demand"] = 0.03 + 0.01 * state_matrix[:, 9] + 0.01 * state_matrix[:, 20]
        c["atp_supply_ratio"] = c["atp"] / np.clip(c["atp_demand"], 0.01, 1.0)
        c["ros"] = state_matrix[:, 22]
        c["oxidative_stress_index"] = np.clip(state_matrix[:, 22] - state_matrix[:, 23], 0, 1)
        c["inflammatory_tone"] = state_matrix[:, 25] + 0.5 * state_matrix[:, 16]
        c["viral_load"] = state_matrix[:, 27]
        c["dna_damage_signal"] = 1.0 - state_matrix[:, 0]
        c["dna_repair_activity"] = state_matrix[:, 24]
        c["autophagy_flux"] = state_matrix[:, 28]
        c["metabolic_load"] = 0.5 + 0.3 * state_matrix[:, 7]
        c["mtor"] = state_matrix[:, 13]
        c["ampk"] = 1.0 - state_matrix[:, 13]
        c["inflammasome_activity"] = state_matrix[:, 25]
        c["aggregates"] = state_matrix[:, 11]
        c["cytokine"] = state_matrix[:, 16]
        return c
