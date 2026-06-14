"""
Map subsystem states to mechanism subclasses.
"""
from .subsystem_classifier import classify_subsystems

def derive_mechanism(overall_class, subsystem_states, trajectory):
    """Derive mechanism from overall class and subsystem states."""
    final = trajectory[-1]

    # Find weakest subsystem
    health_scores = {
        "genetic": 0.5 * final[0] + 0.3 * final[1] + 0.2 * final[2],
        "energy": 0.3 * final[4] + 0.2 * final[5] + 0.2 * final[6] + 0.2 * final[8] + 0.1 * final[7],
        "protein": 0.4 * final[9] + 0.3 * (1 - final[10]) + 0.3 * (1 - final[11]),
        "signaling": 0.3 * final[13] + 0.3 * (1 - final[14]) + 0.2 * final[15] + 0.2 * (1 - final[16]),
        "membrane": 0.3 * final[17] + 0.2 * final[18] + 0.2 * (1 - final[19]) + 0.2 * final[20] + 0.1 * final[21],
        "defense": 0.3 * (1 - final[22]) + 0.3 * final[23] + 0.2 * final[24] + 0.2 * (1 - final[25]),
        "waste": 0.3 * final[28] + 0.3 * final[29] + 0.2 * (1 - final[30]) + 0.2 * (1 - final[31])
    }
    weakest = min(health_scores, key=health_scores.get)

    mechanism_map = {
        "CVR": "CVR_antiviral_suppression",
        "BRT": f"BRT_{weakest}_stress",
        "PVR": f"PVR_{weakest}_stress",
        "VBR": "VBR_resistance_emergence",
        "TRF": f"TRF_{weakest}_toxicity",
        "SMR": "SMR_innate_immunity"
    }

    return mechanism_map.get(overall_class, "unknown")
