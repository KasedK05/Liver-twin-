"""
Per-subsystem state classification and mechanism derivation.
"""
import numpy as np

SUBSYSTEM_STATES = {
    "genetic": ["intact", "damaged", "critical", "arrested"],
    "energy": ["functional", "stressed", "collapsed"],
    "protein": ["homeostatic", "UPR_active", "aggregated", "secretory_stressed"],
    "signaling": ["balanced", "p53_active", "inflammatory", "growth_arrested"],
    "membrane": ["stable", "permeabilized", "lysed", "pump_failure"],
    "defense": ["controlled", "inflamed", "exhausted", "viral_suppressed"],
    "waste": ["clearing", "accumulating", "blocked"]
}

def classify_subsystems(final_state):
    """Classify each subsystem based on final state values."""
    states = {}

    # Genetic: 0-3
    dna = final_state[0]
    tx = final_state[1]
    chrom = final_state[2]
    cycle = final_state[3]
    if dna < 0.15 or (dna < 0.25 and cycle < 0.05):
        states["genetic"] = "critical"
    elif cycle < 0.05:
        states["genetic"] = "arrested"
    elif dna < 0.45 or chrom > 0.6 or tx < 0.4:
        states["genetic"] = "damaged"
    else:
        states["genetic"] = "intact"

    # Energy: 4-8
    atp = final_state[4]
    mito = final_state[6]
    if atp < 0.15 or mito < 0.15:
        states["energy"] = "collapsed"
    elif atp < 0.45 or mito < 0.4:
        states["energy"] = "stressed"
    else:
        states["energy"] = "functional"

    # Protein: 9-12
    upr = final_state[10]
    agg = final_state[11]
    if agg > 0.6 or (upr > 0.7 and final_state[9] < 0.3):
        states["protein"] = "aggregated"
    elif upr > 0.4:
        states["protein"] = "UPR_active"
    elif final_state[12] > 0.7 and final_state[9] > 0.6:
        states["protein"] = "secretory_stressed"
    else:
        states["protein"] = "homeostatic"

    # Signaling: 13-16
    p53 = final_state[14]
    cyto = final_state[16]
    if p53 > 0.6 and dna < 0.5:
        states["signaling"] = "p53_active"
    elif cyto > 0.6 and final_state[25] > 0.5:
        states["signaling"] = "inflammatory"
    elif final_state[15] < 0.2 and final_state[13] < 0.3 and cycle < 0.1:
        states["signaling"] = "growth_arrested"
    else:
        states["signaling"] = "balanced"

    # Membrane: 17-21
    mem = final_state[17]
    pump = final_state[18]
    ca = final_state[19]
    if mem < 0.15 or (ca > 0.8 and pump < 0.3):
        states["membrane"] = "lysed"
    elif mem < 0.4 or pump < 0.3 or final_state[21] < 0.2:
        states["membrane"] = "permeabilized"
    elif pump < 0.5 and ca > 0.6:
        states["membrane"] = "pump_failure"
    else:
        states["membrane"] = "stable"

    # Defense: 22-27
    ros = final_state[22]
    aod = final_state[23]
    inflam = final_state[25]
    ifn = final_state[26]
    viral = final_state[27]
    if viral > 0.1 and ifn > 0.5 and ros < 0.5:
        states["defense"] = "viral_suppressed"
    elif inflam > 0.6 and ros > 0.6 and aod < 0.3:
        states["defense"] = "exhausted"
    elif inflam > 0.5 or (ros > 0.5 and aod < 0.4):
        states["defense"] = "inflamed"
    else:
        states["defense"] = "controlled"

    # Waste: 28-31
    auto = final_state[28]
    ph = final_state[29]
    waste = final_state[30]
    lipo = final_state[31]
    if auto < 0.2 and waste > 0.6:
        states["waste"] = "blocked"
    elif waste > 0.5 or lipo > 0.4 or (ph < 0.5 and auto < 0.4):
        states["waste"] = "accumulating"
    else:
        states["waste"] = "clearing"

    return states
