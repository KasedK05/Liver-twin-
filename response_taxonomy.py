"""
6-Class Response Taxonomy for Drug Response Analysis
Literature basis: EASL 2017, AASLD 2016, FDA DILI Guidance
"""

CLASS_NAMES = ["CVR", "BRT", "PVR", "VBR", "TRF", "SMR"]
CLASS_DESCRIPTIONS = {
    "CVR": "Complete Virological Response — Full viral suppression, all subsystems homeostatic",
    "BRT": "Biochemical Response with Toxicity — Viral suppressed but drug-induced cell damage",
    "PVR": "Partial Virological Response — Reduced but persistent viral load, chronic stress",
    "VBR": "Virological Breakthrough — Initial suppression followed by resistance/rebound",
    "TRF": "Treatment-Related Failure — Death from drug toxicity (not viral progression)",
    "SMR": "Spontaneous/Immune Recovery — Natural viral clearance without drug treatment"
}

MECHANISM_TAXONOMY = {
    "CVR": ["CVR_antiviral_suppression", "CVR_minor_stress"],
    "BRT": ["BRT_mitochondrial_stress", "BRT_ER_stress", "BRT_oxidative_stress", "BRT_mixed_toxicity"],
    "PVR": ["PVR_immune_exhaustion", "PVR_metabolic_strain", "PVR_genotoxic_stress", 
            "PVR_chronic_multi_stress", "PVR_stable_persistence", "PVR_unclassified"],
    "VBR": ["VBR_resistance_emergence"],
    "TRF": ["TRF_mitochondrial_toxicity", "TRF_proteasome_toxicity", "TRF_membrane_lysis",
            "TRF_multi_system_toxicity", "TRF_viral_progression"],
    "SMR": ["SMR_innate_immunity"]
}

def get_class_description(class_name):
    return CLASS_DESCRIPTIONS.get(class_name, "Unknown")

def get_mechanisms_for_class(class_name):
    return MECHANISM_TAXONOMY.get(class_name, [])
