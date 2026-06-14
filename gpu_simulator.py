import torch
import numpy as np
from .perturbation import Perturbation

class CellSimulatorGPU:
    def __init__(self, device="cuda", biological_noise=0.005):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.perturbation = Perturbation()
        self.biological_noise = biological_noise

    def _init_state(self, batch_size, patient_profile="healthy"):
        s = torch.zeros((batch_size, 32), device=self.device)
        base = {
            0: (0.95, 0.02), 1: (0.70, 0.03), 2: (0.80, 0.02), 3: (0.50, 0.05),
            4: (0.90, 0.03), 5: (0.85, 0.03), 6: (0.88, 0.02), 7: (0.30, 0.05),
            8: (0.95, 0.02), 9: (0.60, 0.03), 10: (0.05, 0.01), 11: (0.05, 0.01),
            12: (0.40, 0.03), 13: (0.50, 0.05), 14: (0.10, 0.02), 15: (0.40, 0.03),
            16: (0.10, 0.02), 17: (0.90, 0.02), 18: (0.85, 0.03), 19: (0.10, 0.02),
            20: (0.70, 0.03), 21: (0.60, 0.03), 22: (0.10, 0.02), 23: (0.60, 0.03),
            24: (0.20, 0.03), 25: (0.05, 0.01), 26: (0.05, 0.01), 27: (0.0, 0.0),
            28: (0.20, 0.03), 29: (0.70, 0.03), 30: (0.10, 0.02), 31: (0.05, 0.01)
        }
        adjustments = {
            "healthy": {},
            "compromised": {0: -0.15, 4: -0.10, 6: -0.10, 17: -0.10, 23: -0.15},
            "elderly": {0: -0.10, 4: -0.15, 6: -0.15, 22: +0.10, 24: -0.10},
            "stressed": {4: -0.20, 10: +0.20, 22: +0.15, 25: +0.10}
        }
        adj = adjustments.get(patient_profile, {})
        for idx, (mean, std) in base.items():
            val = mean + adj.get(idx, 0)
            s[:, idx] = torch.normal(val, std, (batch_size,), device=self.device)
        s[:, 27] = 0.0
        return torch.clamp(s, 0, 1)

    def _compute_coupling(self, s):
        c = {}
        c["atp"] = s[:, 4]
        c["atp_demand"] = 0.03 + 0.01 * s[:, 9] + 0.01 * s[:, 20]
        c["atp_supply_ratio"] = s[:, 4] / torch.clamp(0.03 + 0.01 * s[:, 9], min=0.01)
        c["ros"] = s[:, 22]
        c["oxidative_stress_index"] = torch.clamp(s[:, 22] - s[:, 23], min=0)
        c["inflammatory_tone"] = s[:, 25] + 0.5 * s[:, 16]
        c["viral_load"] = s[:, 27]
        c["dna_damage_signal"] = 1.0 - s[:, 0]
        c["dna_repair_activity"] = s[:, 24]
        c["autophagy_flux"] = s[:, 28]
        c["metabolic_load"] = 0.5 + 0.3 * s[:, 7]
        c["mtor"] = s[:, 13]
        c["ampk"] = 1.0 - s[:, 13]
        c["inflammasome_activity"] = s[:, 25]
        c["aggregates"] = s[:, 11]
        return c

    def _apply_perturbation(self, drug, dose, virus, batch_size):
        d = {}
        all_keys = ["atp_boost", "tx_suppress", "viral_replication_factor", "aod_boost",
                    "cytokine_suppress", "ifn_suppress", "ros_boost", "mito_damage",
                    "proteasome_inhibit", "trans_inhibit", "dna_damage", "mem_damage",
                    "viral_load", "ca_influx", "autophagy_block", "mtor_inhibit", "growth_factor"]
        for k in all_keys:
            d[k] = 0.0
        d["viral_replication_factor"] = 1.0

        drug_info = self.perturbation.drug_lib.get(drug, {})
        virus_info = self.perturbation.virus_lib.get(virus, {})

        if drug_info.get("class") == "antiviral":
            d["atp_boost"] = 0.1 * dose
            d["tx_suppress"] = 0.05 * dose
            d["viral_replication_factor"] = max(0.0, 1.0 - 0.8 * dose)
        elif drug_info.get("class") == "anti-inflammatory":
            d["aod_boost"] = 0.3 * dose
            d["cytokine_suppress"] = 0.4 * dose
            d["ifn_suppress"] = 0.1 * dose
        elif drug_info.get("class") == "antioxidant":
            d["aod_boost"] = 0.5 * dose
            d["ros_boost"] = -0.2 * dose
        elif drug_info.get("class") == "mito_toxin":
            d["mito_damage"] = 0.4 * dose
            d["atp_boost"] = -0.3 * dose
        elif drug_info.get("class") == "proteasome_inhibitor":
            d["proteasome_inhibit"] = 0.6 * dose
            d["trans_inhibit"] = 0.1 * dose

        rep = virus_info.get("replication_rate", 0.0)
        cpe = virus_info.get("cytopathic_effect", 0.0)
        d["ros_boost"] = d["ros_boost"] + 0.2 * rep
        d["dna_damage"] = 0.1 * rep
        d["mem_damage"] = 0.15 * cpe
        d["viral_load"] = rep
        d["ifn_suppress"] = d["ifn_suppress"] + 0.3 * virus_info.get("interferon_antagonism", 0)

        for k in d:
            d[k] = torch.full((batch_size,), d[k], device=self.device, dtype=torch.float32)
        return d

    def _update_subsystems(self, s, dt, c, d):
        damage = 0.05 * c["ros"] + 0.03 * c["viral_load"] + d["dna_damage"]
        repair = 0.10 * c["dna_repair_activity"]
        s[:, 0] += dt * (repair - damage)
        tx_drive = 0.5 * s[:, 0] + 0.3 * c["atp"]
        s[:, 1] += dt * (0.05 * (tx_drive - s[:, 1]) - d["tx_suppress"])
        chrom_stress = 0.03 * c["ros"]
        chrom_relax = 0.02 * (1 - s[:, 2])
        s[:, 2] += dt * (chrom_relax - chrom_stress)
        cycle_speed = 0.02 * s[:, 1] * c["atp"]
        arrest = (s[:, 0] < 0.3).float()
        s[:, 3] += dt * (cycle_speed * (1 - arrest))
        s[:, 3] = s[:, 3] % 1.0

        atp_prod = 0.05 * s[:, 6] * s[:, 8] * (1 + d["atp_boost"])
        atp_demand = c["atp_demand"]
        atp_decay = 0.02 * s[:, 4]
        s[:, 4] += dt * (atp_prod - atp_demand - atp_decay)
        nad_shift = 0.01 * (s[:, 7] - 0.3) - 0.02 * c["ros"]
        s[:, 5] += dt * nad_shift
        mito_damage = d["mito_damage"] + 0.1 * c["ros"]
        mito_recovery = 0.03 * (1 - s[:, 6]) * s[:, 4]
        s[:, 6] += dt * (mito_recovery - mito_damage)
        glyco_drive = (s[:, 6] < 0.4).float() * 0.5
        s[:, 7] += dt * (0.02 * glyco_drive - 0.01 * (s[:, 7] - 0.3))
        o2_consump = 0.02 * s[:, 6] + 0.01 * s[:, 7]
        s[:, 8] += dt * (0.05 * (0.95 - s[:, 8]) - o2_consump)

        trans_capacity = 0.5 * c["atp"]
        s[:, 9] += dt * (0.05 * (trans_capacity - s[:, 9]) - d["trans_inhibit"])
        upr_trigger = ((s[:, 11] > 0.4) | (d["proteasome_inhibit"] > 0.5)).float()
        upr_resolve = 0.1 * s[:, 10] * c["atp"]
        s[:, 10] += dt * (0.3 * upr_trigger - upr_resolve)
        agg_form = 0.05 * (1 - s[:, 10]) * s[:, 9]
        agg_clear = 0.10 * c["autophagy_flux"]
        s[:, 11] += dt * (agg_form - agg_clear)
        sec_load = 0.03 * s[:, 9]
        sec_capacity = 0.02 * c["atp"]
        s[:, 12] += dt * (sec_load - sec_capacity)

        mtor_drive = 0.5 * c["atp"] + 0.3 * s[:, 15]
        mtor_inhibit = 0.2 * c["ros"] + d["mtor_inhibit"]
        s[:, 13] += dt * (0.05 * (mtor_drive - s[:, 13]) - mtor_inhibit)
        p53_trigger = ((c["dna_damage_signal"] > 0.5) | (c["ros"] > 0.6)).float()
        p53_decay = 0.08 * s[:, 14]
        s[:, 14] += dt * (0.4 * p53_trigger - p53_decay)
        gf_input = d["growth_factor"]
        gf_decay = 0.05 * s[:, 15]
        s[:, 15] += dt * (gf_input - gf_decay)
        cyto_drive = 0.3 * c["inflammasome_activity"] + 0.2 * c["viral_load"]
        s[:, 16] += dt * (0.05 * cyto_drive - 0.03 * s[:, 16] - d["cytokine_suppress"])

        mem_damage = 0.05 * c["ros"] + 0.03 * c["viral_load"] + d["mem_damage"]
        mem_repair = 0.04 * c["atp"]
        s[:, 17] += dt * (mem_repair - mem_damage)
        pump_capacity = 0.5 * c["atp"]
        leak = 0.03 * (1 - s[:, 17])
        s[:, 18] += dt * (0.05 * (pump_capacity - s[:, 18]) - leak)
        ca_influx = 0.1 * (1 - s[:, 18]) + d["ca_influx"]
        ca_efflux = 0.08 * c["atp"]
        s[:, 19] += dt * (ca_influx - ca_efflux)
        uptake_drive = 0.4 * s[:, 17]
        uptake_inhibit = 0.05 * s[:, 16]
        s[:, 20] += dt * (0.03 * (uptake_drive - s[:, 20]) - uptake_inhibit)
        ves_drive = 0.3 * c["atp"]
        ves_disrupt = 0.04 * c["aggregates"]
        s[:, 21] += dt * (0.03 * (ves_drive - s[:, 21]) - ves_disrupt)

        viral_load_effective = d["viral_load"] * d["viral_replication_factor"]
        s[:, 27] = viral_load_effective
        ros_gen = 0.05 * c["metabolic_load"] + d["ros_boost"]
        ros_scav = 0.10 * s[:, 23] * s[:, 22]
        s[:, 22] += dt * (ros_gen - ros_scav)
        aod_boost = d["aod_boost"]
        aod_exhaust = 0.05 * s[:, 22]
        s[:, 23] += dt * (0.02 * aod_boost - aod_exhaust)
        repair_trigger = (s[:, 22] > 0.4).float()
        repair_cost = 0.02 * s[:, 24]
        s[:, 24] += dt * (0.3 * repair_trigger - repair_cost)
        inflam_trigger = ((s[:, 22] > 0.5) & (s[:, 23] < 0.3)).float()
        s[:, 25] += dt * (0.4 * inflam_trigger - 0.1 * s[:, 25])
        ifn_trigger = s[:, 27] * (1 - d["ifn_suppress"])
        ifn_decay = 0.05 * s[:, 26]
        s[:, 26] += dt * (0.2 * ifn_trigger - ifn_decay)

        auto_drive = ((c["ampk"] > 0.5) | (c["mtor"] < 0.3)).float()
        auto_block = d["autophagy_block"]
        s[:, 28] += dt * (0.3 * auto_drive - 0.1 * s[:, 28] - auto_block)
        ph_shift = 0.02 * (1 - s[:, 28]) + 0.01 * c["ros"]
        ph_recover = 0.03 * (0.7 - s[:, 29])
        s[:, 29] += dt * (ph_recover - ph_shift)
        waste_prod = 0.03 * c["metabolic_load"]
        waste_clear = 0.08 * s[:, 28] * s[:, 29]
        s[:, 30] += dt * (waste_prod - waste_clear)
        lipo_form = 0.01 * s[:, 30] * (1 - s[:, 28])
        s[:, 31] += dt * lipo_form

        s = s + torch.normal(0, self.biological_noise, s.shape, device=self.device)
        return torch.clamp(s, 0, 1)

    def run(self, batch_info, t_max=24.0, dt=0.5, n_steps=49):
        n = len(batch_info)
        profiles = [info.get("patient_profile", "healthy") for info in batch_info]
        unique_profiles = list(set(profiles))

        if len(unique_profiles) == 1:
            s = self._init_state(n, unique_profiles[0])
        else:
            s = torch.zeros((n, 32), device=self.device)
            for i, profile in enumerate(profiles):
                s_i = self._init_state(1, profile)
                s[i] = s_i[0]

        trajectories = torch.zeros((n, n_steps, 32), device=self.device)
        first_scenario = batch_info[0]["scenario"]
        homogeneous = all(
            info["scenario"].get("drug") == first_scenario.get("drug") and
            info["scenario"].get("dose") == first_scenario.get("dose") and
            info["scenario"].get("virus") == first_scenario.get("virus")
            for info in batch_info
        )

        for step in range(n_steps):
            c = self._compute_coupling(s)
            t = step * dt

            if homogeneous:
                drug = first_scenario.get("drug")
                dose = first_scenario.get("dose", 0.0)
                virus = first_scenario.get("virus")
                schedule = first_scenario.get("drug_schedule", [])
                active_dose = dose
                for sched in schedule:
                    if len(sched) == 3:
                        sched_time, sched_drug, sched_dose = sched
                        if abs(t - sched_time) < dt:
                            active_dose = sched_dose
                            drug = sched_drug
                d = self._apply_perturbation(drug, active_dose, virus, n)
            else:
                d_list = []
                for info in batch_info:
                    scenario = info["scenario"]
                    drug = scenario.get("drug")
                    dose = scenario.get("dose", 0.0)
                    virus = scenario.get("virus")
                    schedule = scenario.get("drug_schedule", [])
                    for sched in schedule:
                        if len(sched) == 3:
                            sched_time, sched_drug, sched_dose = sched
                            if abs(t - sched_time) < dt:
                                dose = sched_dose
                                drug = sched_drug
                    d_single = self._apply_perturbation(drug, dose, virus, 1)
                    d_list.append(d_single)
                d = {}
                for k in d_list[0].keys():
                    d[k] = torch.stack([d_item[k] for d_item in d_list]).squeeze(-1)

            s = self._update_subsystems(s, dt, c, d)
            trajectories[:, step, :] = s

        return trajectories.cpu().numpy(), np.arange(n_steps) * dt

    def classify_outcomes(self, trajectories):
        n = trajectories.shape[0]
        outcomes = []
        for i in range(n):
            traj = trajectories[i]
            h = 0.25 * traj[:, 4] + 0.25 * traj[:, 0] + 0.25 * traj[:, 17] + 0.25 * (1 - traj[:, 22])
            min_h = h.min()
            final_h = h[-1]
            mean_h = h.mean()
            trend = np.polyfit(np.arange(len(h)), h, 1)[0]
            if min_h < 0.08 or final_h < 0.10:
                outcomes.append("death")
            elif final_h > 0.70 and mean_h > 0.55 and trend > 0.005:
                outcomes.append("recovery")
            elif 0.10 < min_h < 0.55 and 0.15 < final_h < 0.50 and trend < -0.003:
                outcomes.append("decline")
            else:
                outcomes.append("stable")
        return outcomes

    def classify_subsystems(self, trajectories, label_noise=0.0):
        n = trajectories.shape[0]
        outcomes = []

        for i in range(n):
            traj = trajectories[i]
            atp = traj[:, 4]
            viral = traj[:, 27]
            atp_f = atp[-1]
            atp_m = atp.min()
            vir_m = viral.max()
            vir_f = viral[-1]

            if atp_m < 0.15 or (atp_f < 0.2 and traj[-1, 6] < 0.3):
                base = "SMR"
            elif atp_f < 0.5 and (traj[:, 10].max() > 0.4 or traj[:, 14].max() > 0.4 or traj[:, 22].max() > 0.5):
                base = "TRF"
            elif vir_m > 0.4 and vir_f > 0.3:
                base = "VBR"
            elif vir_m > 0.2 and vir_f > 0.2 and atp_f > 0.3 and atp_f < 0.7:
                base = "PVR"
            elif vir_m < 0.3 and atp_f > 0.5:
                base = "CVR"
            else:
                base = "BRT"

            if label_noise > 0 and np.random.random() < label_noise:
                neighbors = {
                    "CVR": ["PVR", "BRT"],
                    "BRT": ["CVR", "SMR"],
                    "PVR": ["CVR", "VBR", "TRF"],
                    "VBR": ["PVR", "TRF"],
                    "TRF": ["PVR", "VBR", "SMR"],
                    "SMR": ["TRF", "BRT"]
                }
                base = np.random.choice(neighbors[base])

            outcomes.append(base)

        return outcomes
