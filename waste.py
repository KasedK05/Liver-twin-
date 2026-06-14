import numpy as np
from .base import Subsystem

class WasteRecycling(Subsystem):
    def initialize_state(self, params):
        return {
            "autophagy": params.get("auto_0", 0.20),
            "ph": params.get("ph_0", 0.70),
            "waste": params.get("waste_0", 0.10),
            "lipofuscin": params.get("lipo_0", 0.05),
        }

    def update(self, dt, c, d):
        s = self.state
        auto_drive = 1.0 if (c.get("ampk", 0.5) > 0.5 or c.get("mtor", 0.5) < 0.3) else 0.0
        auto_block = d.get("autophagy_block", 0.0)
        s["autophagy"] += dt * (0.3 * auto_drive - 0.1 * s["autophagy"] - auto_block)
        s["autophagy"] = np.clip(s["autophagy"], 0.0, 1.0)

        ph_shift = 0.02 * (1 - s["autophagy"]) + 0.01 * c.get("ros", 0)
        ph_recover = 0.03 * (0.7 - s["ph"])
        s["ph"] += dt * (ph_recover - ph_shift)
        s["ph"] = np.clip(s["ph"], 0.0, 1.0)

        waste_prod = 0.03 * c.get("metabolic_load", 0.5)
        waste_clear = 0.08 * s["autophagy"] * s["ph"]
        s["waste"] += dt * (waste_prod - waste_clear)
        s["waste"] = np.clip(s["waste"], 0.0, 1.0)

        lipo_form = 0.01 * s["waste"] * (1 - s["autophagy"])
        s["lipofuscin"] += dt * lipo_form
        s["lipofuscin"] = np.clip(s["lipofuscin"], 0.0, 1.0)

    def get_health_score(self):
        s = self.state
        return 0.3 * s["autophagy"] + 0.3 * s["ph"] + 0.2 * (1 - s["waste"]) + 0.2 * (1 - s["lipofuscin"])
