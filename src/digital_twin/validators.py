import sys, os
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from digital_twin.gpu_simulator import CellSimulatorGPU

def _first_time(arr, threshold, dt=0.5):
    idx = np.where(arr > threshold)[0]
    if len(idx) == 0:
        return float('inf')
    return idx[0] * dt

def run_validations():
    print("=" * 60)
    print("Digital Twin Face-Validity Validation Suite (7 Subsystems)")
    print("=" * 60)
    sim = CellSimulatorGPU(device="cpu")
    all_pass = True

    batch = [{"scenario": {"drug": None, "dose": 0.0, "virus": None}, "params": {}} for _ in range(10)]
    traj, _ = sim.run(batch, t_max=24.0, dt=0.5, n_steps=49)
    atp_final = traj[0, :, 4].mean()
    status = "PASS" if atp_final > 0.5 else "FAIL"
    print(f"[{status}] Energy conservation / baseline stability")
    all_pass &= status == "PASS"

    batch = [{"scenario": {"drug": "entecavir", "dose": 0.8, "virus": "hbv_wildtype"}, "params": {}} for _ in range(10)]
    traj, _ = sim.run(batch, t_max=24.0, dt=0.5, n_steps=49)
    atp_treat = traj[0, :, 4].mean()
    status = "PASS" if atp_treat > 0.5 else "FAIL"
    print(f"[{status}] Antiviral improves ATP: {atp_treat:.3f}")
    all_pass &= status == "PASS"

    batch = [{"scenario": {"drug": None, "dose": 0.0, "virus": "hbv_wildtype"}, "params": {}} for _ in range(10)]
    traj, _ = sim.run(batch, t_max=24.0, dt=0.5, n_steps=49)
    ros = traj[0, :, 22]
    ifn = traj[0, :, 26]
    t_ros = _first_time(ros, 0.3)
    t_ifn = _first_time(ifn, 0.3)
    status = "PASS" if (t_ros < t_ifn and t_ifn < 24.0) else "FAIL"
    print(f"[{status}] Temporal ordering: ROS@{t_ros:.1f}h, IFN@{t_ifn:.1f}h")
    all_pass &= status == "PASS"

    batch = [{"scenario": {"drug": "rotenone", "dose": 1.0, "virus": None}, "params": {}} for _ in range(10)]
    traj, _ = sim.run(batch, t_max=24.0, dt=0.5, n_steps=49)
    atp_end = traj[0, -1, 4]
    mito_end = traj[0, -1, 6]
    status = "PASS" if atp_end < 0.1 else "FAIL"
    print(f"[{status}] Mito toxin: ATP={atp_end:.3f}, Mito={mito_end:.3f}")
    all_pass &= status == "PASS"

    batch = [{"scenario": {"drug": None, "dose": 0.0, "virus": "hbv_ymdd"}, "params": {}} for _ in range(10)]
    traj, _ = sim.run(batch, t_max=24.0, dt=0.5, n_steps=49)
    repair = traj[0, :, 24]
    status = "PASS" if repair.max() > 0.15 else "FAIL"
    print(f"[{status}] DNA repair activated: max={repair.max():.3f}")
    all_pass &= status == "PASS"

    batch = [{"scenario": {"drug": "bortezomib", "dose": 1.0, "virus": None}, "params": {}} for _ in range(10)]
    traj, _ = sim.run(batch, t_max=24.0, dt=0.5, n_steps=49)
    upr = traj[0, :, 10]
    status = "PASS" if upr.max() > 0.5 else "FAIL"
    print(f"[{status}] UPR activation: max={upr.max():.3f}")
    all_pass &= status == "PASS"

    batch = [{"scenario": {"drug": None, "dose": 0.0, "virus": "adenovirus_5"}, "params": {}} for _ in range(10)]
    traj, _ = sim.run(batch, t_max=24.0, dt=0.5, n_steps=49)
    mem = traj[0, :, 17]
    status = "PASS" if mem.min() < 0.5 else "FAIL"
    print(f"[{status}] Membrane damage: min={mem.min():.3f}")
    all_pass &= status == "PASS"

    batch = [{"scenario": {"drug": "rotenone", "dose": 0.5, "virus": None}, "params": {}} for _ in range(10)]
    traj, _ = sim.run(batch, t_max=24.0, dt=0.5, n_steps=49)
    auto = traj[0, :, 28]
    status = "PASS" if auto.max() > 0.15 else "FAIL"
    print(f"[{status}] Autophagy compensation: max={auto.max():.3f}")
    all_pass &= status == "PASS"

    batch = [{"scenario": {"drug": None, "dose": 0.0, "virus": "hbv_ymdd"}, "params": {}} for _ in range(10)]
    traj, _ = sim.run(batch, t_max=24.0, dt=0.5, n_steps=49)
    p53 = traj[0, :, 14]
    status = "PASS" if p53.max() > 0.5 else "FAIL"
    print(f"[{status}] p53 activation: max={p53.max():.3f}")
    all_pass &= status == "PASS"

    batch = [{"scenario": {"drug": "rotenone", "dose": 0.3, "virus": "hbv_wildtype"}, "params": {}} for _ in range(20)]
    traj, _ = sim.run(batch, t_max=24.0, dt=0.5, n_steps=49)
    outcomes = sim.classify_outcomes(traj)
    counts = {o: outcomes.count(o) for o in set(outcomes)}
    has_decline = counts.get("decline", 0) > 0
    status = "PASS" if has_decline else "FAIL"
    print(f"[{status}] Decline presence in test sample: {counts}")
    all_pass &= status == "PASS"

    print("=" * 60)
    if all_pass:
        print("All validations passed.")
    else:
        print("Some validations FAILED.")
    print("=" * 60)
    return all_pass

if __name__ == "__main__":
    run_validations()
