# Cell Digital Twin — Phase 3 v8 (GPU-Only, Fully Fixed)

## Quick Start
```cmd
conda activate thesis
cd "Phase 3_v8"
python -m src.digital_twin.validators
python generate_realistic_datasets.py --scale 10k
python generate_realistic_datasets.py --scale 50k
python generate_realistic_datasets.py --scale 200k
```

## What is fixed in v8
- GPU-only vectorized simulation (no CPU loops)
- 32 state variables (Defense includes viral_load)
- All perturbation keys initialized to 0.0 (no KeyError)
- Pathlib-based JSON loading (no FileNotFoundError)
- Decline classifier with chronic stress scenarios
- Real drug/virus entities with literature parameters
- 30/30/20/20 stratified distribution
- OpenMP warning suppressed
- Chunked progress reporting every 1024 trajectories
