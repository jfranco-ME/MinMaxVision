# GitHub support files — upload guide

This folder contains the project support files **excluding the main report notebook**.

## Recommended repository layout

```text
Project3/
├── Project3_Report.ipynb          # add later; not included here
├── README.md                      # your final project/repository overview
├── requirements.txt
├── .gitignore
├── config/
│   └── experiment.json
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── physics.py
│   ├── models.py
│   ├── finite_difference.py
│   ├── pinn.py
│   ├── supervised.py
│   └── diagnostics.py
├── scripts/
│   ├── run_p1_baseline.py
│   ├── run_p3_sweep.py
│   ├── run_controlled_diagnosis.py
│   ├── run_p4_diagnostics.py
│   ├── run_p5_cost.py
│   └── run_all.py
├── data/
│   ├── results/                   # saved CSV results used by the report
│   └── solutions/                 # generated wavefunction CSVs after rerunning
├── models/                        # optional generated .pt checkpoints
└── figures/                       # final saved figures
```

## What to upload to GitHub now

Upload everything in this support package while preserving the folder structure.

### `config/experiment.json`
This is the main place to change experiment settings. Change states, seeds, network width/depth, training epochs, grid sizes, etc. without editing the model implementation.

### `src/`
Reusable implementation. The report notebook should import functions from here instead of containing duplicate training code.

### `scripts/`
Standalone commands to regenerate results. Run them from the repository root, for example:

```bash
python scripts/run_p1_baseline.py
python scripts/run_p3_sweep.py
python scripts/run_controlled_diagnosis.py
python scripts/run_p4_diagnostics.py
python scripts/run_p5_cost.py
```

Or regenerate everything:

```bash
python scripts/run_all.py
```

The full PINN sweep can take several minutes.

### `data/results/`
These are the results already produced for the submitted experiment. Keep them in GitHub so the report renders immediately without forcing every reader to retrain the PINNs.

### `data/solutions/`
Initially nearly empty. New runs save detailed wavefunction CSVs here. These files let P4 be regenerated without retraining.

### `models/`
Optional PyTorch checkpoints. `.pt` files are ignored by default because they can become large. Remove that rule from `.gitignore` if your instructor specifically wants trained checkpoints in the repository.

### `figures/`
Current final figures. The notebook can display these immediately; regenerated experiments can overwrite or create updated figures.

## Installation

From the repository root:

```bash
python -m pip install -r requirements.txt
```

## How the data flow should work

```text
config/experiment.json
        ↓
src/ + scripts/
        ↓
data/solutions/ + data/results/
        ↓
figures/
        ↓
Project3_Report.ipynb
```

The notebook should **read CSV files from `data/results/` and `data/solutions/`**, rather than embedding large result tables manually.

## Recommended GitHub upload method

1. Create/open the project repository.
2. Use **Add file → Upload files**.
3. Drag the *contents* of this support folder into the repository while preserving directories.
4. Commit with a message such as `Add reproducible experiment code and results`.
5. Add the final report notebook at the repository root afterward.

For a cleaner workflow with Git installed locally, copy this whole folder into your cloned repository, then run:

```bash
git add .
git commit -m "Add reproducible experiment code and results"
git push
```
