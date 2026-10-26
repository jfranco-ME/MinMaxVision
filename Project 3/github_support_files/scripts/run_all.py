"""Convenience launcher. Full PINN runs can take several minutes."""
import subprocess
import sys

STAGES = [
    "scripts/run_p1_baseline.py",
    "scripts/run_p3_sweep.py",
    "scripts/run_controlled_diagnosis.py",
    "scripts/run_p4_diagnostics.py",
    "scripts/run_p5_cost.py",
]

for stage in STAGES:
    print(f"\n=== Running {stage} ===")
    subprocess.run([sys.executable, stage], check=True)
