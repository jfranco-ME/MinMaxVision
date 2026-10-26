import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import platform
import time

import numpy as np
import pandas as pd
import torch

from src.config import load_config, resolve_path
from src.finite_difference import benchmark_fd
from src.models import TrialNet
from src.pinn import normalized_projected


def benchmark_inference(model, n, L, n_points=1002, repeats=200):
    x = torch.linspace(0.0, L, n_points).reshape(-1, 1)
    model.eval()
    with torch.no_grad():
        for _ in range(20):
            normalized_projected(model, n, x, domain_length=L)
        vals = []
        for _ in range(repeats):
            t0 = time.perf_counter_ns()
            normalized_projected(model, n, x, domain_length=L)
            vals.append((time.perf_counter_ns() - t0) / 1e6)
    return float(np.median(vals))


def main():
    cfg = load_config()
    res = resolve_path(cfg, "results")
    raw = pd.read_csv(res / "p3_full_sweep_all_runs.csv")
    by_n = raw.groupby("target_n")["training_time_s"].agg(["median", "mean", "min", "max"]).reset_index()
    by_n.columns = ["target_n", "median_training_s", "mean_training_s", "min_training_s", "max_training_s"]
    by_n.to_csv(res / "p5_training_time_by_state_regenerated.csv", index=False)

    fd_cfg = cfg["finite_difference"]
    L = cfg["problem"]["domain_length"]
    fd_ms = benchmark_fd(fd_cfg["final_n_interior"], fd_cfg["n_states"], L, repeats=100)

    net = cfg["network"]
    infer = []
    for n in [1, 4, 8, 10]:
        model = TrialNet(net["width"], net["depth"], net["activation"], L)
        infer.append({"target_n": n, "median_ms": benchmark_inference(model, n, L)})
    pd.DataFrame(infer).to_csv(res / "p5_pinn_inference_benchmark_regenerated.csv", index=False)

    hw = pd.DataFrame([{
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "logical_cpus_visible": os.cpu_count(),
        "fd_median_ms_first_states": fd_ms,
    }])
    hw.to_csv(res / "p5_hardware_regenerated.csv", index=False)
    print(hw.to_string(index=False))


if __name__ == "__main__":
    main()
