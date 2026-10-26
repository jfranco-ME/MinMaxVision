import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import load_config, resolve_path
from src.diagnostics import aggregate_sweep
from src.pinn import train_pinn_state


def main():
    cfg = load_config()
    results_dir = resolve_path(cfg, "results")
    solutions_dir = resolve_path(cfg, "solutions")
    models_dir = resolve_path(cfg, "models")
    results_dir.mkdir(parents=True, exist_ok=True)
    solutions_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    net = cfg["network"]
    p = cfg["pinn"]
    rows = []
    for n in cfg["problem"]["states"]:
        for seed in cfg["reproducibility"]["seeds"]:
            summary, _, _ = train_pinn_state(
                n, seed,
                domain_length=cfg["problem"]["domain_length"],
                width=net["width"], depth=net["depth"], activation=net["activation"],
                torch_num_threads=cfg["reproducibility"]["torch_num_threads"],
                output_dir=solutions_dir, model_dir=models_dir,
                **p,
            )
            rows.append(summary)
            pd.DataFrame(rows).to_csv(results_dir / "p3_full_sweep_all_runs.csv", index=False)
            print(summary)
    raw = pd.DataFrame(rows)
    aggregate_sweep(raw).to_csv(results_dir / "p3_full_sweep_aggregate.csv", index=False)


if __name__ == "__main__":
    main()
