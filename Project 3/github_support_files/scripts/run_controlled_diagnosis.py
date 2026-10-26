import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import load_config, resolve_path
from src.supervised import train_supervised_state


def run_block(cfg, block_name, filename):
    net = cfg["network"]
    settings = cfg[block_name]
    rows = []
    out = resolve_path(cfg, "solutions")
    for n in cfg["controlled_states"]:
        for seed in cfg["reproducibility"]["seeds"]:
            summary, _ = train_supervised_state(
                n, seed,
                domain_length=cfg["problem"]["domain_length"],
                width=net["width"], depth=net["depth"], activation=net["activation"],
                torch_num_threads=cfg["reproducibility"]["torch_num_threads"],
                output_dir=out,
                **settings,
            )
            rows.append(summary)
            print(block_name, summary)
    pd.DataFrame(rows).to_csv(resolve_path(cfg, "results") / filename, index=False)


def main():
    cfg = load_config()
    resolve_path(cfg, "results").mkdir(parents=True, exist_ok=True)
    resolve_path(cfg, "solutions").mkdir(parents=True, exist_ok=True)
    run_block(cfg, "supervised_equal_budget", "supervised_equal_budget_all_runs.csv")
    run_block(cfg, "supervised_capacity", "supervised_capacity_all_runs.csv")


if __name__ == "__main__":
    main()
