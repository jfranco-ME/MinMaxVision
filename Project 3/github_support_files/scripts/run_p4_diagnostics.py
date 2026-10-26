import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import load_config, resolve_path
from src.diagnostics import analyze_solution_dataframe


def main():
    cfg = load_config()
    sol = resolve_path(cfg, "solutions")
    res = resolve_path(cfg, "results")
    res.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    coeffs = []

    for case in cfg["p4"]["representative_cases"]:
        n, seed, label = case["n"], case["seed"], case["label"]
        path = sol / f"pinn_n{n:02d}_seed{seed}.csv"
        if not path.exists():
            print(f"Skipping missing solution file: {path}")
            continue
        df = pd.read_csv(path)
        d = analyze_solution_dataframe(df, n, cfg["problem"]["domain_length"], cfg["p4"]["max_sine_mode"])
        summary_rows.append({
            "case": f"n={n}, seed={seed} ({label})",
            "n": n,
            "seed": seed,
            "relative_L2_percent": d["relative_L2_percent"],
            "max_abs_error": d["max_abs_error"],
            "x_of_max_error": d["x_of_max_error"],
        })
        c = d["mode_coefficients"].copy()
        c["case"] = f"n={n}, seed={seed} ({label})"
        coeffs.append(c)

    pd.DataFrame(summary_rows).to_csv(res / "p4_error_spectrum_summary_regenerated.csv", index=False)
    if coeffs:
        pd.concat(coeffs, ignore_index=True).to_csv(res / "p4_sine_mode_coefficients_regenerated.csv", index=False)
    print("P4 diagnostics complete.")


if __name__ == "__main__":
    main()
