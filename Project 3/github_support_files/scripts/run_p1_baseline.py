import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt

from src.config import load_config, resolve_path
from src.finite_difference import convergence_table, solve_fd
from src.physics import exact_wavefunction_np


def main():
    cfg = load_config()
    fd = cfg["finite_difference"]
    L = cfg["problem"]["domain_length"]
    results = resolve_path(cfg, "results")
    figures = resolve_path(cfg, "figures")
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    table = convergence_table(fd["grid_sizes"], fd["n_states"], L, fd["timing_repeats"])
    table.to_csv(results / "p1_convergence_all_states.csv", index=False)
    final = table[table.N_interior == fd["final_n_interior"]].copy()
    final["E_rel_error_percent"] = 100 * final["E_rel_error"]
    final["psi_rel_L2_error_percent"] = 100 * final["psi_rel_L2_error"]
    final.to_csv(results / "p1_final_grid_summary.csv", index=False)
    table.groupby("N_interior", as_index=False)["runtime_ms_first_states"].first().to_csv(results / "p1_timing.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    for n in [1, 5, 10]:
        d = table[table.state_n == n].sort_values("h")
        ax.loglog(d.h, d.E_rel_error, marker="o", label=f"n={n}")
    ax.set(xlabel="Grid spacing h", ylabel="Relative eigenvalue error", title="P1 baseline convergence")
    ax.legend(); ax.grid(True, which="both", alpha=.3); fig.tight_layout()
    fig.savefig(figures / "p1_energy_convergence.png", dpi=200); plt.close(fig)

    x, _, psi = solve_fd(fd["final_n_interior"], fd["n_states"], L)
    fig, ax = plt.subplots(figsize=(8, 5))
    for n in [1, 5, 10]:
        ax.plot(x, exact_wavefunction_np(n, x, L), label=f"Exact n={n}")
        ax.plot(x, psi[:, n-1], "--", label=f"FD n={n}")
    ax.set(xlabel="x", ylabel="psi", title="Exact vs finite-difference eigenfunctions")
    ax.legend(ncol=2); fig.tight_layout(); fig.savefig(figures / "p1_wavefunctions_exact_vs_fd.png", dpi=200); plt.close(fig)
    print(f"Saved P1 results to {results}")


if __name__ == "__main__":
    main()
