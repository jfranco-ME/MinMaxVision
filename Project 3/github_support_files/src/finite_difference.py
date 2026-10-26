from __future__ import annotations

import time
import numpy as np
import pandas as pd
from scipy.linalg import eigh_tridiagonal

from .physics import exact_energy, exact_wavefunction_np, relative_l2_np


def solve_fd(n_interior: int, n_states: int = 10, domain_length: float = 1.0):
    if n_states > n_interior:
        raise ValueError("n_states must be <= n_interior")
    L = float(domain_length)
    h = L / (n_interior + 1)
    x_interior = np.linspace(h, L - h, n_interior)
    x_full = np.concatenate(([0.0], x_interior, [L]))

    main = np.full(n_interior, 2.0 / h**2)
    off = np.full(n_interior - 1, -1.0 / h**2)
    energies, vecs = eigh_tridiagonal(
        main, off, select="i", select_range=(0, n_states - 1), check_finite=False
    )

    psi_full = np.zeros((n_interior + 2, n_states))
    psi_full[1:-1, :] = vecs
    for j in range(n_states):
        psi_full[:, j] /= np.sqrt(np.trapezoid(psi_full[:, j] ** 2, x_full))
        ref = exact_wavefunction_np(j + 1, x_full, L)
        if np.trapezoid(psi_full[:, j] * ref, x_full) < 0:
            psi_full[:, j] *= -1.0
    return x_full, energies, psi_full


def benchmark_fd(n_interior: int, n_states: int, domain_length: float = 1.0, repeats: int = 25) -> float:
    solve_fd(n_interior, n_states, domain_length)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        solve_fd(n_interior, n_states, domain_length)
        times.append(time.perf_counter() - t0)
    return 1000.0 * float(np.median(times))


def convergence_table(grid_sizes, n_states=10, domain_length=1.0, timing_repeats=25) -> pd.DataFrame:
    rows = []
    L = float(domain_length)
    for N in grid_sizes:
        x, energies, psi = solve_fd(N, n_states, L)
        h = L / (N + 1)
        runtime_ms = benchmark_fd(N, n_states, L, timing_repeats)
        for j in range(n_states):
            n = j + 1
            e_ref = exact_energy(n, L)
            psi_ref = exact_wavefunction_np(n, x, L)
            rows.append({
                "N_interior": N,
                "h": h,
                "state_n": n,
                "E_exact": e_ref,
                "E_FD": energies[j],
                "E_rel_error": abs(energies[j] - e_ref) / abs(e_ref),
                "psi_rel_L2_error": relative_l2_np(psi[:, j], psi_ref, x),
                "runtime_ms_first_states": runtime_ms,
            })
    return pd.DataFrame(rows)
