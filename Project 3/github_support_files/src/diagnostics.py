from __future__ import annotations

import numpy as np
import pandas as pd

from .physics import exact_wavefunction_np


def analyze_solution_dataframe(df: pd.DataFrame, target_n: int, domain_length: float = 1.0, max_sine_mode: int = 30):
    x = df["x"].to_numpy(float)
    exact = df["psi_exact"].to_numpy(float)
    pred_col = "psi_pinn" if "psi_pinn" in df.columns else "psi_supervised"
    pred = df[pred_col].to_numpy(float)
    err = pred - exact
    l2e = np.sqrt(np.trapezoid(err**2, x))
    l2r = np.sqrt(np.trapezoid(exact**2, x))

    e0 = err[:-1]
    dx = x[1] - x[0]
    freqs = np.fft.rfftfreq(len(e0), d=dx)
    amp = 2 * np.abs(np.fft.rfft(e0)) / len(e0)
    amp[0] *= 0.5

    coeff_rows = []
    for m in range(1, max_sine_mode + 1):
        phi = exact_wavefunction_np(m, x, domain_length)
        c = float(np.trapezoid(err * phi, x))
        coeff_rows.append({"target_n": target_n, "mode_m": m, "error_coefficient": c, "abs_error_coefficient": abs(c)})

    return {
        "relative_L2_percent": 100.0 * l2e / l2r,
        "max_abs_error": float(np.max(np.abs(err))),
        "x_of_max_error": float(x[np.argmax(np.abs(err))]),
        "frequencies": freqs,
        "fft_amplitudes": amp,
        "error": err,
        "mode_coefficients": pd.DataFrame(coeff_rows),
    }


def aggregate_sweep(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.groupby("target_n")
        .agg(
            wavefunction_error_median_percent=("wavefunction_error_percent", "median"),
            wavefunction_error_min_percent=("wavefunction_error_percent", "min"),
            wavefunction_error_max_percent=("wavefunction_error_percent", "max"),
            energy_error_median_percent=("energy_error_percent", "median"),
            median_training_s=("training_time_s", "median"),
        )
        .reset_index()
    )
