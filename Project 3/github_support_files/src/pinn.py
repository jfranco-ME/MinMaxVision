from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .models import TrialNet
from .physics import exact_energy, exact_wavefunction_torch


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_grids(device, domain_length=1.0, n_residual=96, n_quadrature=257):
    L = float(domain_length)
    x_r = torch.linspace(0.0, L, n_residual + 2, device=device)[1:-1].reshape(-1, 1)
    x_r.requires_grad_(True)
    x_q = torch.linspace(0.0, L, n_quadrature, device=device).reshape(-1, 1)
    return x_r, x_q


def _projection_coefficients(raw_q, x_q, target_n, L):
    coeffs = []
    base = raw_q[:, 0]
    for j in range(1, target_n):
        phi = exact_wavefunction_torch(x_q[:, 0], j, L)
        coeff = torch.trapezoid(base * phi, x_q[:, 0]) / torch.trapezoid(phi**2, x_q[:, 0])
        coeffs.append(coeff)
    return coeffs


def _apply_projection(raw, x, coeffs, L):
    projected = raw[:, 0]
    for j, coeff in enumerate(coeffs, start=1):
        phi = exact_wavefunction_torch(x[:, 0], j, L)
        projected = projected - coeff * phi
    return projected


def normalized_projected(model, target_n, x_q, x_other=None, domain_length=1.0):
    L = float(domain_length)
    raw_q = model(x_q)
    coeffs = _projection_coefficients(raw_q, x_q, target_n, L)
    proj_q = _apply_projection(raw_q, x_q, coeffs, L)
    norm = torch.sqrt(torch.trapezoid(proj_q**2, x_q[:, 0]) + 1e-12)
    psi_q = proj_q / norm
    if x_other is None:
        return psi_q
    raw_o = model(x_other)
    proj_o = _apply_projection(raw_o, x_other, coeffs, L)
    return psi_q, proj_o / norm


def rayleigh_energy(psi_q, x_q):
    dx = x_q[1:, 0] - x_q[:-1, 0]
    slopes = (psi_q[1:] - psi_q[:-1]) / dx
    return torch.sum(slopes**2 * dx)


def loss_components(model, target_n, x_r, x_q, domain_length=1.0, lambda_energy=0.01):
    L = float(domain_length)
    psi_q, psi_r = normalized_projected(model, target_n, x_q, x_r, L)
    energy = rayleigh_energy(psi_q, x_q)
    dpsi = torch.autograd.grad(psi_r, x_r, torch.ones_like(psi_r), create_graph=True)[0]
    ddpsi = torch.autograd.grad(dpsi, x_r, torch.ones_like(dpsi), create_graph=True)[0]
    residual = -ddpsi[:, 0] - energy * psi_r
    pde_loss = torch.mean(residual**2)
    energy_term = lambda_energy * energy
    return pde_loss, energy, energy_term


def evaluate(model, target_n, domain_length=1.0, n_eval=2001, device=None):
    if device is None:
        device = next(model.parameters()).device
    L = float(domain_length)
    x = torch.linspace(0.0, L, n_eval, device=device).reshape(-1, 1)
    psi = normalized_projected(model, target_n, x, domain_length=L)
    energy = rayleigh_energy(psi, x)
    ref = exact_wavefunction_torch(x[:, 0], target_n, L)
    if torch.trapezoid(psi * ref, x[:, 0]) < 0:
        psi = -psi
    rel = torch.linalg.vector_norm(psi - ref) / torch.linalg.vector_norm(ref)
    e_ref = exact_energy(target_n, L)
    return {
        "x": x[:, 0].detach().cpu().numpy(),
        "psi_exact": ref.detach().cpu().numpy(),
        "psi_pinn": psi.detach().cpu().numpy(),
        "energy_exact": e_ref,
        "energy_pinn": float(energy.detach().cpu()),
        "energy_error_percent": 100.0 * abs(float(energy.detach().cpu()) - e_ref) / e_ref,
        "wavefunction_error_percent": 100.0 * float(rel.detach().cpu()),
    }


def train_pinn_state(
    target_n: int,
    seed: int,
    *,
    domain_length: float = 1.0,
    width: int = 32,
    depth: int = 2,
    activation: str = "tanh",
    n_residual: int = 96,
    n_quadrature: int = 257,
    adam_epochs: int = 3000,
    adam_lr: float = 1e-3,
    adam_lr_final: float = 1e-4,
    lr_drop_fraction: float = 0.65,
    lbfgs_max_iter: int = 60,
    lbfgs_lr: float = 0.8,
    lambda_energy: float = 0.01,
    evaluation_points: int = 2001,
    torch_num_threads: int = 1,
    output_dir: str | Path | None = None,
    model_dir: str | Path | None = None,
):
    set_seed(seed)
    if not torch.cuda.is_available():
        torch.set_num_threads(torch_num_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrialNet(width, depth, activation, domain_length).to(device)
    x_r, x_q = make_grids(device, domain_length, n_residual, n_quadrature)

    opt = torch.optim.Adam(model.parameters(), lr=adam_lr)
    drop = int(lr_drop_fraction * adam_epochs)
    best = float("inf")
    best_state = None
    history = []
    start = time.perf_counter()

    for epoch in range(adam_epochs + 1):
        if epoch == drop:
            opt.param_groups[0]["lr"] = adam_lr_final
        opt.zero_grad()
        lpde, energy, le = loss_components(model, target_n, x_r, x_q, domain_length, lambda_energy)
        loss = lpde + le
        loss.backward()
        opt.step()
        val = float(loss.detach().cpu())
        if val < best:
            best = val
            best_state = copy.deepcopy(model.state_dict())
        if epoch % 25 == 0 or epoch == adam_epochs:
            history.append({
                "epoch": epoch,
                "learning_rate": opt.param_groups[0]["lr"],
                "total_loss": val,
                "pde_loss": float(lpde.detach().cpu()),
                "energy": float(energy.detach().cpu()),
            })

    model.load_state_dict(best_state)
    lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=lbfgs_lr, max_iter=lbfgs_max_iter, history_size=50,
        tolerance_grad=1e-10, tolerance_change=1e-12, line_search_fn="strong_wolfe"
    )

    def closure():
        lbfgs.zero_grad()
        lpde, _, le = loss_components(model, target_n, x_r, x_q, domain_length, lambda_energy)
        loss = lpde + le
        loss.backward()
        return loss

    lbfgs.step(closure)
    elapsed = time.perf_counter() - start
    metrics = evaluate(model, target_n, domain_length, evaluation_points, device)
    summary = {
        "target_n": target_n,
        "seed": seed,
        "training_time_s": elapsed,
        "energy_exact": metrics["energy_exact"],
        "energy_pinn": metrics["energy_pinn"],
        "energy_error_percent": metrics["energy_error_percent"],
        "wavefunction_error_percent": metrics["wavefunction_error_percent"],
        "device": str(device),
        "architecture": f"1-{width}" + (f"-{width}" * (depth - 1)) + f"-1 {activation}",
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(out / f"pinn_training_n{target_n:02d}_seed{seed}.csv", index=False)
        pd.DataFrame({"x": metrics["x"], "psi_exact": metrics["psi_exact"], "psi_pinn": metrics["psi_pinn"]}).to_csv(
            out / f"pinn_n{target_n:02d}_seed{seed}.csv", index=False
        )
        pd.DataFrame([summary]).to_csv(out / f"pinn_summary_n{target_n:02d}_seed{seed}.csv", index=False)
    if model_dir is not None:
        md = Path(model_dir)
        md.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), md / f"pinn_n{target_n:02d}_seed{seed}.pt")
    return summary, metrics, model
