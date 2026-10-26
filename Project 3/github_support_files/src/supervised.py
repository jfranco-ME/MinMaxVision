from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .models import TrialNet
from .physics import exact_wavefunction_torch
from .pinn import set_seed


def train_supervised_state(
    target_n: int,
    seed: int,
    *,
    domain_length: float = 1.0,
    width: int = 32,
    depth: int = 2,
    activation: str = "tanh",
    n_train: int = 257,
    adam_epochs: int = 3000,
    adam_lr: float = 1e-3,
    adam_lr_final: float = 1e-4,
    lr_drop_fraction: float = 0.65,
    lbfgs_max_iter: int = 60,
    lbfgs_lr: float = 0.8,
    evaluation_points: int = 2001,
    torch_num_threads: int = 1,
    output_dir: str | Path | None = None,
):
    set_seed(seed)
    if not torch.cuda.is_available():
        torch.set_num_threads(torch_num_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrialNet(width, depth, activation, domain_length).to(device)
    x = torch.linspace(0.0, domain_length, n_train, device=device).reshape(-1, 1)
    target = exact_wavefunction_torch(x[:, 0], target_n, domain_length)

    opt = torch.optim.Adam(model.parameters(), lr=adam_lr)
    drop = int(lr_drop_fraction * adam_epochs)
    best = float("inf")
    best_state = None
    start = time.perf_counter()
    for epoch in range(adam_epochs + 1):
        if epoch == drop:
            opt.param_groups[0]["lr"] = adam_lr_final
        opt.zero_grad()
        y = model(x)[:, 0]
        loss = torch.mean((y - target) ** 2)
        loss.backward()
        opt.step()
        val = float(loss.detach().cpu())
        if val < best:
            best = val
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=lbfgs_lr, max_iter=lbfgs_max_iter, history_size=50,
        tolerance_grad=1e-10, tolerance_change=1e-12, line_search_fn="strong_wolfe"
    )

    def closure():
        lbfgs.zero_grad()
        y = model(x)[:, 0]
        loss = torch.mean((y - target) ** 2)
        loss.backward()
        return loss

    lbfgs.step(closure)
    elapsed = time.perf_counter() - start

    xe = torch.linspace(0.0, domain_length, evaluation_points, device=device).reshape(-1, 1)
    with torch.no_grad():
        pred = model(xe)[:, 0]
        ref = exact_wavefunction_torch(xe[:, 0], target_n, domain_length)
        if torch.trapezoid(pred * ref, xe[:, 0]) < 0:
            pred = -pred
        rel = torch.linalg.vector_norm(pred - ref) / torch.linalg.vector_norm(ref)
    summary = {
        "target_n": target_n,
        "seed": seed,
        "training_time_s": elapsed,
        "wavefunction_error_percent": 100.0 * float(rel.cpu()),
        "device": str(device),
        "architecture": f"1-{width}" + (f"-{width}" * (depth - 1)) + f"-1 {activation}",
    }
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "x": xe[:, 0].cpu().numpy(),
            "psi_exact": ref.cpu().numpy(),
            "psi_supervised": pred.cpu().numpy(),
        }).to_csv(out / f"supervised_n{target_n:02d}_seed{seed}.csv", index=False)
        pd.DataFrame([summary]).to_csv(out / f"supervised_summary_n{target_n:02d}_seed{seed}.csv", index=False)
    return summary, model
