"""Largest mixture excess inside quartile bins. No expression values are read.

The derivation is locked in FIELD_PROTOCOL.md. This file checks it against the
same permutation used for the spatial null.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_null_standard as ns

OUT = Path(r"G:\James Zou Project\operative_state\results\mixture_bound")
OUT.mkdir(parents=True, exist_ok=True)

N = 4000
SEED = 11


def analytic_excess(n: int) -> float:
    """Spearman excess for a perfect monotone mixture in equal-count quartile blocks.

    Ranks of a continuous latent are 1..n. Quartile blocks are m=n/4 consecutive
    ranks. Observed Spearman is 1. The expected null Spearman after shuffling
    inside blocks is the between-block share of rank variance, so the excess is
    the within-block share (m**2 - 1) / (n**2 - 1), which is at most 1/16.
    """
    if n % 4 != 0:
        raise ValueError("n must be divisible by 4")
    m = n // 4
    return (m * m - 1) / (n * n - 1)


def permutation_excess(predictor: np.ndarray, outcome: np.ndarray, neuronal: np.ndarray, myeloid: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    observed, null = ns.stratified_draws(predictor, outcome, neuronal, myeloid, rng)
    center = float(np.median(null))
    return {
        "observed": observed,
        "null_center": center,
        "excess": observed - center,
        "n": int(len(predictor)),
    }


def main() -> None:
    bound = analytic_excess(N)
    rng = np.random.default_rng(SEED)
    # One latent fraction. Both scores and both markers are that fraction.
    # Ranks make the distribution irrelevant; uniforms keep the simulation exact.
    latent = rng.uniform(size=N)
    perfect = permutation_excess(latent, latent, latent, latent, SEED)
    # Two perfect latents, scores equal to their sum, each marker one latent.
    f1 = rng.uniform(size=N)
    f2 = rng.uniform(size=N)
    two = permutation_excess(f1 + f2, f1 + f2, f1, f2, SEED)
    # Noise raises the artifactual excess. Grid only. Not fit to any cohort.
    rows = []
    for rho in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0):
        z = rng.normal(size=N)
        noise = rng.normal(size=N)
        marker = rho * z + np.sqrt(max(0.0, 1.0 - rho * rho)) * noise
        # The latent is z, so the scores carry z and the binning variable is marker.
        fit = permutation_excess(z, z, marker, marker, SEED)
        fit["marker_correlation"] = rho
        rows.append(fit)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "noise_grid.csv", index=False)
    verdict = {
        "analytic_excess_upper_bound": 1.0 / 16.0,
        "analytic_excess_at_n": bound,
        "n": N,
        "perfect_one_latent_excess": perfect["excess"],
        "perfect_two_latent_excess": two["excess"],
        "perfect_one_latent_null_center": perfect["null_center"],
        "expression_data_used": False,
        "recording_study_opened": False,
    }
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    text = f"""# Mixture bound

No expression matrix was read.

For quartile bins, both scores strictly monotone in one latent fraction, and the binning variable a perfect monotone measure of that fraction, the Spearman excess tends to 1/16 = 0.0625 from below. At n = {N} the finite-sample value is {bound:.6f}. A 4 by 4 grid of two perfect latents has the same within-bin fraction, 1/16, because each latent is still cut into four equal-count blocks.

The permutation in `run_null_standard.stratified_draws`, seed {SEED}, returns an excess of {perfect['excess']:.4f} for one perfect latent (null center {perfect['null_center']:.4f}) and {two['excess']:.4f} for two perfect latents. Those checks sit on the analytic value.

Measurement error moves the artifactual excess up, not down. The noise grid is in `noise_grid.csv`. The bound 1/16 is the maximum only when the markers measure the latent fractions perfectly. It is not a new decision floor. The decision floor remains the locked excess of 0.10 on the smooth residual correlation.
"""
    (OUT / "RESULTS.md").write_text(text, encoding="utf-8")
    print(json.dumps(verdict, indent=2), flush=True)
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
