# -*- coding: utf-8 -*-
"""Artifactual excess for the measured neuronal marker curve.

Written before the excess was read. Does not overwrite the myeloid file
results/saturating_marker/excess.csv or noise_grid.csv.

Knots are the mean equal-depth neuronal singscores and their sample standard
deviations. Two steps fall: fraction 0.4 to 0.5 by 0.00014, and 0.7 to 0.8
by 0.0053. The curve is not forced to increase.

Same permutation as the myeloid curve. n = 4000. Both scores equal the
fraction. Both binning axes equal the marker. Seed 11 draws the fraction and
the noiseless permutation. Seed 12 draws the measured scatter and its
permutation.

The reported number is the scatter excess. The noiseless interpolant is stored
beside it. Reading, fixed before either excess is known:
    scatter excess <= 0.30: 0.546 clears it by a wide margin.
    scatter excess >= 0.40: the margin is thin.
    otherwise: report the number and call the margin neither wide nor thin.
The floor of 0.10 is not changed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import run_null_standard as ns

OUT = Path(r"G:\James Zou Project\operative_state\results\neuron_marker")
N = 4000
FRACTION = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
MEAN = np.array([
    0.2495868816, 0.3284113955, 0.3518443269, 0.3685371647, 0.3798714004,
    0.3797348019, 0.3849810485, 0.3939785873, 0.3887213881, 0.3951793010,
    0.3968998277,
])
SD = np.array([
    0.0202789336, 0.0128508687, 0.0143245255, 0.0116512886, 0.0122922104,
    0.0102468615, 0.0117337354, 0.0124990649, 0.0123806820, 0.0089215897,
    0.0055156839,
])


def one(marker: np.ndarray, latent: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    observed, null = ns.stratified_draws(latent, latent, marker, marker, rng)
    center = float(np.median(null))
    return {
        "observed": float(observed),
        "null_center": center,
        "excess": float(observed - center),
        "n": int(latent.size),
        "spearman_with_fraction": float(spearmanr(latent, marker).statistic),
        "pearson_with_fraction": float(pearsonr(latent, marker).statistic),
    }


def main() -> None:
    latent = np.random.default_rng(11).uniform(size=N)
    noiseless = one(np.interp(latent, FRACTION, MEAN), latent, 11)
    scatter_rng = np.random.default_rng(12)
    noisy = np.interp(latent, FRACTION, MEAN) + scatter_rng.normal(size=N) * np.interp(latent, FRACTION, SD)
    scatter = one(noisy, latent, 12)
    excess = scatter["excess"]
    if excess <= 0.30:
        reading = "wide_margin"
    elif excess >= 0.40:
        reading = "thin_margin"
    else:
        reading = "intermediate_margin"
    table = pd.DataFrame([
        {"marker": "neuronal_measured_curve", **noiseless},
        {"marker": "neuronal_curve_plus_measured_sd", **scatter},
    ])
    table.to_csv(OUT / "saturating.csv", index=False)
    verdict = {
        "myeloid_file_overwritten": False,
        "curve_strictly_increasing": False,
        "noiseless": noiseless,
        "scatter": scatter,
        "reading": reading,
    }
    (OUT / "saturating_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
