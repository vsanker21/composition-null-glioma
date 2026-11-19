# -*- coding: utf-8 -*-
"""Excess when the marker is the measured saturating curve.

Written before the excess was read. Does not overwrite noise_grid.csv.

The Gaussian grid builds marker = rho * z + sqrt(1 - rho^2) * noise.
That rho is a linear weight. The bins and the leftover are rank operations,
so a strictly increasing curve of any shape has the same quartiles as the
fraction it comes from.

Knots are the mean myeloid singscores already stored in myeloid_mixtures.csv.
They increase at every step, so the piecewise-linear interpolant is strictly
increasing. No target correlation is imposed.

Primary: that interpolant, no added noise. n = 4000, seed 11, the same
quartile permutation as the Gaussian grid. Both scores equal the fraction.
Both binning axes equal the marker, matching the one-latent grid.

Named check, not a replacement of the primary: the same curve plus independent
Gaussian noise with the standard deviation measured at each knot. Seed 12.
That scatter is the replicate scatter of the 200-cell mixtures. It is not a
noise level chosen to hit a correlation.

Reading, fixed before the excess is known:
    monotone excess within 0.02 of 1/16: curvature alone does not raise the leftover.
    scatter excess <= 0.10: measured scatter on this curve does not manufacture
        an excess of the observed size.
    scatter excess >= 0.40: the curve plus its own scatter can manufacture a
        leftover of the observed size.
    otherwise: report the number and do not call the spatial layer secured.
The decision floor of 0.10 is not changed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import run_null_standard as ns

OUT = Path(r"G:\James Zou Project\operative_state\results\saturating_marker")
OUT.mkdir(parents=True, exist_ok=True)

N = 4000
FRACTION = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
MEAN = np.array([
    0.1472916592, 0.3148118501, 0.3867858164, 0.4164072830, 0.4391042776,
    0.4542209556, 0.4662517421, 0.4771383519, 0.4823093485, 0.4873449058,
    0.4917107708,
])
SD = np.array([
    0.0394732591, 0.0427389392, 0.0294984817, 0.0210943267, 0.0155030609,
    0.0128273692, 0.0126103572, 0.0069931752, 0.0065827667, 0.0049911088,
    0.0020451437,
])


def one(marker: np.ndarray, latent: np.ndarray, seed: int) -> dict:
    fit = {}
    rng = np.random.default_rng(seed)
    observed, null = ns.stratified_draws(latent, latent, marker, marker, rng)
    center = float(np.median(null))
    fit["observed"] = float(observed)
    fit["null_center"] = center
    fit["excess"] = float(observed - center)
    fit["n"] = int(latent.size)
    fit["spearman_with_fraction"] = float(spearmanr(latent, marker).statistic)
    fit["pearson_with_fraction"] = float(pearsonr(latent, marker).statistic)
    return fit


def main() -> None:
    if not np.all(np.diff(MEAN) > 0):
        raise SystemExit("knots are not strictly increasing; the curve is not the measured one")
    latent = np.random.default_rng(11).uniform(size=N)
    monotone = one(np.interp(latent, FRACTION, MEAN), latent, 11)
    scatter_rng = np.random.default_rng(12)
    noisy = np.interp(latent, FRACTION, MEAN) + scatter_rng.normal(size=N) * np.interp(latent, FRACTION, SD)
    scatter = one(noisy, latent, 12)
    monotone["within_0.02_of_one_sixteenth"] = bool(abs(monotone["excess"] - 1.0 / 16.0) <= 0.02)
    if scatter["excess"] <= 0.10:
        scatter_reading = "scatter_does_not_manufacture_0.55"
    elif scatter["excess"] >= 0.40:
        scatter_reading = "scatter_can_manufacture_observed_leftover"
    else:
        scatter_reading = "between_floor_and_observed"
    table = pd.DataFrame([
        {"marker": "saturating_monotone", **monotone},
        {"marker": "saturating_plus_measured_sd", **scatter},
    ])
    table.to_csv(OUT / "excess.csv", index=False)
    verdict = {
        "expression_data_used": False,
        "gaussian_grid_overwritten": False,
        "monotone": monotone,
        "scatter": scatter,
        "scatter_reading": scatter_reading,
    }
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
