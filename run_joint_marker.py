# -*- coding: utf-8 -*-
"""Artifactual excess when the 4 by 4 bins use both measured markers.

Written before the excess was read. Does not overwrite
results/saturating_marker/excess.csv, results/neuron_marker/saturating.csv,
or results/mixture_bound/noise_grid.csv.

The one-marker runs pass the same array as both binning axes, so the quartile
cross collapses to four bins. Here the fractions are independent, both program
scores equal their sum, and each binning axis is that fraction's own measured
curve. This is the two-latent mixture already used for the perfect-marker
check, with the measured curves in place of the fractions.

Knots and standard deviations are the values already stored for the myeloid
count mixtures and the equal-depth neuronal mixtures. n = 4000. Seed 11 draws
the fractions and the noiseless permutation. Seed 12 draws the two independent
scatter terms and the scatter permutation. The scatter is the replicate
standard deviation of the 200-cell mixtures. It is not a noise level chosen
to hit a correlation.

The reported number is the scatter excess. Reading, fixed before it is known,
against the observed leftover of 0.546:
    excess <= 0.40: the leftover still clears this artifact, and the statement
        that the measured scatter does not manufacture the leftover stands.
    excess >= 0.45: that statement is not supported. The smooth residual is
        uninformative and is not a decision row.
    otherwise: report the number. Do not say the leftover clears it, and do
        not call the residual uninformative.
The floor of 0.10 is not changed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import run_null_standard as ns

OUT = Path(r"G:\James Zou Project\operative_state\results\joint_marker")
OUT.mkdir(parents=True, exist_ok=True)

N = 4000
FRACTION = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
NEURON_MEAN = np.array([
    0.2495868816, 0.3284113955, 0.3518443269, 0.3685371647, 0.3798714004,
    0.3797348019, 0.3849810485, 0.3939785873, 0.3887213881, 0.3951793010,
    0.3968998277,
])
NEURON_SD = np.array([
    0.0202789336, 0.0128508687, 0.0143245255, 0.0116512886, 0.0122922104,
    0.0102468615, 0.0117337354, 0.0124990649, 0.0123806820, 0.0089215897,
    0.0055156839,
])
MYELOID_MEAN = np.array([
    0.1472916592, 0.3148118501, 0.3867858164, 0.4164072830, 0.4391042776,
    0.4542209556, 0.4662517421, 0.4771383519, 0.4823093485, 0.4873449058,
    0.4917107708,
])
MYELOID_SD = np.array([
    0.0394732591, 0.0427389392, 0.0294984817, 0.0210943267, 0.0155030609,
    0.0128273692, 0.0126103572, 0.0069931752, 0.0065827667, 0.0049911088,
    0.0020451437,
])


def n_bins(neuronal: np.ndarray, myeloid: np.ndarray) -> int:
    frame = pd.DataFrame({"n": neuronal, "m": myeloid})
    frame["nq"] = pd.qcut(frame["n"], 4, labels=False, duplicates="drop")
    frame["mq"] = pd.qcut(frame["m"], 4, labels=False, duplicates="drop")
    return int((frame["nq"].astype(str) + "_" + frame["mq"].astype(str)).nunique())


def one(score: np.ndarray, neuronal: np.ndarray, myeloid: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    observed, null = ns.stratified_draws(score, score, neuronal, myeloid, rng)
    center = float(np.median(null))
    return {
        "observed": float(observed),
        "null_center": center,
        "excess": float(observed - center),
        "n": int(score.size),
        "n_bins": n_bins(neuronal, myeloid),
    }


def marker_fidelity(fraction: np.ndarray, marker: np.ndarray) -> dict:
    return {
        "spearman": float(spearmanr(fraction, marker).statistic),
        "pearson": float(pearsonr(fraction, marker).statistic),
    }


def main() -> None:
    latent = np.random.default_rng(11)
    neuronal_fraction = latent.uniform(size=N)
    myeloid_fraction = latent.uniform(size=N)
    score = neuronal_fraction + myeloid_fraction
    neuron_curve = np.interp(neuronal_fraction, FRACTION, NEURON_MEAN)
    myeloid_curve = np.interp(myeloid_fraction, FRACTION, MYELOID_MEAN)
    noiseless = one(score, neuron_curve, myeloid_curve, 11)
    noiseless["neuronal_marker"] = marker_fidelity(neuronal_fraction, neuron_curve)
    noiseless["myeloid_marker"] = marker_fidelity(myeloid_fraction, myeloid_curve)
    scatter = np.random.default_rng(12)
    neuron_noisy = neuron_curve + scatter.normal(size=N) * np.interp(neuronal_fraction, FRACTION, NEURON_SD)
    myeloid_noisy = myeloid_curve + scatter.normal(size=N) * np.interp(myeloid_fraction, FRACTION, MYELOID_SD)
    noisy = one(score, neuron_noisy, myeloid_noisy, 12)
    noisy["neuronal_marker"] = marker_fidelity(neuronal_fraction, neuron_noisy)
    noisy["myeloid_marker"] = marker_fidelity(myeloid_fraction, myeloid_noisy)
    excess = noisy["excess"]
    if excess <= 0.40:
        reading = "leftover_clears"
    elif excess >= 0.45:
        reading = "residual_uninformative"
    else:
        reading = "intermediate"
    table = pd.DataFrame([
        {
            "marker": "joint_measured_curves",
            "observed": noiseless["observed"],
            "null_center": noiseless["null_center"],
            "excess": noiseless["excess"],
            "n": noiseless["n"],
            "n_bins": noiseless["n_bins"],
            "neuronal_spearman": noiseless["neuronal_marker"]["spearman"],
            "myeloid_spearman": noiseless["myeloid_marker"]["spearman"],
            "neuronal_pearson": noiseless["neuronal_marker"]["pearson"],
            "myeloid_pearson": noiseless["myeloid_marker"]["pearson"],
        },
        {
            "marker": "joint_curves_plus_measured_sd",
            "observed": noisy["observed"],
            "null_center": noisy["null_center"],
            "excess": noisy["excess"],
            "n": noisy["n"],
            "n_bins": noisy["n_bins"],
            "neuronal_spearman": noisy["neuronal_marker"]["spearman"],
            "myeloid_spearman": noisy["myeloid_marker"]["spearman"],
            "neuronal_pearson": noisy["neuronal_marker"]["pearson"],
            "myeloid_pearson": noisy["myeloid_marker"]["pearson"],
        },
    ])
    table.to_csv(OUT / "excess.csv", index=False)
    verdict = {
        "scores": "sum of two independent fractions",
        "bins": "quartiles of the neuronal curve crossed with quartiles of the myeloid curve",
        "single_marker_files_overwritten": False,
        "noiseless": noiseless,
        "scatter": noisy,
        "reading": reading,
    }
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
