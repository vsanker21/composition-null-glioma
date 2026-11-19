# -*- coding: utf-8 -*-
"""In-tissue marker reliability, and the joint artifact at that reliability.

Written before either number was read. Does not overwrite
results/joint_marker/excess.csv, the myeloid or neuronal saturating files,
or noise_grid.csv. Does not scale a standard deviation by the square root
of a cell count.

On each tumor section the eight marker genes are divided into every balanced
partition of four against four (35 partitions). Each half is the singscore of
its four genes, ranked inside the spot over the whole gene universe. The
split-half correlation is the Pearson correlation of the two halves across
spots in that section. Pearson is the input because the Spearman-Brown
correction, 2r / (1 + r), is a variance ratio for the full eight-gene score.
A section's reliability is the mean of those 35 corrected values. The cohort
reliability is the unweighted mean across sections. Sections are the same
folders as the spatial layer, and a section enters when at least 100 spots
have both full marker scores.

That reliability replaces the pseudobulk scatter. Both program scores are
still the sum of two independent fractions, and the binning axes are still
the measured curves, n = 4000, seeds 11 and 12, the same 4 by 4 permutation.
The noise on a curve is independent and Gaussian with
standard deviation s * sqrt((1 - R) / R), where s is the standard deviation
of that curve across the simulated fractions and R is the cohort reliability.
No knot-wise mixture standard deviation is used.

Reading of the resulting excess, fixed before it is known, against the
observed leftover of 0.546:
    excess <= 0.40: the leftover still clears this artifact.
    excess >= 0.45: the leftover is not clear of this artifact. The smooth
        residual is uninformative and is not a decision row.
    excess >= 0.546: this artifact meets or exceeds the observed leftover.
    otherwise: report the number. Do not say the leftover clears it, and do
        not call the residual uninformative.
The floor of 0.10 is not changed. A correlation of either marker with
detected genes is stored and is not used to change R.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import run_gates as rg
import run_joint_marker as jm
import run_null_standard as ns

OUT = Path(r"G:\James Zou Project\operative_state\results\joint_marker")
N = 4000


def partitions():
    for rest in combinations(range(1, 8), 3):
        first = (0,) + rest
        second = tuple(i for i in range(8) if i not in first)
        yield first, second


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a0 = a - a.mean()
    b0 = b - b.mean()
    denom = np.sqrt(np.dot(a0, a0) * np.dot(b0, b0))
    if denom == 0:
        return np.nan
    return float(np.dot(a0, b0) / denom)


def split_half(centered: np.ndarray) -> tuple[float, float, float]:
    """Mean Spearman-Brown reliability, mean Pearson r, mean Spearman r."""
    corrected = []
    raw = []
    ranks = []
    for first, second in partitions():
        h1 = centered[list(first)].mean(axis=0)
        h2 = centered[list(second)].mean(axis=0)
        r = pearson(h1, h2)
        raw.append(r)
        corrected.append(2.0 * r / (1.0 + r))
        ranks.append(ns.spearman(h1, h2))
    return float(np.mean(corrected)), float(np.mean(raw)), float(np.mean(ranks))


def section_rows() -> pd.DataFrame:
    folders = sorted(
        p for p in ns.VISIUM.iterdir()
        if p.is_dir() and "_T_ST" in p.name and "_TI_" not in p.name and "_TC_" not in p.name and "_C_ST" not in p.name
    )
    rows = []
    neuronal = list(rg.NEURONAL)
    myeloid = list(rg.MYELOID)
    if len(neuronal) != 8 or len(myeloid) != 8:
        raise SystemExit("each marker list must contain eight genes")
    if len(list(partitions())) != 35:
        raise SystemExit("a list of eight genes has 35 balanced partitions")
    for folder in folders:
        h5_path = folder / "outs" / "filtered_feature_bc_matrix.h5"
        if not h5_path.exists():
            continue
        print(folder.name, flush=True)
        counts, names = ns.read_visium_counts(h5_path)
        index = pd.Index(names)
        missing = [g for g in neuronal + myeloid if g not in index]
        if missing:
            raise SystemExit("marker gene absent from " + folder.name + ": " + ",".join(missing))
        ranks = rankdata(counts, axis=0, method="average")
        n_genes = ranks.shape[0]
        centered = (ranks - (n_genes + 1) / 2.0) / n_genes
        del ranks
        detected = np.asarray((counts > 0).sum(axis=0)).ravel()
        library = np.asarray(counts.sum(axis=0)).ravel()
        del counts
        take = index.get_indexer(neuronal + myeloid)
        gene_centered = centered[take]
        del centered
        full_n = gene_centered[:8].mean(axis=0)
        full_m = gene_centered[8:].mean(axis=0)
        keep = np.isfinite(full_n) & np.isfinite(full_m)
        if int(keep.sum()) < 100:
            print("skip", folder.name, int(keep.sum()), flush=True)
            continue
        rel_n, raw_n, sp_n = split_half(gene_centered[:8, keep])
        rel_m, raw_m, sp_m = split_half(gene_centered[8:, keep])
        rows.append({
            "sample": folder.name.lstrip("#"),
            "n_spots": int(keep.sum()),
            "neuronal_reliability": rel_n,
            "myeloid_reliability": rel_m,
            "neuronal_split_pearson": raw_n,
            "myeloid_split_pearson": raw_m,
            "neuronal_split_spearman": sp_n,
            "myeloid_split_spearman": sp_m,
            "neuronal_detected_pearson": pearson(full_n[keep], detected[keep].astype(float)),
            "myeloid_detected_pearson": pearson(full_m[keep], detected[keep].astype(float)),
            "neuronal_library_pearson": pearson(full_n[keep], np.log1p(library[keep])),
            "myeloid_library_pearson": pearson(full_m[keep], np.log1p(library[keep])),
        })
        print(
            f"  spots {int(keep.sum())} neuronal R {rel_n:.4f} myeloid R {rel_m:.4f}",
            flush=True,
        )
    return pd.DataFrame(rows)


def artifact(r_neuronal: float, r_myeloid: float) -> dict:
    latent = np.random.default_rng(11)
    neuronal_fraction = latent.uniform(size=N)
    myeloid_fraction = latent.uniform(size=N)
    score = neuronal_fraction + myeloid_fraction
    neuron_curve = np.interp(neuronal_fraction, jm.FRACTION, jm.NEURON_MEAN)
    myeloid_curve = np.interp(myeloid_fraction, jm.FRACTION, jm.MYELOID_MEAN)
    var_n = float(np.var(neuron_curve))
    var_m = float(np.var(myeloid_curve))
    sigma_n = float(np.sqrt(var_n * (1.0 - r_neuronal) / r_neuronal))
    sigma_m = float(np.sqrt(var_m * (1.0 - r_myeloid) / r_myeloid))
    noise = np.random.default_rng(12)
    neuron_marker = neuron_curve + noise.normal(size=N) * sigma_n
    myeloid_marker = myeloid_curve + noise.normal(size=N) * sigma_m
    observed, null = ns.stratified_draws(score, score, neuron_marker, myeloid_marker, np.random.default_rng(12))
    center = float(np.median(null))
    frame = pd.DataFrame({"n": neuron_marker, "m": myeloid_marker})
    frame["nq"] = pd.qcut(frame["n"], 4, labels=False, duplicates="drop")
    frame["mq"] = pd.qcut(frame["m"], 4, labels=False, duplicates="drop")
    n_bins = int((frame["nq"].astype(str) + "_" + frame["mq"].astype(str)).nunique())
    return {
        "observed": float(observed),
        "null_center": center,
        "excess": float(observed - center),
        "n": N,
        "n_bins": n_bins,
        "sigma_neuronal": sigma_n,
        "sigma_myeloid": sigma_m,
        "curve_sd_neuronal": float(np.sqrt(var_n)),
        "curve_sd_myeloid": float(np.sqrt(var_m)),
        "realized_reliability_neuronal": float(var_n / np.var(neuron_marker)),
        "realized_reliability_myeloid": float(var_m / np.var(myeloid_marker)),
        "neuronal_spearman": float(ns.spearman(neuronal_fraction, neuron_marker)),
        "myeloid_spearman": float(ns.spearman(myeloid_fraction, myeloid_marker)),
    }


def main() -> None:
    sections = section_rows()
    if len(sections) == 0:
        raise SystemExit("no section entered")
    sections.to_csv(OUT / "spot_reliability_sections.csv", index=False)
    r_n = float(sections["neuronal_reliability"].mean())
    r_m = float(sections["myeloid_reliability"].mean())
    if not (0.0 < r_n < 1.0 and 0.0 < r_m < 1.0):
        raise SystemExit(f"reliability outside (0, 1): neuronal {r_n}, myeloid {r_m}")
    fit = artifact(r_n, r_m)
    excess = fit["excess"]
    if excess >= 0.546:
        reading = "artifact_meets_leftover"
    elif excess >= 0.45:
        reading = "residual_uninformative"
    elif excess <= 0.40:
        reading = "leftover_clears"
    else:
        reading = "intermediate"
    verdict = {
        "pseudobulk_scatter_used": False,
        "sqrt_n_scaling_used": False,
        "joint_excess_file_overwritten": False,
        "split": "mean of 35 balanced partitions, Spearman-Brown of the Pearson correlation",
        "cohort": "unweighted mean across sections",
        "n_sections": int(len(sections)),
        "neuronal_reliability": r_n,
        "myeloid_reliability": r_m,
        "neuronal_reliability_min": float(sections["neuronal_reliability"].min()),
        "neuronal_reliability_max": float(sections["neuronal_reliability"].max()),
        "myeloid_reliability_min": float(sections["myeloid_reliability"].min()),
        "myeloid_reliability_max": float(sections["myeloid_reliability"].max()),
        "mean_neuronal_detected_pearson": float(sections["neuronal_detected_pearson"].mean()),
        "mean_myeloid_detected_pearson": float(sections["myeloid_detected_pearson"].mean()),
        "artifact": fit,
        "reading": reading,
    }
    (OUT / "spot_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
