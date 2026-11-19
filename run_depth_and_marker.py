# -*- coding: utf-8 -*-
"""Two locked checks. Rules below were written before either result was read.

Depth partial
    Same cells, donors, and floor as run_within_donor.py.
    Primary covariate pair: cell-cycle score and log1p(total counts).
    Named sensitivity, not a rescue: cell-cycle score and log1p(genes detected).
    Estimator: the same partial-Spearman formula, extended to two covariates.
    Decision: unweighted mean. Holds only if both the 81-donor mean and the
    leave-Neftel mean are at least 0.20. The original file is not overwritten.

Marker calibration
    Pools fixed from the released cell_type labels:
        myeloid = microglial cell, macrophage, monocyte
        background = malignant cell
        neuron is counted and not mixed, because a pool under 100 cells
        cannot support a 200-cell pseudobulk.
    Reservoir, seed 11, filled on the sequential pass:
        4000 microglial, 4000 macrophage, 2000 monocyte, 4000 malignant.
    Each pseudobulk: 200 cells, myeloid fraction 0, 0.1, ..., 1,
    25 draws, without replacement inside a draw.
    Score: singscore of the eight myeloid markers on the summed counts.
    Decision number: Spearman of that score with the known fraction.
    Neuronal markers are not given a correlation from 22 neurons.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, pearsonr

sys.path.insert(0, r"G:\James Zou Project\operative_state")
import run_gates as rg  # noqa: E402

H5 = Path(r"G:\James Zou Project\operative_state\data_neftel\gbmap_core.h5ad")
CELLS = Path(r"G:\James Zou Project\operative_state\results\malignant_cells\cell_scores.csv")
OUT = Path(r"G:\James Zou Project\operative_state\results\depth_marker")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 11
MIN_CELLS = 20
FLOOR = 0.20
PSEUDO_N = 200
N_REP = 25
FRACTIONS = np.round(np.arange(0, 1.01, 0.1), 1)
RESERVOIR = {
    "microglial cell": 4000,
    "macrophage": 4000,
    "monocyte": 2000,
    "malignant cell": 4000,
}
MYELOID_LABELS = ("microglial cell", "macrophage", "monocyte")


def decode(dataset):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in dataset[:]]


def categorical(group):
    return decode(group["categories"]), np.asarray(group["codes"])


def partial_pair(r_xy, r_xz, r_yz):
    denom = np.sqrt((1.0 - r_xz * r_xz) * (1.0 - r_yz * r_yz))
    if not np.isfinite(denom) or denom < 1e-8:
        return np.nan
    return float((r_xy - r_xz * r_yz) / denom)


def partial_two(h, m, c, d):
    def rho(a, b):
        if np.unique(a).size < 2 or np.unique(b).size < 2:
            return np.nan
        return float(spearmanr(a, b).statistic)

    r_hm, r_hc, r_hd = rho(h, m), rho(h, c), rho(h, d)
    r_mc, r_md, r_cd = rho(m, c), rho(m, d), rho(c, d)
    if not all(np.isfinite(v) for v in (r_hm, r_hc, r_hd, r_mc, r_md, r_cd)):
        return np.nan
    r_hm_c = partial_pair(r_hm, r_hc, r_mc)
    r_hd_c = partial_pair(r_hd, r_hc, r_cd)
    r_md_c = partial_pair(r_md, r_mc, r_cd)
    if not all(np.isfinite(v) for v in (r_hm_c, r_hd_c, r_md_c)):
        return np.nan
    return partial_pair(r_hm_c, r_hd_c, r_md_c)


def singscore(totals: np.ndarray, gene_index: np.ndarray) -> float:
    n = totals.size
    ranks = rankdata(totals, method="average")
    centered = (ranks[gene_index] - (n + 1) / 2.0) / n
    return float(centered.mean())


def main() -> None:
    cells = pd.read_csv(
        CELLS, usecols=["hypoxia_f1", "mes1_f1", "cell_cycle", "donor", "author"]
    )
    rng = np.random.default_rng(SEED)
    with h5py.File(H5, "r") as handle:
        type_names, type_codes = categorical(handle["obs"]["cell_type"])
        donor_names, donor_codes = categorical(handle["obs"]["donor_id"])
        gene_names, gene_codes = categorical(handle["var"]["feature_name"])
        symbols = [gene_names[int(c)] for c in gene_codes]
        symbol_index = {g: i for i, g in enumerate(symbols)}
        missing = [g for g in rg.MYELOID if g not in symbol_index]
        if missing:
            raise SystemExit(f"myeloid markers absent: {missing}")
        myeloid_genes = np.array([symbol_index[g] for g in rg.MYELOID], dtype=np.int32)
        malignant_code = type_names.index("malignant cell")
        malignant = np.flatnonzero(type_codes == malignant_code)
        file_donors = [donor_names[int(donor_codes[i])] for i in malignant]
        if file_donors != cells["donor"].tolist():
            raise SystemExit("cell_scores.csv is not in h5ad malignant order; depth not joined")
        label_code = {name: i for i, name in enumerate(type_names)}
        keep_codes = {label_code[name] for name in RESERVOIR}
        seen = {name: 0 for name in RESERVOIR}
        held_idx = {name: [] for name in RESERVOIR}
        held_val = {name: [] for name in RESERVOIR}
        n_malignant = len(malignant)
        total_counts = np.zeros(n_malignant, dtype=np.float64)
        n_detected = np.zeros(n_malignant, dtype=np.float64)
        slot = np.full(len(type_codes), -1, dtype=np.int32)
        slot[malignant] = np.arange(n_malignant)
        indptr = np.asarray(handle["X"]["indptr"])
        indices = handle["X"]["indices"]
        data = handle["X"]["data"]
        n_cells = len(type_codes)
        neuron_n = int(np.sum(type_codes == label_code["neuron"]))
        for start in range(0, n_cells, 4000):
            stop = min(start + 4000, n_cells)
            a = int(indptr[start])
            b = int(indptr[stop])
            row_index = np.asarray(indices[a:b])
            row_value = np.asarray(data[a:b], dtype=np.float64)
            local = indptr[start:stop + 1] - a
            for offset, cell in enumerate(range(start, stop)):
                lo = int(local[offset])
                hi = int(local[offset + 1])
                genes = row_index[lo:hi]
                values = row_value[lo:hi]
                place = int(slot[cell])
                if place >= 0:
                    total_counts[place] = float(values.sum()) if hi > lo else 0.0
                    n_detected[place] = float(hi - lo)
                code = int(type_codes[cell])
                if code not in keep_codes:
                    continue
                name = type_names[code]
                cap = RESERVOIR[name]
                seen[name] += 1
                if len(held_idx[name]) < cap:
                    held_idx[name].append(genes.copy())
                    held_val[name].append(values.copy())
                else:
                    j = int(rng.integers(0, seen[name]))
                    if j < cap:
                        held_idx[name][j] = genes.copy()
                        held_val[name][j] = values.copy()
            if start % 40000 == 0:
                print("scanned", stop, flush=True)
        n_genes = len(symbols)

    cells = cells.copy()
    cells["log_counts"] = np.log1p(total_counts)
    cells["log_detected"] = np.log1p(n_detected)
    rows = []
    for (donor, author), sub in cells.groupby(["donor", "author"], sort=False):
        sub = sub.dropna()
        if len(sub) < MIN_CELLS:
            continue
        h = sub["hypoxia_f1"].to_numpy()
        m = sub["mes1_f1"].to_numpy()
        c = sub["cell_cycle"].to_numpy()
        rows.append({
            "donor": donor,
            "author": author,
            "n_cells": int(len(sub)),
            "partial_counts": partial_two(h, m, c, sub["log_counts"].to_numpy()),
            "partial_detected": partial_two(h, m, c, sub["log_detected"].to_numpy()),
        })
    donors = pd.DataFrame(rows)
    donors.to_csv(OUT / "donor_depth.csv", index=False)

    def mean_of(frame, col):
        v = frame[col].to_numpy()
        v = v[np.isfinite(v)]
        return {
            "n": int(v.size),
            "mean": float(np.mean(v)) if v.size else None,
            "se": float(np.std(v, ddof=1) / np.sqrt(v.size)) if v.size > 1 else None,
            "fraction_at_least_0.20": float(np.mean(v >= FLOOR)) if v.size else None,
            "holds": bool(v.size >= 8 and np.mean(v) >= FLOOR),
        }

    entered = donors.dropna(subset=["partial_counts"])
    depth = {
        "counts_all": mean_of(entered, "partial_counts"),
        "counts_without_neftel": mean_of(entered.loc[entered.author != "Neftel2019"], "partial_counts"),
        "detected_all": mean_of(donors.dropna(subset=["partial_detected"]), "partial_detected"),
        "detected_without_neftel": mean_of(
            donors.dropna(subset=["partial_detected"]).loc[lambda d: d.author != "Neftel2019"],
            "partial_detected",
        ),
    }
    depth["primary_holds"] = bool(
        depth["counts_all"]["holds"] and depth["counts_without_neftel"]["holds"]
    )

    # Myeloid mixtures from the reservoir. Neurons are not mixed.
    myeloid_idx = held_idx["microglial cell"] + held_idx["macrophage"] + held_idx["monocyte"]
    myeloid_val = held_val["microglial cell"] + held_val["macrophage"] + held_val["monocyte"]
    mal_idx, mal_val = held_idx["malignant cell"], held_val["malignant cell"]
    records = []
    for frac in FRACTIONS:
        n_my = int(round(frac * PSEUDO_N))
        n_mal = PSEUDO_N - n_my
        for rep in range(N_REP):
            totals = np.zeros(n_genes, dtype=np.float64)
            if n_my:
                pick = rng.choice(len(myeloid_idx), size=n_my, replace=False)
                for j in pick:
                    totals[myeloid_idx[j]] += myeloid_val[j]
            if n_mal:
                pick = rng.choice(len(mal_idx), size=n_mal, replace=False)
                for j in pick:
                    totals[mal_idx[j]] += mal_val[j]
            records.append({"fraction": float(frac), "myeloid_singscore": singscore(totals, myeloid_genes)})
    mixes = pd.DataFrame(records)
    mixes.to_csv(OUT / "myeloid_mixtures.csv", index=False)
    spear = float(spearmanr(mixes["fraction"], mixes["myeloid_singscore"]).statistic)
    pears = float(pearsonr(mixes["fraction"], mixes["myeloid_singscore"]).statistic)
    marker = {
        "neuron_n": neuron_n,
        "neuron_correlation": None,
        "neuron_reason": "22 neurons; pool is below 100 cells, so no mixture was built",
        "myeloid_labels": list(MYELOID_LABELS),
        "reservoir_filled": {k: len(held_idx[k]) for k in RESERVOIR},
        "pseudobulk_cells": PSEUDO_N,
        "n_draws": int(len(mixes)),
        "myeloid_spearman": spear,
        "myeloid_pearson": pears,
        "decision_correlation": spear,
    }
    # Locked reading of the noise curve. Not a refit of the curve.
    # excess 0.801 at 0.50, 0.512 at 0.75, 0.254 at 0.90
    if spear >= 0.90:
        spatial_reading = "residual_above_noise_band"
    elif spear <= 0.80:
        spatial_reading = "demote_spatial"
    else:
        # linear interpolation of excess between the locked grid points 0.75 and 0.90
        noise_at = 0.5117024417939027 + (spear - 0.75) / 0.15 * (0.25385365670960347 - 0.5117024417939027)
        spatial_reading = "demote_spatial" if 0.546 <= noise_at else "residual_above_interpolated_noise"
        marker["interpolated_noise_excess"] = float(noise_at)
    marker["spatial_reading"] = spatial_reading
    verdict = {"depth": depth, "marker": marker, "original_within_donor_overwritten": False}
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
