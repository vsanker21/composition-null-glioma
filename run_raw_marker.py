# -*- coding: utf-8 -*-
"""Recompute depth and myeloid mixtures from raw counts.

GBmap X is a normalized matrix. Within a cell its ranks match the raw counts,
so a singscore is unchanged, and the nonzero pattern matches, so the number of
detected genes is unchanged. A sum across cells is not a count. This script
reads raw/X. It does not overwrite results/within_donor.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import run_depth_and_marker as dm

H5 = dm.H5
CELLS = dm.CELLS
OUT = dm.OUT
ARCHIVE = OUT / "normalized_X"
ARCHIVE.mkdir(parents=True, exist_ok=True)
for name in ("verdict.json", "donor_depth.csv", "myeloid_mixtures.csv"):
    source = OUT / name
    target = ARCHIVE / name
    if source.exists() and not target.exists():
        shutil.copyfile(source, target)

previous = pd.read_csv(ARCHIVE / "donor_depth.csv")


def main() -> None:
    cells = pd.read_csv(CELLS, usecols=["hypoxia_f1", "mes1_f1", "cell_cycle", "donor", "author"])
    rng = np.random.default_rng(dm.SEED)
    with h5py.File(H5, "r") as handle:
        type_names, type_codes = dm.categorical(handle["obs"]["cell_type"])
        donor_names, donor_codes = dm.categorical(handle["obs"]["donor_id"])
        gene_names, gene_codes = dm.categorical(handle["var"]["feature_name"])
        symbols = [gene_names[int(c)] for c in gene_codes]
        symbol_index = {g: i for i, g in enumerate(symbols)}
        myeloid_genes = np.array([symbol_index[g] for g in dm.rg.MYELOID], dtype=np.int32)
        malignant_code = type_names.index("malignant cell")
        malignant = np.flatnonzero(type_codes == malignant_code)
        file_donors = [donor_names[int(donor_codes[i])] for i in malignant]
        if file_donors != cells["donor"].tolist():
            raise SystemExit("cell_scores.csv is not in h5ad malignant order")
        if not np.array_equal(np.asarray(handle["X"]["indptr"]), np.asarray(handle["raw"]["X"]["indptr"])):
            raise SystemExit("normalized and raw nonzero patterns differ; detected genes are not the same")
        label_code = {name: i for i, name in enumerate(type_names)}
        keep_codes = {label_code[name] for name in dm.RESERVOIR}
        seen = {name: 0 for name in dm.RESERVOIR}
        held_idx = {name: [] for name in dm.RESERVOIR}
        held_val = {name: [] for name in dm.RESERVOIR}
        n_malignant = len(malignant)
        total_counts = np.zeros(n_malignant, dtype=np.float64)
        n_detected = np.zeros(n_malignant, dtype=np.float64)
        slot = np.full(len(type_codes), -1, dtype=np.int32)
        slot[malignant] = np.arange(n_malignant)
        indptr = np.asarray(handle["raw"]["X"]["indptr"])
        indices = handle["raw"]["X"]["indices"]
        data = handle["raw"]["X"]["data"]
        n_cells = len(type_codes)
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
                cap = dm.RESERVOIR[name]
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
        if len(sub) < dm.MIN_CELLS:
            continue
        rows.append({
            "donor": donor,
            "author": author,
            "n_cells": int(len(sub)),
            "partial_counts": dm.partial_two(
                sub["hypoxia_f1"].to_numpy(), sub["mes1_f1"].to_numpy(),
                sub["cell_cycle"].to_numpy(), sub["log_counts"].to_numpy(),
            ),
            "partial_detected": dm.partial_two(
                sub["hypoxia_f1"].to_numpy(), sub["mes1_f1"].to_numpy(),
                sub["cell_cycle"].to_numpy(), sub["log_detected"].to_numpy(),
            ),
        })
    donors = pd.DataFrame(rows)
    merged = donors.merge(previous, on=["donor", "author", "n_cells"], suffixes=("", "_old"))
    delta = np.max(np.abs(merged["partial_detected"] - merged["partial_detected_old"]))
    print("detected partial max change", delta, flush=True)
    if delta > 1e-8:
        raise SystemExit("detected-gene partial changed; the nonzero pattern was not what it seemed")
    donors.to_csv(OUT / "donor_depth.csv", index=False)

    def mean_of(frame, col):
        v = frame[col].to_numpy()
        v = v[np.isfinite(v)]
        return {
            "n": int(v.size),
            "mean": float(np.mean(v)) if v.size else None,
            "se": float(np.std(v, ddof=1) / np.sqrt(v.size)) if v.size > 1 else None,
            "fraction_at_least_0.20": float(np.mean(v >= dm.FLOOR)) if v.size else None,
            "n_below_0.20": int(np.sum(v < dm.FLOOR)),
            "holds": bool(v.size >= 8 and np.mean(v) >= dm.FLOOR),
        }

    entered = donors.dropna(subset=["partial_counts"])
    depth = {
        "matrix": "raw counts",
        "counts_all": mean_of(entered, "partial_counts"),
        "counts_without_neftel": mean_of(entered.loc[entered.author != "Neftel2019"], "partial_counts"),
        "detected_all": mean_of(donors.dropna(subset=["partial_detected"]), "partial_detected"),
        "detected_without_neftel": mean_of(
            donors.dropna(subset=["partial_detected"]).loc[lambda d: d.author != "Neftel2019"],
            "partial_detected",
        ),
    }
    myeloid_idx = held_idx["microglial cell"] + held_idx["macrophage"] + held_idx["monocyte"]
    myeloid_val = held_val["microglial cell"] + held_val["macrophage"] + held_val["monocyte"]
    mal_idx, mal_val = held_idx["malignant cell"], held_val["malignant cell"]
    records = []
    for frac in dm.FRACTIONS:
        n_my = int(round(float(frac) * dm.PSEUDO_N))
        n_mal = dm.PSEUDO_N - n_my
        for _rep in range(dm.N_REP):
            totals = np.zeros(n_genes, dtype=np.float64)
            if n_my:
                pick = rng.choice(len(myeloid_idx), size=n_my, replace=False)
                for j in pick:
                    totals[myeloid_idx[j]] += myeloid_val[j]
            if n_mal:
                pick = rng.choice(len(mal_idx), size=n_mal, replace=False)
                for j in pick:
                    totals[mal_idx[j]] += mal_val[j]
            records.append({"fraction": float(frac), "myeloid_singscore": dm.singscore(totals, myeloid_genes)})
    mixes = pd.DataFrame(records)
    mixes.to_csv(OUT / "myeloid_mixtures.csv", index=False)
    spear = float(spearmanr(mixes["fraction"], mixes["myeloid_singscore"]).statistic)
    pears = float(pearsonr(mixes["fraction"], mixes["myeloid_singscore"]).statistic)
    means = mixes.groupby("fraction")["myeloid_singscore"].agg(["mean", "std"])
    print(means.to_string(), flush=True)
    verdict = {
        "depth": depth,
        "marker": {
            "matrix": "raw counts",
            "myeloid_spearman": spear,
            "myeloid_pearson": pears,
            "decision_correlation": "spearman",
            "reservoir_filled": {k: len(held_idx[k]) for k in dm.RESERVOIR},
            "normalized_X_archived": True,
        },
        "original_within_donor_overwritten": False,
    }
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
