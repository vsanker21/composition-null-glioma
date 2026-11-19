# -*- coding: utf-8 -*-
"""Neuronal-marker mixtures across two sources. Rules written before the correlation.

Neurons: Allen Institute adult human MTG SMART-seq taxonomy matrix
(Human_MTG_SMART_seq_08082025.h5ad). class is GABAergic or Glutamatergic.
Non-neuronal nuclei are not used. The 22 GBmap neurons are not used.
Background: GBmap cells labeled malignant cell.
Gene universe: the intersection of the two symbol lists. All eight neuronal
markers must be present or the test stops.
Reservoirs: 4000 neurons and 4000 malignant cells, each drawn with seed 11.
Pseudobulks: 200 cells, neuronal fraction 0 to 1 in steps of 0.1, 25 draws,
without replacement inside a draw. Each design draws with its own
generator seeded 11.
Score: singscore of RBFOX3, SNAP25, SYP, SYN1, STMN2, ENO2, TUBB3, MAP2.
Ranks include zeros. No batch correction.

The raw sum is stored and is not the decision. Median SMART-seq depth is
about 2.6 million reads, so a raw sum confounds cell fraction with read
fraction. The decision downsamples every retained cell to T counts on the
shared genes, where T is the minimum of the two reservoir medians. Cells
below T are left out of that design. It is not run if either retained pool
has fewer than 200 cells. Downsampling uses seed 17.

Reading of the decision Spearman, fixed before it is known:
    >= 0.90: the markers order a known neuronal fraction at equal depth.
    <= 0.80: they do not, and the neuronal axis is not calibrated.
    between: report the number and do not call the axis calibrated.
The raw-sum Spearman cannot change that reading.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr

MTG = Path(r"G:\James Zou Project\operative_state\data_cortex\Human_MTG_SMART_seq.h5ad")
GB = Path(r"G:\James Zou Project\operative_state\data_neftel\gbmap_core.h5ad")
OUT = Path(r"G:\James Zou Project\operative_state\results\neuron_marker")
OUT.mkdir(parents=True, exist_ok=True)

MARKERS = ["RBFOX3", "SNAP25", "SYP", "SYN1", "STMN2", "ENO2", "TUBB3", "MAP2"]
SEED = 11
N_EACH = 4000
PSEUDO_N = 200
N_REP = 25
FRACTIONS = np.round(np.arange(0, 1.01, 0.1), 1)
NEURON_CLASSES = ("GABAergic", "Glutamatergic")


def decode(dataset):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in dataset[:]]


def categorical(group):
    return decode(group["categories"]), np.asarray(group["codes"])


def singscore(totals: np.ndarray, gene_index: np.ndarray) -> float:
    n = totals.size
    ranks = rankdata(totals, method="average")
    centered = (ranks[gene_index] - (n + 1) / 2.0) / n
    return float(centered.mean())


def reservoir_malignant(inter_index: dict[str, int], n_genes: int):
    rng = np.random.default_rng(SEED)
    held_i, held_v = [], []
    seen = 0
    with h5py.File(GB, "r") as handle:
        type_names, type_codes = categorical(handle["obs"]["cell_type"])
        gene_names, gene_codes = categorical(handle["var"]["feature_name"])
        symbols = [gene_names[int(c)] for c in gene_codes]
        gb_to_inter = np.full(len(symbols), -1, dtype=np.int32)
        for symbol, slot in inter_index.items():
            gb_to_inter[symbols.index(symbol)] = slot
        missing = [g for g in MARKERS if inter_index.get(g) is None]
        if missing:
            raise SystemExit(f"markers missing from the intersection: {missing}")
        malignant = type_names.index("malignant cell")
        indptr = np.asarray(handle["raw"]["X"]["indptr"])
        if indptr.size != type_codes.size + 1:
            raise SystemExit("GBmap raw counts are not cells by genes")
        indices = handle["raw"]["X"]["indices"]
        data = handle["raw"]["X"]["data"]
        n_cells = type_codes.size
        for start in range(0, n_cells, 8000):
            stop = min(start + 8000, n_cells)
            a = int(indptr[start])
            b = int(indptr[stop])
            row_index = np.asarray(indices[a:b])
            row_value = np.asarray(data[a:b], dtype=np.float64)
            local = indptr[start:stop + 1] - a
            for offset, cell in enumerate(range(start, stop)):
                if int(type_codes[cell]) != malignant:
                    continue
                lo = int(local[offset])
                hi = int(local[offset + 1])
                mapped = gb_to_inter[row_index[lo:hi]]
                keep = mapped >= 0
                seen += 1
                packed_i = mapped[keep].astype(np.int32, copy=False)
                packed_v = row_value[lo:hi][keep]
                if len(held_i) < N_EACH:
                    held_i.append(packed_i)
                    held_v.append(packed_v)
                else:
                    j = int(rng.integers(0, seen))
                    if j < N_EACH:
                        held_i[j] = packed_i
                        held_v[j] = packed_v
            if start % 80000 == 0:
                print("gbmap", stop, "malignant seen", seen, flush=True)
    dense = np.zeros((len(held_i), n_genes), dtype=np.float64)
    for i, (gi, gv) in enumerate(zip(held_i, held_v)):
        if gi.size:
            dense[i, gi] += gv
    return dense, seen


def neuron_matrix(inter_symbols: list[str]):
    rng = np.random.default_rng(SEED)
    with h5py.File(MTG, "r") as handle:
        symbols = decode(handle["raw"]["var"]["gene"])
        classes, codes = categorical(handle["obs"]["class"])
        class_ok = {classes.index(name) for name in NEURON_CLASSES}
        neuron_rows = np.flatnonzero(np.isin(codes, list(class_ok)))
        class_of = {name: int(np.sum(codes[neuron_rows] == classes.index(name))) for name in NEURON_CLASSES}
        chosen = np.sort(rng.choice(neuron_rows, size=N_EACH, replace=False))
        chosen_class = {
            name: int(np.sum(codes[chosen] == classes.index(name))) for name in NEURON_CLASSES
        }
        col_of = {g: i for i, g in enumerate(symbols)}
        cols = np.array([col_of[g] for g in inter_symbols], dtype=np.int64)
        slot = np.full(codes.size, -1, dtype=np.int32)
        slot[chosen] = np.arange(N_EACH)
        indptr = np.asarray(handle["raw"]["X"]["indptr"])
        indices = np.asarray(handle["raw"]["X"]["indices"])
        data = np.asarray(handle["raw"]["X"]["data"])
        print("mtg counts loaded", data.size, flush=True)
        matrix = np.zeros((N_EACH, len(inter_symbols)), dtype=np.float64)
        for out_j, col in enumerate(cols):
            a = int(indptr[col])
            b = int(indptr[col + 1])
            rows = indices[a:b]
            slots = slot[rows]
            keep = slots >= 0
            if np.any(keep):
                matrix[slots[keep], out_j] = data[a:b][keep]
        del indices, data
    return matrix, class_of, chosen_class


def downsample(counts: np.ndarray, t: int, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros(counts.shape, dtype=np.float64)
    for i in range(counts.shape[0]):
        total = int(counts[i].sum())
        if total < t:
            raise SystemExit(f"cell {i} has {total} counts, below T={t}")
        out[i] = rng.multivariate_hypergeometric(counts[i], t)
    return out


def mixtures(neurons: np.ndarray, malignant: np.ndarray, marker_index: np.ndarray, rng: np.random.Generator, design: str):
    rows = []
    for frac in FRACTIONS:
        n_neu = int(round(float(frac) * PSEUDO_N))
        n_mal = PSEUDO_N - n_neu
        for _rep in range(N_REP):
            totals = np.zeros(neurons.shape[1], dtype=np.float64)
            if n_neu:
                pick = rng.choice(neurons.shape[0], size=n_neu, replace=False)
                totals += neurons[pick].sum(axis=0)
            if n_mal:
                pick = rng.choice(malignant.shape[0], size=n_mal, replace=False)
                totals += malignant[pick].sum(axis=0)
            rows.append({
                "fraction": float(frac),
                "design": design,
                "neuronal_singscore": singscore(totals, marker_index),
            })
    return rows


def summarize(frame: pd.DataFrame) -> dict:
    spear = float(spearmanr(frame["fraction"], frame["neuronal_singscore"]).statistic)
    pears = float(pearsonr(frame["fraction"], frame["neuronal_singscore"]).statistic)
    means = frame.groupby("fraction")["neuronal_singscore"].mean()
    return {
        "spearman": spear,
        "pearson": pears,
        "mean_at_0": float(means.loc[0.0]),
        "mean_at_0.2": float(means.loc[0.2]),
        "mean_at_1": float(means.loc[1.0]),
        "n_draws": int(len(frame)),
    }


def main() -> None:
    with h5py.File(MTG, "r") as handle:
        mtg_symbols = decode(handle["raw"]["var"]["gene"])
    with h5py.File(GB, "r") as handle:
        gene_names, gene_codes = categorical(handle["var"]["feature_name"])
        gb_symbols = [gene_names[int(c)] for c in gene_codes]
    inter = sorted(set(mtg_symbols) & set(gb_symbols))
    missing = [g for g in MARKERS if g not in inter]
    if missing:
        raise SystemExit(f"markers absent from the intersection: {missing}")
    inter_index = {g: i for i, g in enumerate(inter)}
    marker_index = np.array([inter_index[g] for g in MARKERS], dtype=np.int64)
    print("intersection", len(inter), flush=True)
    neurons, class_of, chosen_class = neuron_matrix(inter)
    print("neurons", neurons.shape, "class pool", class_of, "reservoir", chosen_class, flush=True)
    malignant, n_malignant = reservoir_malignant(inter_index, len(inter))
    print("malignant reservoir", malignant.shape, "seen", n_malignant, flush=True)
    neu_counts = np.rint(neurons).astype(np.int64)
    mal_counts = np.rint(malignant).astype(np.int64)
    if max(float(np.max(np.abs(neurons - neu_counts))), float(np.max(np.abs(malignant - mal_counts)))) > 1e-6:
        raise SystemExit("shared-gene values are not integer counts")
    neu_depth = neu_counts.sum(axis=1)
    mal_depth = mal_counts.sum(axis=1)
    t = int(min(np.median(neu_depth), np.median(mal_depth)))
    neu_keep = neu_depth >= t
    mal_keep = mal_depth >= t
    print("T", t, "kept", int(neu_keep.sum()), int(mal_keep.sum()), flush=True)
    raw_rows = mixtures(neurons, malignant, marker_index, np.random.default_rng(SEED), "raw_sum")
    if int(neu_keep.sum()) < PSEUDO_N or int(mal_keep.sum()) < PSEUDO_N:
        equal_rows = []
        equal = None
        equal_reason = "fewer than 200 cells remained on one side"
    else:
        down_rng = np.random.default_rng(17)
        neu_eq = downsample(neu_counts[neu_keep], t, down_rng)
        mal_eq = downsample(mal_counts[mal_keep], t, down_rng)
        print("downsampled", flush=True)
        equal_rows = mixtures(neu_eq, mal_eq, marker_index, np.random.default_rng(SEED), "equal_depth")
        equal = summarize(pd.DataFrame(equal_rows))
        equal_reason = None
    raw = summarize(pd.DataFrame(raw_rows))
    decision = None if equal is None else equal["spearman"]
    if decision is None:
        reading = "not_run"
    elif decision >= 0.90:
        reading = "markers_order_fraction"
    elif decision <= 0.80:
        reading = "not_calibrated"
    else:
        reading = "between"
    table = pd.DataFrame(raw_rows + equal_rows)
    table.to_csv(OUT / "mixtures.csv", index=False)
    verdict = {
        "source": "Allen adult human MTG SMART-seq, control nuclei, eight donors",
        "file": MTG.name,
        "neuron_classes": list(NEURON_CLASSES),
        "neuron_class_counts": class_of,
        "reservoir_class_counts": chosen_class,
        "gbmap_neurons_used": False,
        "n_malignant_seen": int(n_malignant),
        "intersection_genes": len(inter),
        "markers": MARKERS,
        "median_shared_counts_neurons": float(np.median(neu_depth)),
        "median_shared_counts_malignant": float(np.median(mal_depth)),
        "equal_depth_T": t,
        "equal_depth_kept_neurons": int(neu_keep.sum()),
        "equal_depth_kept_malignant": int(mal_keep.sum()),
        "raw_sum": raw,
        "equal_depth": equal,
        "equal_depth_reason": equal_reason,
        "decision": "equal_depth_spearman",
        "decision_spearman": decision,
        "reading": reading,
        "batch_correction": False,
    }
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
