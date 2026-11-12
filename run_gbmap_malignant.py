"""Malignant-cell test on the locked GBmap label. Rules are in FIELD_PROTOCOL.md.

cell_type == malignant cell. One row per donor. Expression is not read until the
label and the donor filter are applied.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import rankdata

sys.path.insert(0, str(Path(r"G:\James Zou Project\operative_state")))
import run_full_list as fl  # noqa: E402
import run_gates as rg  # noqa: E402

OUT = Path(r"G:\James Zou Project\operative_state\results\malignant_cells")
OUT.mkdir(parents=True, exist_ok=True)
H5 = Path(r"G:\James Zou Project\operative_state\data_neftel\gbmap_core.h5ad")

COEF_FLOOR = 0.20
P_FLOOR = 0.10
MIN_CELLS = 20
MIN_TUMORS = 8
MALIGNANT = "malignant cell"
CELL_CYCLE = ("HALLMARK_E2F_TARGETS", "HALLMARK_G2M_CHECKPOINT")


def decode_categories(dataset) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in dataset[:]]


def categorical(group) -> tuple[list[str], np.ndarray]:
    return decode_categories(group["categories"]), np.asarray(group["codes"])


def gene_names(handle) -> list[str]:
    group = handle["var"]["feature_name"]
    categories, codes = categorical(group)
    return [categories[int(code)] for code in codes]


def fit_arm(tumors: pd.DataFrame) -> dict:
    d = tumors[["mes1_f1", "hypoxia_f1", "cell_cycle"]].dropna().copy()
    if len(d) < MIN_TUMORS:
        return {"status": "uninterpretable", "n_tumors": int(len(d))}
    for col in d.columns:
        d[col] = rg.zscore(d[col])
    unadj = smf.ols("mes1_f1 ~ hypoxia_f1", data=d).fit(cov_type="HC3")
    adj = smf.ols("mes1_f1 ~ hypoxia_f1 + cell_cycle", data=d).fit(cov_type="HC3")
    coef = float(adj.params["hypoxia_f1"])
    pval = float(adj.pvalues["hypoxia_f1"])
    if coef >= COEF_FLOOR and coef > 0 and pval < P_FLOOR:
        status = "holds"
    elif coef < COEF_FLOOR or coef <= 0:
        status = "niche_cooccupancy"
    else:
        status = "inconclusive"
    low, high = adj.conf_int().loc["hypoxia_f1"].tolist()
    return {
        "status": status,
        "n_tumors": int(adj.nobs),
        "unadjusted_coef": float(unadj.params["hypoxia_f1"]),
        "unadjusted_p": float(unadj.pvalues["hypoxia_f1"]),
        "adjusted_coef": coef,
        "adjusted_se": float(adj.bse["hypoxia_f1"]),
        "adjusted_p": pval,
        "adjusted_ci_low": float(low),
        "adjusted_ci_high": float(high),
        "r2_adjusted": float(adj.rsquared),
    }


def score_malignant(handle, gene_index: dict[str, int], programs: dict[str, list[int]]) -> pd.DataFrame:
    labels, codes = categorical(handle["obs"]["cell_type"])
    if MALIGNANT not in labels:
        raise SystemExit("Locked malignant token is absent")
    malignant_code = labels.index(MALIGNANT)
    malignant = np.flatnonzero(codes == malignant_code)
    donors, donor_codes = categorical(handle["obs"]["donor_id"])
    authors, author_codes = categorical(handle["obs"]["author"])
    indptr = np.asarray(handle["X"]["indptr"])
    n_genes = len(gene_index)
    program_names = list(programs)
    wanted = {}
    for name, positions in programs.items():
        for position in positions:
            wanted.setdefault(position, []).append(name)
    n_defined = {name: len(positions) for name, positions in programs.items()}
    sums = {name: np.zeros(len(malignant), dtype=np.float64) for name in program_names}
    # Map matrix row -> position in the malignant array. -1 if not malignant.
    slot = np.full(len(codes), -1, dtype=np.int32)
    slot[malignant] = np.arange(len(malignant))
    indices = handle["X"]["indices"]
    data = handle["X"]["data"]
    center = (n_genes + 1) / 2.0
    chunk = 4000
    n_cells = len(codes)
    for start in range(0, n_cells, chunk):
        stop = min(start + chunk, n_cells)
        if not np.any(slot[start:stop] >= 0):
            continue
        a = int(indptr[start])
        b = int(indptr[stop])
        row_index = np.asarray(indices[a:b])
        row_value = np.asarray(data[a:b])
        local = indptr[start : stop + 1] - a
        for offset, cell in enumerate(range(start, stop)):
            place = int(slot[cell])
            if place < 0:
                continue
            lo = int(local[offset])
            hi = int(local[offset + 1])
            genes = row_index[lo:hi]
            values = row_value[lo:hi]
            n_nonzero = len(genes)
            n_zero = n_genes - n_nonzero
            zero_centered = ((n_zero + 1) / 2.0 - center) / n_genes
            for name in program_names:
                sums[name][place] = zero_centered * n_defined[name]
            if n_nonzero == 0:
                continue
            ranks = rankdata(values, method="average") + n_zero
            centered = (ranks - center) / n_genes
            for gene, centered_rank, base in zip(genes, centered, [zero_centered] * n_nonzero):
                owners = wanted.get(int(gene))
                if not owners:
                    continue
                for name in owners:
                    sums[name][place] += centered_rank - base
        if start % 40000 == 0:
            print("scored through cell", stop, flush=True)
    frame = pd.DataFrame({name: sums[name] / n_defined[name] for name in program_names})
    frame["donor"] = [donors[int(donor_codes[i])] for i in malignant]
    frame["author"] = [authors[int(author_codes[i])] for i in malignant]
    frame["cell_cycle"] = (frame["e2f"] + frame["g2m"]) / 2.0
    return frame


def main() -> None:
    with h5py.File(H5, "r") as handle:
        names = gene_names(handle)
        if len(names) != len(set(names)):
            raise SystemExit("Duplicate gene symbols. Test not run.")
        universe = set(names)
        previous = fl.OUT
        fl.OUT = OUT
        fl.build_sets(universe)
        banned = set(fl.SETS["hypoxia_f1"]) | set(fl.SETS["mes1_f1"]) | set(rg.NEURONAL) | set(rg.MYELOID)
        cycle = {}
        for line in fl.GMT.read_text(encoding="utf-8").splitlines():
            parts = line.rstrip("\n").split("\t")
            if parts[0] in CELL_CYCLE:
                kept = [g for g in parts[2:] if g not in banned and g in universe]
                if len(kept) < 20:
                    raise SystemExit(f"{parts[0]} left with {len(kept)} genes")
                cycle[parts[0]] = kept
        fl.OUT = previous
        index = {gene: i for i, gene in enumerate(names)}
        programs = {
            "hypoxia_f1": [index[g] for g in fl.SETS["hypoxia_f1"] if g in index],
            "mes1_f1": [index[g] for g in fl.SETS["mes1_f1"] if g in index],
            "e2f": [index[g] for g in cycle["HALLMARK_E2F_TARGETS"]],
            "g2m": [index[g] for g in cycle["HALLMARK_G2M_CHECKPOINT"]],
        }
        if min(len(v) for v in programs.values()) < 20:
            raise SystemExit("A program fell below 20 genes in this universe")
        print({k: len(v) for k, v in programs.items()}, flush=True)
        cells = score_malignant(handle, index, programs)
    cells.to_csv(OUT / "cell_scores.csv", index=False)
    grouped = cells.groupby(["donor", "author"], as_index=False).agg(
        hypoxia_f1=("hypoxia_f1", "mean"),
        mes1_f1=("mes1_f1", "mean"),
        cell_cycle=("cell_cycle", "mean"),
        n_cells=("hypoxia_f1", "size"),
    )
    tumors = grouped.loc[grouped["n_cells"] >= MIN_CELLS].copy()
    tumors.to_csv(OUT / "tumor_means.csv", index=False)
    primary = fit_arm(tumors)
    primary["n_cells_used"] = int(tumors["n_cells"].sum()) if len(tumors) else 0
    without_neftel = fit_arm(tumors.loc[tumors["author"] != "Neftel2019"])
    by_author = []
    for author, sub in tumors.groupby("author"):
        if len(sub) < MIN_TUMORS:
            continue
        fit = fit_arm(sub)
        fit["author"] = author
        by_author.append(fit)
    pd.DataFrame(by_author).to_csv(OUT / "by_author.csv", index=False)
    upgrade = primary["status"] == "holds" and without_neftel["status"] == "holds"
    if primary["status"] == "niche_cooccupancy":
        sentence = "niche_cooccupancy"
    elif primary["status"] == "holds" and without_neftel["status"] != "holds":
        sentence = "depends_on_neftel_discovery_cohort"
    elif upgrade:
        sentence = "cell_intrinsic"
    else:
        sentence = primary["status"]
    verdict = {
        "label": "cell_type == malignant cell",
        "primary": primary,
        "without_neftel2019": without_neftel,
        "sentence": sentence,
        "cell_intrinsic_upgrade": upgrade,
        "recording_study_opened": False,
        "circularity": "MES1 was learned on the Neftel malignant cells, which are inside this atlas. The leave-Neftel-out regression is required for the upgrade.",
    }
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
