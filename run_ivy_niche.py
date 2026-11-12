"""Ivy reference-histology arm. Rules are in NICHE_PROTOCOL.md.

No threshold here is chosen from a coefficient.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(r"G:\James Zou Project\operative_state")))
import run_full_list as fl  # noqa: E402
import run_gates as rg  # noqa: E402

OUT = Path(r"G:\James Zou Project\operative_state\results\niche_ivy")
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path(r"G:\James Zou Project\operative_state\data_ivy")
CACHE.mkdir(parents=True, exist_ok=True)


def local_table(name: str) -> Path:
    path = CACHE / name
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(name)
    return path


def load_meta() -> pd.DataFrame:
    path = local_table("GSE107559_ivygap_columns-samples.xlsx")
    meta = pd.read_excel(path)
    meta.columns = [str(c).strip().lower().replace(" ", "_") for c in meta.columns]
    return meta


def reference_mask(meta: pd.DataFrame) -> pd.Series:
    name_col = next(c for c in meta.columns if "structure" in c and "name" in c)
    names = meta[name_col].astype(str)
    return names.str.contains("reference histology", case=False, na=False), name_col


def attach_symbols(expr: pd.DataFrame) -> pd.DataFrame:
    path = local_table("GSE107559_ivygap_rows-genes.csv.gz")
    genes = pd.read_csv(path)
    genes.columns = [str(c).strip().lower() for c in genes.columns]
    print("gene columns", list(genes.columns), flush=True)
    symbol_col = next(c for c in genes.columns if "symbol" in c)
    id_col = next(c for c in genes.columns if "entrez" in c or c in {"gene_id", "id"})
    # Prefer the column that matches the matrix index.
    index = set(expr.index.astype(str))
    best_col, best_hit = id_col, genes[id_col].astype(str).isin(index).sum()
    for col in genes.columns:
        hit = genes[col].astype(str).isin(index).sum()
        if hit > best_hit:
            best_col, best_hit = col, int(hit)
    print("gene id column", best_col, "matches", int(best_hit), flush=True)
    mapped = genes[[best_col, symbol_col]].copy()
    mapped["entrez"] = mapped[best_col].astype(str)
    mapped["symbol"] = mapped[symbol_col].astype(str).str.strip()
    mapped = mapped[(mapped["symbol"] != "") & (mapped["symbol"].str.lower() != "nan")]
    expr = expr.copy()
    expr.index = expr.index.astype(str)
    expr = expr.join(mapped.set_index("entrez")["symbol"], how="left")
    expr = expr.dropna(subset=["symbol"])
    numeric = [c for c in expr.columns if c != "symbol"]
    collapsed = expr.groupby("symbol", as_index=True)[numeric].sum()
    return collapsed


def restore_full_list_audit() -> None:
    """The full-list audit counts genes in the CGGA symbol universe. Put that file back."""
    gene_col = pd.read_csv(
        Path(r"E:\Glycocalyx Immune Escape\data\raw\cgga\mRNAseq_325\CGGA.mRNAseq_325.RSEM-genes.20200506.txt"),
        sep="\t", usecols=[0],
    ).iloc[:, 0].astype(str)
    previous = fl.OUT
    fl.OUT = FULL_LIST_OUT
    fl.build_sets(set(gene_col))
    fl.OUT = previous


def main() -> None:
    meta = load_meta()
    meta.to_csv(OUT / "metadata_columns_seen.csv", index=False)
    print("meta columns", list(meta.columns), flush=True)
    keep, name_col = reference_mask(meta)
    ref = meta.loc[keep].copy()
    print(ref[name_col].value_counts().to_string(), flush=True)
    ct_name = [n for n in ref[name_col].unique() if str(n).lower().startswith("cellular tumor")]
    pan_name = [n for n in ref[name_col].unique() if str(n).lower().startswith("pseudopalisading")]
    if len(ct_name) != 1 or len(pan_name) != 1:
        verdict = {
            "ivy_arm": "uninterpretable",
            "reason": "reference-histology structure names did not match the locked prefixes",
            "cellular_names": ct_name,
            "pseudopalisade_names": pan_name,
            "recording_study_opened": False,
        }
        (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
        print(json.dumps(verdict, indent=2), flush=True)
        return

    fpkm_path = local_table("GSE107559_ivygap_fpkm_table.csv.gz")
    expr = pd.read_csv(fpkm_path, index_col=0)
    expr.index = expr.index.astype(str).str.strip()
    expr = expr.apply(pd.to_numeric, errors="coerce")
    expr = attach_symbols(expr)
    print("expr", expr.shape, flush=True)

    # Sample identifiers in the matrix have to be matched to the metadata without using expression.
    sample_col = next(
        (c for c in meta.columns if c in {"sample_id", "rna_well_id", "rnaseq_sample_id", "well_id", "sample"}),
        None,
    )
    tumor_col = next((c for c in meta.columns if "tumor" in c and "name" in c), None)
    if tumor_col is None:
        tumor_col = next((c for c in meta.columns if c.startswith("tumor")), None)
    overlap = set(expr.columns.astype(str)) & set(meta.astype(str).to_numpy().ravel())
    print("sample_col", sample_col, "tumor_col", tumor_col, "id overlap", len(overlap), flush=True)
    (OUT / "meta_head.csv").write_text(meta.head(3).to_csv(index=False), encoding="utf-8")

    # Resolve which metadata column holds the matrix column ids.
    id_col = None
    best = 0
    for col in meta.columns:
        hit = meta[col].astype(str).isin(set(expr.columns.astype(str))).sum()
        if hit > best:
            best = int(hit)
            id_col = col
    print("id column", id_col, "matches", best, flush=True)
    if id_col is None or best < 20:
        verdict = {"ivy_arm": "uninterpretable", "reason": "matrix columns do not match metadata ids", "recording_study_opened": False}
        (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
        print(json.dumps(verdict, indent=2), flush=True)
        return

    universe = set(expr.index)
    fl.OUT = OUT
    fl.build_sets(universe)
    needed = ["hypoxia_f1", "mes1_f1", "neuronal_composition", "myeloid_composition"]
    ref = ref.copy()
    ref["sample_key"] = ref[id_col].astype(str)
    use = ref[ref["sample_key"].isin(expr.columns.astype(str))]
    sub = expr[use["sample_key"].tolist()]
    scores = fl.score_matrix(sub)
    scores["sample_key"] = scores.index.astype(str)
    merged = scores.merge(
        use[["sample_key", name_col, tumor_col]].rename(columns={name_col: "structure", tumor_col: "tumor"}),
        on="sample_key",
        how="inner",
    )
    merged.to_csv(OUT / "sample_scores.csv", index=False)

    def tumor_means(structure: str) -> pd.DataFrame:
        part = merged[merged["structure"] == structure]
        return part.groupby("tumor", as_index=False)[needed].mean()

    ct = tumor_means(ct_name[0])
    pan = tumor_means(pan_name[0])
    ct.to_csv(OUT / "ct_tumor_means.csv", index=False)
    pan.to_csv(OUT / "pan_tumor_means.csv", index=False)
    print(f"CT tumors {ct['tumor'].nunique()} samples {int((merged['structure']==ct_name[0]).sum())}", flush=True)
    print(f"PAN tumors {pan['tumor'].nunique()} samples {int((merged['structure']==pan_name[0]).sum())}", flush=True)

    if ct["tumor"].nunique() < MIN_TUMORS:
        verdict = {
            "ivy_arm": "uninterpretable",
            "reason": "fewer than 8 cellular-tumor tumors",
            "n_tumors": int(ct["tumor"].nunique()),
            "recording_study_opened": False,
        }
        (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
        print(json.dumps(verdict, indent=2), flush=True)
        return

    covariates = []
    for cov in ("neuronal_composition", "myeloid_composition"):
        if float(ct[cov].std(ddof=0) or 0) > 0:
            covariates.append(cov)
    d = ct[["mes1_f1", "hypoxia_f1"] + covariates].apply(pd.to_numeric, errors="coerce").dropna().copy()
    if len(d) < MIN_TUMORS:
        verdict = {
            "ivy_arm": "uninterpretable",
            "reason": "fewer than 8 tumors with finite F1 scores",
            "n_tumors": int(len(d)),
            "recording_study_opened": False,
        }
        (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
        print(json.dumps(verdict, indent=2), flush=True)
        return
    for col in d.columns:
        d[col] = rg.zscore(d[col])
    rhs = " + ".join(["hypoxia_f1"] + covariates)
    fit = smf.ols(f"mes1_f1 ~ {rhs}", data=d).fit(cov_type="HC3")
    coef = float(fit.params["hypoxia_f1"])
    se = float(fit.bse["hypoxia_f1"])
    ci = fit.conf_int().loc["hypoxia_f1"]
    lo, hi = float(ci.iloc[0]), float(ci.iloc[1])
    supports = bool(coef >= 0.20 and lo > 0)
    against = bool(hi < 0.20)
    if supports:
        arm = "supports_within_cellular_tumor"
    elif against:
        arm = "against_within_cellular_tumor"
    else:
        arm = "inconclusive"

    paired = ct.merge(pan, on="tumor", suffixes=("_ct", "_pan"))
    paired["d_hypoxia"] = paired["hypoxia_f1_pan"] - paired["hypoxia_f1_ct"]
    paired["d_mes"] = paired["mes1_f1_pan"] - paired["mes1_f1_ct"]
    paired.to_csv(OUT / "paired_pan_minus_ct.csv", index=False)

    verdict = {
        "ivy_arm": arm,
        "n_ct_tumors": int(d.shape[0]),
        "n_ct_samples": int((merged["structure"] == ct_name[0]).sum()),
        "n_pan_tumors": int(pan["tumor"].nunique()),
        "n_paired_tumors": int(paired["tumor"].nunique()),
        "ct_adjusted_coef": coef,
        "ct_se": se,
        "ct_ci95": [lo, hi],
        "ct_p": float(fit.pvalues["hypoxia_f1"]),
        "covariates": covariates,
        "mean_pan_minus_ct_hypoxia": float(paired["d_hypoxia"].mean()) if len(paired) else None,
        "mean_pan_minus_ct_mes1": float(paired["d_mes"].mean()) if len(paired) else None,
        "visium_image_test": "not_run",
        "recording_study_opened": False,
    }
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        restore_full_list_audit()
