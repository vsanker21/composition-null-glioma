"""Full-list hypoxia vs mesenchymal replication. Rules are in FULL_LIST_PROTOCOL.md.

Do not change the contrasts, the floors, or the populations after reading the output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import rankdata

sys.path.insert(0, str(Path(r"G:\James Zou Project\operative_state")))
import run_gates as rg  # noqa: E402
import run_null_standard as ns  # noqa: E402

OUT = Path(r"G:\James Zou Project\operative_state\results\full_list")
OUT.mkdir(parents=True, exist_ok=True)

GMT = Path(r"G:\GBM Detection\data\references\gene_sets\msigdb\h.all.v2023.2.Hs.symbols.gmt")
NEFTEL = Path(r"G:\James Zou Project\operative_state\data_neftel_metamodules.csv")

ALIASES = {"ERO1L": "ERO1A", "C8orf4": "TCIM", "WARS": "WARS1"}
MIN_GENES = 20
GLASS_FLOOR = 0.20
EXCESS_FLOOR = 0.10
P_FLOOR = 0.10

# Filled by build_sets before any expression matrix is scored.
SETS: dict[str, list[str]] = {}
CONTRASTS: list[tuple[str, str, str, str]] = []


def load_hallmark() -> list[str]:
    for line in GMT.read_text(encoding="utf-8").splitlines():
        if line.startswith("HALLMARK_HYPOXIA\t"):
            return line.rstrip("\n").split("\t")[2:]
    raise RuntimeError("HALLMARK_HYPOXIA not in the GMT")


def load_meta(name: str) -> list[str]:
    table = pd.read_csv(NEFTEL)
    genes = table.loc[table["geneset"] == name, "genesymbol"].astype(str).tolist()
    return [g for g in genes if g != "NPC2MM"]


def resolve(genes: list[str], universe: set[str]) -> list[str]:
    """Keep a symbol, or its alias if the symbol is absent. Never keep both."""
    out = []
    for gene in genes:
        if gene in universe:
            out.append(gene)
        elif ALIASES.get(gene) in universe:
            out.append(ALIASES[gene])
    # Preserve order, drop duplicates.
    seen = set()
    unique = []
    for gene in out:
        if gene not in seen:
            seen.add(gene)
            unique.append(gene)
    return unique


def build_sets(universe: set[str]) -> pd.DataFrame:
    """Apply the locked set arithmetic. No expression values are used."""
    comp = set(rg.NEURONAL) | set(rg.MYELOID)
    hallmark = set(load_hallmark())
    mes1 = set(load_meta("MES1"))
    mes2 = set(load_meta("MES2"))
    share_1 = hallmark & mes1
    share_2 = hallmark & mes2
    share_both = hallmark & (mes1 | mes2)

    defined = {
        "hypoxia_f1": sorted(hallmark - share_1 - comp),
        "mes1_f1": sorted(mes1 - share_1 - comp),
        "hypoxia_f2": sorted(hallmark - comp),
        "mes2_f2": sorted(mes2 - comp),
        "hypoxia_f3": sorted(hallmark - share_2 - comp),
        "mes2_f3": sorted(mes2 - share_2 - comp),
        "hypoxia_f4": sorted(hallmark - share_both - comp),
        "neuronal_composition": list(rg.NEURONAL),
        "myeloid_composition": list(rg.MYELOID),
    }
    rows = []
    SETS.clear()
    for name, genes in defined.items():
        kept = genes if name.endswith("composition") else resolve(genes, universe)
        SETS[name] = kept
        rows.append({
            "program": name,
            "n_defined": len(genes),
            "n_in_universe": len(kept),
            "interpretable": len(kept) >= MIN_GENES or name.endswith("composition"),
        })
    # F4 outcome is the mean of two scores, so those lists are stored too.
    SETS["mes1_f1_for_f4"] = SETS["mes1_f1"]
    SETS["mes2_f3_for_f4"] = SETS["mes2_f3"]
    audit = pd.DataFrame(rows)
    audit.to_csv(OUT / "gene_sets.csv", index=False)
    overlap = pd.DataFrame([
        {"pair": "hallmark_mes1", "n": len(share_1), "genes": ";".join(sorted(share_1))},
        {"pair": "hallmark_mes2", "n": len(share_2), "genes": ";".join(sorted(share_2))},
        {"pair": "hallmark_composition", "n": len(hallmark & comp), "genes": ";".join(sorted(hallmark & comp))},
    ])
    overlap.to_csv(OUT / "overlaps.csv", index=False)
    return audit


def score_matrix(expr: pd.DataFrame) -> pd.DataFrame:
    centered = rg.centered_ranks(expr)
    scores = {}
    for name, genes in SETS.items():
        if name.endswith("_for_f4"):
            continue
        scores[name] = rg.singscore_from_ranks(centered, genes, len(genes) if genes else 1)
    scores["mes_f4"] = (scores["mes1_f1"] + scores["mes2_f3"]) / 2.0
    return pd.DataFrame(scores)


def load_cgga() -> pd.DataFrame:
    base = Path(r"E:\Glycocalyx Immune Escape\data\raw\cgga")
    pairs = [
        ("mRNAseq_325", base / "mRNAseq_325" / "CGGA.mRNAseq_325.RSEM-genes.20200506.txt",
         base / "mRNAseq_325" / "CGGA.mRNAseq_325_clinical.20200506.txt"),
        ("mRNAseq_693", base / "mRNAseq_693" / "CGGA.mRNAseq_693.RSEM-genes.20200506.txt",
         base / "mRNAseq_693" / "CGGA.mRNAseq_693_clinical.20200506.txt"),
    ]
    frames = []
    for cohort, expr_path, clin_path in pairs:
        print(f"CGGA {cohort}", flush=True)
        expr = rg.load_expression(expr_path)
        clin = pd.read_csv(clin_path, sep="\t")
        colmap = {c.lower(): c for c in clin.columns}
        id_col = next(colmap[k] for k in ("cgga_id", "sample", "sampleid", "id") if k in colmap)
        clin["sample_id"] = clin[id_col].astype(str).str.strip()
        expr = expr.rename(columns={c: str(c).strip() for c in expr.columns})
        expr = expr[[c for c in expr.columns if c in set(clin["sample_id"])]]
        scores = score_matrix(expr)
        del expr
        scores["sample_id"] = scores.index.astype(str)
        merged = scores.merge(clin, on="sample_id", how="inner")
        merged["cohort"] = cohort
        gcol = next((colmap[k] for k in ("grade", "who_grade") if k in colmap), None)
        roman = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "1": 1, "2": 2, "3": 3, "4": 4}
        if gcol:
            grade = (
                merged[gcol].astype(str).str.lower().str.replace("who", "", regex=False)
                .str.replace("grade", "", regex=False).str.strip()
                .str.replace(r"[^ivx0-9]", "", regex=True)
            )
            merged["grade_ord"] = grade.map(roman).astype(float)
        else:
            merged["grade_ord"] = np.nan
        idh_col = next((colmap[k] for k in ("idh_mutation_status", "idh_status", "idh_mutation") if k in colmap), None)
        if idh_col:
            low = merged[idh_col].astype(str).str.lower()
            merged["idh_mutant"] = np.where(
                low.str.contains("wild|wt", regex=True), 0.0,
                np.where(low.str.contains("mut"), 1.0, np.nan),
            )
        else:
            merged["idh_mutant"] = np.nan
        frames.append(merged)
    return pd.concat(frames, ignore_index=True)


def fit_models(dat: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for cohort, sub in dat.groupby("cohort"):
        for cid, family, pred, outcome in CONTRASTS:
            for model, extra in (
                ("unadjusted", ["grade_ord", "idh_mutant"]),
                ("composition", ["grade_ord", "idh_mutant", "neuronal_composition", "myeloid_composition"]),
            ):
                try:
                    fit = ns.fit_one(sub, outcome, pred, extra)
                except Exception as exc:  # noqa: BLE001
                    fit = {"error": str(exc)}
                rows.append({
                    "coupling": cid, "family": family, "cohort": cohort, "model": model, **fit,
                })
    tab = pd.DataFrame(rows)
    meta_rows = []
    ok = tab.dropna(subset=["coef", "se"]) if "coef" in tab.columns else tab
    for (cid, model), group in ok.groupby(["coupling", "model"]):
        coef, se, pval = rg.meta_iv(group["coef"].to_numpy(), group["se"].to_numpy())
        meta_rows.append({
            "coupling": cid,
            "family": group["family"].iloc[0],
            "model": model,
            "meta_coef": coef,
            "meta_se": se,
            "meta_pval": pval,
            "n_total": int(group["n"].sum()),
        })
    meta = pd.DataFrame(meta_rows)
    parts = []
    for (_, _), group in meta.groupby(["family", "model"]):
        group = group.copy()
        group["qval_bh"] = rg.bh(group["meta_pval"])
        parts.append(group)
    return pd.concat(parts, ignore_index=True), tab


def load_glass() -> pd.DataFrame:
    print("GLASS", flush=True)
    expr = rg.load_expression(Path(r"G:\GBM Detection\data\natcomms\glass\glass_gene_tpm_matrix_all_samples.tsv"))
    parsed = [rg.parse_glass_column(c) for c in expr.columns]
    keep = [i for i, (case, time) in enumerate(parsed) if case and time in {"TP", "R1"}]
    expr = expr.iloc[:, keep]
    scores = score_matrix(expr)
    del expr
    scores["case"] = [parsed[i][0] for i in keep]
    scores["time"] = [parsed[i][1] for i in keep]
    score_cols = [c for c in scores.columns if c not in {"case", "time"}]
    averaged = scores.groupby(["case", "time"], as_index=False)[score_cols].mean()
    averaged["case_key"] = averaged["case"].str.upper()
    parts = []
    for time, sub in averaged.groupby("time"):
        renamed = sub.drop(columns=["time"]).rename(
            columns={c: f"{c}__{time}" for c in sub.columns if c not in {"case", "case_key"}}
        )
        parts.append(renamed)
    wide = parts[0].merge(parts[1], on=["case", "case_key"], how="inner")
    clin = rg.clinical_lookup()
    wide = wide.merge(clin, on="case_key", how="left")
    return wide[(wide["idh_wt"] == True) & (wide["gbm"] == True)].copy()  # noqa: E712


def glass_models(primary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cid, family, pred, outcome in CONTRASTS:
        d = pd.DataFrame({
            "dy": primary[f"{outcome}__R1"] - primary[f"{outcome}__TP"],
            "dx": primary[f"{pred}__R1"] - primary[f"{pred}__TP"],
            "dn": primary["neuronal_composition__R1"] - primary["neuronal_composition__TP"],
            "dm": primary["myeloid_composition__R1"] - primary["myeloid_composition__TP"],
        }).dropna()
        for model, extras in (("unadjusted", []), ("composition", ["dn", "dm"])):
            dd = d[["dy", "dx"] + extras].copy()
            for col in dd.columns:
                dd[col] = rg.zscore(dd[col])
            fit = smf.ols("dy ~ " + " + ".join(["dx"] + extras), data=dd).fit(cov_type="HC3")
            rows.append({
                "coupling": cid, "family": family, "model": model,
                "meta_coef": float(fit.params["dx"]), "meta_se": float(fit.bse["dx"]),
                "meta_pval": float(fit.pvalues["dx"]), "n_total": int(fit.nobs),
                "r2": float(fit.rsquared),
            })
    tab = pd.DataFrame(rows)
    parts = []
    for (_, _), group in tab.groupby(["family", "model"]):
        group = group.copy()
        group["qval_bh"] = rg.bh(group["meta_pval"])
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def score_spots(counts: np.ndarray, gene_names: list[str]) -> pd.DataFrame:
    ranks = rankdata(counts, axis=0, method="average")
    n_genes = ranks.shape[0]
    centered = (ranks - (n_genes + 1) / 2.0) / n_genes
    index = pd.Index(gene_names)
    out = {}
    for name, genes in SETS.items():
        if name.endswith("_for_f4"):
            continue
        present = [g for g in genes if g in index]
        if len(present) < max(3, int(0.3 * max(len(genes), 1))):
            out[name] = np.full(counts.shape[1], np.nan)
            continue
        out[name] = centered[index.get_indexer(present)].mean(axis=0)
    out["mes_f4"] = (out["mes1_f1"] + out["mes2_f3"]) / 2.0
    return pd.DataFrame(out)


def spatial_layer(rng: np.random.Generator) -> pd.DataFrame:
    section_rows = []
    draws = {cid: [] for cid, *_ in CONTRASTS}
    observed = {cid: [] for cid, *_ in CONTRASTS}
    families = {cid: family for cid, family, *_ in CONTRASTS}
    folders = sorted(
        p for p in ns.VISIUM.iterdir()
        if p.is_dir() and "_T_ST" in p.name and "_TI_" not in p.name and "_TC_" not in p.name and "_C_ST" not in p.name
    )
    print(f"Visium tumor sections {len(folders)}", flush=True)
    for folder in folders:
        h5_path = folder / "outs" / "filtered_feature_bc_matrix.h5"
        if not h5_path.exists():
            continue
        print(folder.name, flush=True)
        counts, names = ns.read_visium_counts(h5_path)
        scores = score_spots(counts, names)
        del counts
        keep = scores[["neuronal_composition", "myeloid_composition"]].notna().all(axis=1)
        scores = scores.loc[keep]
        if len(scores) < 100:
            continue
        sample = folder.name.lstrip("#")
        neuronal = scores["neuronal_composition"].to_numpy()
        myeloid = scores["myeloid_composition"].to_numpy()
        for cid, _family, pred, outcome in CONTRASTS:
            mask = (
                np.isfinite(scores[pred].to_numpy())
                & np.isfinite(scores[outcome].to_numpy())
                & np.isfinite(neuronal)
                & np.isfinite(myeloid)
            )
            if int(mask.sum()) < 100:
                continue
            rho, null = ns.stratified_draws(
                scores[pred].to_numpy()[mask], scores[outcome].to_numpy()[mask],
                neuronal[mask], myeloid[mask], rng,
            )
            if not np.isfinite(rho):
                continue
            observed[cid].append(rho)
            draws[cid].append(null)
            section_rows.append({
                "sample": sample, "coupling": cid, "n_spots": int(mask.sum()),
                "rho": rho, "null_median": float(np.nanmedian(null)),
            })
    pd.DataFrame(section_rows).to_csv(OUT / "visium_section_permutation.csv", index=False)
    rows = []
    for cid, rhos in observed.items():
        if not rhos:
            continue
        obs_mean = float(np.mean(rhos))
        null_means = np.nanmean(np.vstack(draws[cid]), axis=0)
        null_center = float(np.median(null_means))
        pvalue = (1 + np.sum(np.abs(null_means) >= abs(obs_mean))) / (ns.N_PERM + 1)
        rows.append({
            "coupling": cid,
            "family": families[cid],
            "n_sections": len(rhos),
            "mean_rho": obs_mean,
            "null_center": null_center,
            "excess": obs_mean - null_center,
            "perm_p": float(pvalue),
        })
    summary = pd.DataFrame(rows)
    parts = []
    for _family, group in summary.groupby("family"):
        group = group.copy()
        group["qval_bh"] = rg.bh(group["perm_p"])
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def clause(ok: bool, text: str) -> dict:
    return {"pass": bool(ok), "rule": text}


def main() -> None:
    global CONTRASTS
    # Alias resolution uses the CGGA 325 symbol column only. No expression values.
    gene_col = pd.read_csv(
        Path(r"E:\Glycocalyx Immune Escape\data\raw\cgga\mRNAseq_325\CGGA.mRNAseq_325.RSEM-genes.20200506.txt"),
        sep="\t", usecols=[0],
    ).iloc[:, 0].astype(str)
    universe = set(gene_col)
    audit = build_sets(universe)
    print(audit.to_string(index=False), flush=True)
    f1_ok = bool(audit.loc[audit["program"].isin(["hypoxia_f1", "mes1_f1"]), "interpretable"].all())
    CONTRASTS = [
        ("F1", "primary", "hypoxia_f1", "mes1_f1"),
        ("F2", "control", "hypoxia_f2", "mes2_f2"),
        ("F3", "control", "hypoxia_f3", "mes2_f3"),
        ("F4", "control", "hypoxia_f4", "mes_f4"),
    ]
    if not f1_ok:
        verdict = {"biological_result": False, "reason": "F1 gene set below 20 genes"}
        (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
        print(verdict, flush=True)
        return

    cgga_meta, cgga_by = fit_models(load_cgga())
    cgga_meta.to_csv(OUT / "cgga_meta.csv", index=False)
    cgga_by.to_csv(OUT / "cgga_by_cohort.csv", index=False)
    glass = glass_models(load_glass())
    glass.to_csv(OUT / "glass_models.csv", index=False)
    spatial = spatial_layer(np.random.default_rng(ns.SEED))
    spatial.to_csv(OUT / "visium_excess.csv", index=False)

    def row(table: pd.DataFrame, cid: str, model: str) -> pd.Series:
        hit = table[(table["coupling"] == cid) & (table["model"] == model)]
        return hit.iloc[0]

    f1g = row(glass, "F1", "composition")
    f1c = row(cgga_meta, "F1", "composition")
    f1s = spatial[spatial["coupling"] == "F1"].iloc[0]
    glass_clause = bool(f1g["meta_coef"] > 0 and f1g["meta_pval"] < P_FLOOR and f1g["meta_coef"] >= GLASS_FLOOR)
    spatial_clause = bool(f1s["excess"] >= EXCESS_FLOOR and f1s["perm_p"] < P_FLOOR)
    cgga_clause = bool(f1c["meta_coef"] > 0 and f1c["meta_pval"] < P_FLOOR)
    holds = bool(glass_clause and spatial_clause and cgga_clause)
    verdict = {
        "biological_result": holds,
        "draft_methods_paper": holds,
        "recording_study_opened": False,
        "glass_adjusted_coef": float(f1g["meta_coef"]),
        "glass_p": float(f1g["meta_pval"]),
        "glass_n": int(f1g["n_total"]),
        "cgga_adjusted_coef": float(f1c["meta_coef"]),
        "cgga_p": float(f1c["meta_pval"]),
        "spatial_mean_rho": float(f1s["mean_rho"]),
        "spatial_null_center": float(f1s["null_center"]),
        "spatial_excess": float(f1s["excess"]),
        "spatial_perm_p": float(f1s["perm_p"]),
        "clauses": {
            "within_patient": clause(glass_clause, "GLASS adjusted coef >= 0.20, positive, p < 0.10"),
            "spatial_excess": clause(spatial_clause, "excess >= 0.10 and perm p < 0.10"),
            "cross_section": clause(cgga_clause, "CGGA adjusted coef positive, p < 0.10"),
        },
    }
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
