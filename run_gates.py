"""Public-data gates for the operative-state claim. Rules are in PROTOCOL.md."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

OUT = Path(r"G:\James Zou Project\operative_state\results")
OUT.mkdir(parents=True, exist_ok=True)

PROGRAMS = json.loads(
    Path(r"F:\Glioma Radiotranscriptomics\data\05_signatures\neurotransmitter_programs_prespecified.json").read_text(
        encoding="utf-8"
    )
)["programs"]

IMMUNE = {
    "macrophage_m2_like": ["CD163", "MRC1", "MSR1", "MS4A4A", "TGFB1", "IL10", "ARG1", "CLEC7A", "VSIG4"],
    "macrophage_m1_like": ["TNF", "IL1B", "IL6", "NOS2", "CXCL9", "CXCL10", "CD80", "CD86", "IRF5"],
    "immune_estimate_like": [
        "CD8A", "CD8B", "CD3D", "CD3E", "CD3G", "GZMA", "GZMB", "PRF1", "NKG7",
        "CD19", "MS4A1", "CD79A", "HLA-DRA", "HLA-DRB1", "ITGAM", "CD14", "CSF1R", "TYROBP", "FCGR3A",
    ],
    "microglia_like": ["TMEM119", "P2RY12", "CX3CR1", "SALL1", "HEXB", "OLFML3", "SELPLG"],
}

P1 = [
    ("P1.1", "purinergic_adenosine", "macrophage_m2_like", 0.3404223653128615),
    ("P1.2", "gaba_receptor", "immune_estimate_like", -0.4747941161671991),
    ("P1.3", "dopamine_receptor", "m2_minus_m1", -0.4259650178845405),
    ("P1.4", "glutamate_ligand", "immune_estimate_like", -0.3343706260034489),
    ("P1.5", "glutamate_receptor", "microglia_like", -0.3962891069880083),
]
PUBLISHED_META = {
    "P1.1": 0.4104375041950957,
    "P1.2": -0.3866436112273842,
    "P1.3": -0.12580547003307488,
    "P1.4": -0.3830418112131485,
    "P1.5": -0.128216592244997,
}

NEURONAL = ["RBFOX3", "SNAP25", "SYP", "SYN1", "STMN2", "ENO2", "TUBB3", "MAP2"]
MYELOID = ["AIF1", "LST1", "FCER1G", "C1QA", "C1QB", "C1QC", "SPI1", "LYZ"]
TIMEPOINTS = {"TP", "R1", "R2", "R3", "R4", "NB"}


def outcome_genes(name: str) -> set[str]:
    if name == "m2_minus_m1":
        return set(IMMUNE["macrophage_m2_like"]) | set(IMMUNE["macrophage_m1_like"])
    return set(IMMUNE[name])


def centered_ranks(expr: pd.DataFrame) -> pd.DataFrame:
    ranks = expr.rank(axis=0, method="average")
    n = ranks.shape[0]
    return (ranks - (n + 1) / 2.0) / n


def singscore_from_ranks(centered: pd.DataFrame, genes: list[str], n_defined: int) -> pd.Series:
    present = [g for g in genes if g in centered.index]
    if len(present) < max(3, int(0.3 * n_defined)):
        return pd.Series(np.nan, index=centered.columns)
    return centered.loc[present].mean(axis=0)


def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    sd = float(s.std(ddof=0) or 0)
    if sd <= 0 or not np.isfinite(sd):
        return s * np.nan
    return (s - s.mean()) / sd


def bh(p: pd.Series) -> pd.Series:
    q = np.full(len(p), np.nan)
    ok = p.notna().to_numpy()
    if ok.sum():
        q[ok] = multipletests(p.to_numpy()[ok], method="fdr_bh")[1]
    return pd.Series(q, index=p.index)


def two_sided_norm_p(z: float) -> float:
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def meta_iv(coefs: np.ndarray, ses: np.ndarray) -> tuple[float, float, float]:
    w = 1.0 / np.square(ses)
    coef = float(np.sum(w * coefs) / np.sum(w))
    se = float(np.sqrt(1.0 / np.sum(w)))
    return coef, se, two_sided_norm_p(coef / se)


def load_expression(path: Path) -> pd.DataFrame:
    expr = pd.read_csv(path, sep="\t")
    expr = expr.set_index(expr.columns[0])
    expr.index = expr.index.astype(str).str.strip()
    expr = expr[~expr.index.duplicated(keep="first")]
    expr = expr.apply(pd.to_numeric, errors="coerce")
    return expr


def assert_composition_disjoint() -> None:
    banned = set()
    for _, nt, tme, _ in P1:
        banned |= set(PROGRAMS[nt])
        banned |= outcome_genes(tme)
    hit_n = sorted(set(NEURONAL) & banned)
    hit_m = sorted(set(MYELOID) & banned)
    if hit_n or hit_m:
        raise SystemExit(f"Composition genes intersect P1 sets: neuronal={hit_n} myeloid={hit_m}")


def load_cgga() -> pd.DataFrame:
    base = Path(r"E:\Glycocalyx Immune Escape\data\raw\cgga")
    pairs = [
        ("mRNAseq_325", base / "mRNAseq_325" / "CGGA.mRNAseq_325.RSEM-genes.20200506.txt",
         base / "mRNAseq_325" / "CGGA.mRNAseq_325_clinical.20200506.txt"),
        ("mRNAseq_693", base / "mRNAseq_693" / "CGGA.mRNAseq_693.RSEM-genes.20200506.txt",
         base / "mRNAseq_693" / "CGGA.mRNAseq_693_clinical.20200506.txt"),
    ]
    frames = []
    for cohort, expr_p, clin_p in pairs:
        print(f"CGGA {cohort}", flush=True)
        expr = load_expression(expr_p)
        clin = pd.read_csv(clin_p, sep="\t")
        colmap = {c.lower(): c for c in clin.columns}
        id_col = next(colmap[k] for k in ("cgga_id", "sample", "sampleid", "id") if k in colmap)
        clin["sample_id"] = clin[id_col].astype(str).str.strip()
        expr = expr.rename(columns={c: str(c).strip() for c in expr.columns})
        keep = [c for c in expr.columns if c in set(clin["sample_id"])]
        expr = expr[keep]
        centered = centered_ranks(expr)
        del expr
        scores = {
            "neuronal_composition": singscore_from_ranks(centered, NEURONAL, len(NEURONAL)),
            "myeloid_composition": singscore_from_ranks(centered, MYELOID, len(MYELOID)),
        }
        for _, nt, _, _ in P1:
            scores[f"nt_{nt}"] = singscore_from_ranks(centered, PROGRAMS[nt], len(PROGRAMS[nt]))
        for name, genes in IMMUNE.items():
            scores[f"tme_{name}"] = singscore_from_ranks(centered, genes, len(genes))
        scores["tme_m2_minus_m1"] = scores["tme_macrophage_m2_like"] - scores["tme_macrophage_m1_like"]
        for eid, nt, tme, _ in P1:
            inter = set(PROGRAMS[nt]) & outcome_genes(tme)
            scores[f"nt_drop_{eid}"] = singscore_from_ranks(
                centered, [g for g in PROGRAMS[nt] if g not in inter], len(PROGRAMS[nt])
            )
            if tme == "m2_minus_m1":
                m2 = [g for g in IMMUNE["macrophage_m2_like"] if g not in inter]
                m1 = [g for g in IMMUNE["macrophage_m1_like"] if g not in inter]
                scores[f"tme_drop_{eid}"] = singscore_from_ranks(centered, m2, len(IMMUNE["macrophage_m2_like"])) - singscore_from_ranks(
                    centered, m1, len(IMMUNE["macrophage_m1_like"])
                )
            else:
                scores[f"tme_drop_{eid}"] = singscore_from_ranks(
                    centered, [g for g in IMMUNE[tme] if g not in inter], len(IMMUNE[tme])
                )
        sc = pd.DataFrame(scores)
        sc["sample_id"] = sc.index.astype(str)
        merged = sc.merge(clin, on="sample_id", how="inner")
        merged["cohort"] = cohort
        gcol = next((colmap[k] for k in ("grade", "who_grade") if k in colmap), None)
        roman = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "1": 1, "2": 2, "3": 3, "4": 4}
        if gcol:
            g = (
                merged[gcol].astype(str).str.lower().str.replace("who", "", regex=False)
                .str.replace("grade", "", regex=False).str.strip()
                .str.replace(r"[^ivx0-9]", "", regex=True)
            )
            merged["grade_ord"] = g.map(roman).astype(float)
        else:
            merged["grade_ord"] = np.nan
        idhcol = next((colmap[k] for k in ("idh_mutation_status", "idh_status", "idh_mutation", "idh1_mutation") if k in colmap), None)
        if idhcol:
            low = merged[idhcol].astype(str).str.lower()
            merged["idh_mutant"] = np.where(low.str.contains("wild|wt", regex=True), 0.0,
                                            np.where(low.str.contains("mut"), 1.0, np.nan))
        else:
            merged["idh_mutant"] = np.nan
        prs = next((colmap[k] for k in ("prs_type", "prstype") if k in colmap), None)
        merged["prs_type"] = merged[prs].astype(str) if prs else "unknown"
        frames.append(merged)
        del centered
    return pd.concat(frames, ignore_index=True)


def fit_endpoint(df: pd.DataFrame, y: str, x: str, extra: list[str]) -> dict:
    cols = [y, x, "grade_ord", "idh_mutant"] + extra
    d = df[cols].apply(pd.to_numeric, errors="coerce").dropna().copy()
    if len(d) < 40:
        return {"n": int(len(d)), "error": "n<40"}
    for c in d.columns:
        if c == "idh_mutant":
            continue
        d[c] = zscore(d[c])
    rhs = " + ".join([x, "grade_ord", "idh_mutant"] + extra)
    m = smf.ols(f"{y} ~ {rhs}", data=d).fit(cov_type="HC3")
    return {
        "coef": float(m.params[x]),
        "se": float(m.bse[x]),
        "pval": float(m.pvalues[x]),
        "n": int(m.nobs),
        "r2": float(m.rsquared),
    }


def cgga_family(dat: pd.DataFrame, model: str, extra: list[str], drop: bool) -> pd.DataFrame:
    rows = []
    for cohort, sub in dat.groupby("cohort"):
        for eid, nt, tme, disc in P1:
            y = f"tme_drop_{eid}" if drop else (f"tme_{tme}" if tme != "m2_minus_m1" else "tme_m2_minus_m1")
            x = f"nt_drop_{eid}" if drop else f"nt_{nt}"
            # Patsy treats '.' as a method call, so endpoint ids cannot enter the formula.
            work = sub.rename(columns={y: "y_end", x: "x_end"})
            try:
                fit = fit_endpoint(work, "y_end", "x_end", extra)
            except Exception as exc:  # noqa: BLE001
                fit = {"error": str(exc)}
            rows.append({
                "model": model, "endpoint_id": eid, "cohort": cohort,
                "discovery_coef": disc, **fit,
                "direction_concordant": bool(fit.get("coef") is not None and np.isfinite(fit.get("coef", np.nan)) and fit["coef"] * disc > 0),
            })
    tab = pd.DataFrame(rows)
    parts = []
    for cohort, g in tab.groupby("cohort"):
        g = g.copy()
        g["qval_bh"] = bh(g["pval"]) if "pval" in g else np.nan
        parts.append(g)
    tab = pd.concat(parts, ignore_index=True)
    meta_rows = []
    if "coef" not in tab.columns:
        tab["coef"] = np.nan
        tab["se"] = np.nan
    ok = tab.dropna(subset=["coef", "se"])
    for eid, g in ok.groupby("endpoint_id"):
        coef, se, p = meta_iv(g["coef"].to_numpy(), g["se"].to_numpy())
        disc = float(g["discovery_coef"].iloc[0])
        meta_rows.append({
            "model": model, "endpoint_id": eid, "meta_coef": coef, "meta_se": se, "meta_pval": p,
            "n_total": int(g["n"].sum()), "n_cohorts": int(g["cohort"].nunique()),
            "discovery_coef": disc,
            "direction_concordant": bool(coef * disc > 0),
            "published_meta": PUBLISHED_META[eid],
            "abs_diff_vs_published": abs(coef - PUBLISHED_META[eid]) if model == "unadjusted" else np.nan,
        })
    meta = pd.DataFrame(meta_rows)
    if len(meta):
        meta["qval_bh"] = bh(meta["meta_pval"])
    return tab, meta


def gate_from_meta(meta: pd.DataFrame) -> dict:
    if meta is None or len(meta) < 5 or meta["meta_coef"].isna().any():
        return {"identifiable": False, "n_endpoints": int(0 if meta is None else len(meta)), "fail": None}
    lost = ~meta["direction_concordant"] | (meta["qval_bh"] >= 0.10)
    return {
        "identifiable": True,
        "n_endpoints": int(len(meta)),
        "n_direction_and_q_ok": int((~lost).sum()),
        "n_lost": int(lost.sum()),
        "fail": bool(int(lost.sum()) >= 3),
    }


def parse_glass_column(col: str) -> tuple[str | None, str | None]:
    parts = col.split(".")
    for i, part in enumerate(parts):
        if part in TIMEPOINTS and i >= 2:
            return "-".join(parts[:i]), part
    return None, None


def normalize_lobe(value: str) -> str:
    text = str(value).lower()
    for name in ("frontal", "temporal", "parietal", "occipital", "insular", "cerebell"):
        if name in text:
            return "cerebellum" if name == "cerebell" else name
    if text in {"", "nan", "none"}:
        return np.nan
    return "other"


def normalize_eor(value: str) -> str:
    text = str(value).lower()
    if "gross" in text or text.strip() in {"gtr", "total"}:
        return "GTR"
    if any(k in text for k in ("subtotal", "partial", "biopsy", "str", "ntr")):
        return "less_than_GTR"
    return np.nan


def is_idh_wt(value: str) -> bool:
    text = str(value).lower().replace("_", "").replace("-", "").replace(" ", "")
    return text in {"idhwt", "idhwildtype", "wildtype", "wt"} or "wildtype" in text or text.endswith("wt")


def is_gbm(row: pd.Series) -> bool:
    histology = f"{row.get('histology', '')} {row.get('who_classification', '')}".lower()
    grade = str(row.get("grade", "")).upper()
    return ("glioblastoma" in histology) or grade in {"IV", "4"}


def load_glass_scores() -> pd.DataFrame:
    print("GLASS TPM", flush=True)
    path = Path(r"G:\GBM Detection\data\natcomms\glass\glass_gene_tpm_matrix_all_samples.tsv")
    expr = load_expression(path)
    parsed = [parse_glass_column(c) for c in expr.columns]
    keep_idx = [i for i, (case, tp) in enumerate(parsed) if case and tp in {"TP", "R1"}]
    expr = expr.iloc[:, keep_idx]
    meta = pd.DataFrame({
        "aliquot": expr.columns,
        "case": [parsed[i][0] for i in keep_idx],
        "time": [parsed[i][1] for i in keep_idx],
    })
    centered = centered_ranks(expr)
    del expr
    scores = {
        "neuronal_composition": singscore_from_ranks(centered, NEURONAL, len(NEURONAL)),
        "myeloid_composition": singscore_from_ranks(centered, MYELOID, len(MYELOID)),
    }
    for _, nt, _, _ in P1:
        scores[f"nt_{nt}"] = singscore_from_ranks(centered, PROGRAMS[nt], len(PROGRAMS[nt]))
    for name, genes in (
        ("macrophage_m2_like", IMMUNE["macrophage_m2_like"]),
        ("macrophage_m1_like", IMMUNE["macrophage_m1_like"]),
        ("immune_estimate_like", IMMUNE["immune_estimate_like"]),
        ("microglia_like", IMMUNE["microglia_like"]),
    ):
        scores[f"tme_{name}"] = singscore_from_ranks(centered, genes, len(genes))
    scores["tme_m2_minus_m1"] = scores["tme_macrophage_m2_like"] - scores["tme_macrophage_m1_like"]
    sc = pd.DataFrame(scores)
    sc["aliquot"] = sc.index
    merged = sc.merge(meta, on="aliquot", how="inner")
    score_cols = [c for c in merged.columns if c.startswith("nt_") or c.startswith("tme_") or c.endswith("composition")]
    return merged.groupby(["case", "time"], as_index=False)[score_cols].mean()


def clinical_lookup() -> pd.DataFrame:
    surg = pd.read_csv(
        Path(r"G:\GBM Detection\data\natcomms\glass\glass_clinical_surgeries.tsv"), sep="\t"
    )
    surg["case_key"] = surg["case_barcode"].astype(str).str.upper()
    # Prefer the primary surgery row.
    surg["is_tp"] = surg["sample_barcode"].astype(str).str.contains(r"-TP($|[^A-Z0-9])", regex=True) | (
        surg["surgery_number"].astype(str) == "1"
    )
    tp = surg.sort_values(["case_key", "is_tp"], ascending=[True, False]).groupby("case_key", as_index=False).head(1)
    tp = tp.copy()
    tp["lobe"] = tp["surgery_location"].map(normalize_lobe)
    tp["eor"] = tp["surgery_extent_of_resection"].map(normalize_eor)
    tp["rt"] = pd.to_numeric(tp["treatment_radiotherapy"], errors="coerce")
    tp["idh_wt"] = tp["idh_status"].map(is_idh_wt)
    tp["gbm"] = tp.apply(is_gbm, axis=1)
    return tp[["case_key", "lobe", "eor", "rt", "idh_wt", "gbm", "idh_status", "histology", "grade", "surgery_location", "surgery_extent_of_resection"]]


def glass_analysis(scores: pd.DataFrame) -> dict:
    clin = clinical_lookup()
    scores["case_key"] = scores["case"].str.upper()
    wide_parts = []
    for time, sub in scores.groupby("time"):
        ren = sub.drop(columns=["time"]).rename(columns={c: f"{c}__{time}" for c in sub.columns if c not in {"case", "case_key"}})
        wide_parts.append(ren)
    wide = wide_parts[0].merge(wide_parts[1], on=["case", "case_key"], how="inner") if len(wide_parts) == 2 else pd.DataFrame()
    wide = wide.merge(clin, on="case_key", how="left")
    wide.to_csv(OUT / "glass_pair_table.csv", index=False)
    primary = wide[(wide["idh_wt"] == True) & (wide["gbm"] == True)].copy()  # noqa: E712
    primary.to_csv(OUT / "glass_primary_population.csv", index=False)
    diag = {
        "n_tp_aliquot_cases": int((scores["time"] == "TP").sum()),
        "n_r1_aliquot_cases": int((scores["time"] == "R1").sum()),
        "n_pairs_any_clinical": int(len(wide)),
        "n_pairs_with_clinical_row": int(wide["idh_status"].notna().sum()) if "idh_status" in wide else 0,
        "n_primary_idhwt_gbm_pairs": int(len(primary)),
        "lobe_missing_fraction": float(primary["lobe"].isna().mean()) if len(primary) else None,
        "rt_counts": primary["rt"].value_counts(dropna=False).to_dict() if len(primary) else {},
        "eor_counts": primary["eor"].value_counts(dropna=False).astype(int).to_dict() if len(primary) else {},
    }
    # Co-transition
    rows = []
    if len(primary) >= 30:
        for eid, nt, tme, disc in P1:
            y = f"tme_{tme}__R1" 
            y0 = f"tme_{tme}__TP"
            x1 = f"nt_{nt}__R1"
            x0 = f"nt_{nt}__TP"
            d = pd.DataFrame({
                "dy": primary[f"tme_{tme}__R1"] - primary[f"tme_{tme}__TP"],
                "dx": primary[f"nt_{nt}__R1"] - primary[f"nt_{nt}__TP"],
                "dn": primary["neuronal_composition__R1"] - primary["neuronal_composition__TP"],
                "dm": primary["myeloid_composition__R1"] - primary["myeloid_composition__TP"],
            }).dropna()
            for model, extras in (("delta_unadjusted", []), ("delta_composition", ["dn", "dm"])):
                dd = d[["dy", "dx"] + extras].copy()
                for c in dd.columns:
                    dd[c] = zscore(dd[c])
                m = smf.ols("dy ~ " + " + ".join(["dx"] + extras), data=dd).fit(cov_type="HC3")
                rows.append({
                    "endpoint_id": eid, "model": model, "coef": float(m.params["dx"]),
                    "se": float(m.bse["dx"]), "pval": float(m.pvalues["dx"]),
                    "n": int(m.nobs), "r2": float(m.rsquared), "discovery_coef": disc,
                    "direction_concordant": bool(m.params["dx"] * disc > 0),
                    "median_delta_predictor": float(d["dx"].median()),
                    "median_delta_outcome": float(d["dy"].median()),
                })
    delta = pd.DataFrame(rows)
    if len(delta):
        parts = []
        for model, g in delta.groupby("model"):
            g = g.copy()
            g["qval_bh"] = bh(g["pval"])
            parts.append(g)
        delta = pd.concat(parts, ignore_index=True)
    delta.to_csv(OUT / "glass_cotransition.csv", index=False)

    # Location model on TP samples of the primary population, using TP scores joined back.
    tp = scores[scores["time"] == "TP"].merge(
        primary[["case_key", "lobe"]], on="case_key", how="inner"
    )
    loc_rows = []
    lobe_missing = float(tp["lobe"].isna().mean()) if len(tp) else 1.0
    if len(tp) >= 30 and lobe_missing <= 0.40:
        tp = tp.dropna(subset=["lobe"]).copy()
        for eid, nt, tme, disc in P1:
            d = tp[[f"nt_{nt}", f"tme_{tme}", "lobe"]].dropna().copy()
            d["y"] = zscore(d[f"tme_{tme}"])
            d["x"] = zscore(d[f"nt_{nt}"])
            m_lobe = smf.ols("y ~ C(lobe)", data=d).fit()
            m_nt = smf.ols("y ~ x", data=d).fit()
            m_both = smf.ols("y ~ x + C(lobe)", data=d).fit(cov_type="HC3")
            loc_rows.append({
                "endpoint_id": eid,
                "n": int(m_both.nobs),
                "coef_nt_given_lobe": float(m_both.params["x"]),
                "se": float(m_both.bse["x"]),
                "pval": float(m_both.pvalues["x"]),
                "r2_lobe": float(m_lobe.rsquared),
                "r2_nt": float(m_nt.rsquared),
                "r2_both": float(m_both.rsquared),
                "gain_from_nt": float(m_both.rsquared - m_lobe.rsquared),
                "gain_from_lobe": float(m_both.rsquared - m_nt.rsquared),
                "discovery_coef": disc,
                "direction_concordant": bool(m_both.params["x"] * disc > 0),
            })
    loc = pd.DataFrame(loc_rows)
    if len(loc):
        loc["qval_bh"] = bh(loc["pval"])
        loc["clears"] = (
            loc["direction_concordant"] & (loc["qval_bh"] < 0.10) & (loc["gain_from_nt"] > loc["gain_from_lobe"])
        )
    loc.to_csv(OUT / "glass_location_model.csv", index=False)
    comp = delta[delta["model"] == "delta_composition"] if len(delta) else delta
    if len(comp):
        ok = comp["direction_concordant"] & (comp["qval_bh"] < 0.10)
        gate_c = {
            "identifiable": True,
            "n_pairs": int(comp["n"].iloc[0]),
            "n_supported": int(ok.sum()),
            "supported": bool(ok.sum() >= 3),
        }
    else:
        gate_c = {"identifiable": len(primary) >= 30, "n_pairs": int(len(primary)), "supported": False, "n_supported": 0}
    if lobe_missing > 0.40 or loc.empty:
        gate_d = {"identifiable": False, "lobe_missing_fraction": lobe_missing, "n_clear": 0}
    else:
        gate_d = {
            "identifiable": True,
            "lobe_missing_fraction": lobe_missing,
            "n_clear": int(loc["clears"].sum()),
            "location_loses": bool(loc["clears"].sum() >= 3),
        }
    return {"diagnostics": diag, "gate_c": gate_c, "gate_d": gate_d}


def visium_gate() -> dict:
    spots = pd.read_csv(
        Path(r"F:\Glioma Radiotranscriptomics\results\13_visium_core_edge\visium_spot_scores.csv")
    )
    pairs = [
        ("P1.1", "purinergic_adenosine", "macrophage_m2_like", 1),
        ("P1.2", "gaba_receptor", "immune_estimate_like", -1),
        ("P1.3", "dopamine_receptor", "macrophage_m2_like", -1),  # m2-m1 not stored; M2 is the positive arm of the balance
        ("P1.4", "glutamate_ligand", "immune_estimate_like", -1),
        ("P1.5", "glutamate_receptor", "microglia_like", -1),
    ]
    # P1.3 outcome in bulk is M2 minus M1. M1 is not on the spot table. Do not substitute M2 and call it P1.3.
    pairs = [p for p in pairs if p[0] != "P1.3"]
    tumor = spots[spots["section_class"] == "tumor_section"].copy()
    rows = []
    for sample, sub in tumor.groupby("sample"):
        patient = sub["patient"].iloc[0]
        for eid, x, y, sign in pairs:
            if len(sub) < 50:
                rho, p = np.nan, np.nan
            else:
                rho, p = stats.spearmanr(sub[x], sub[y], nan_policy="omit")
            rows.append({"sample": sample, "patient": patient, "endpoint_id": eid, "rho": rho, "expected_sign": sign, "n_spots": int(len(sub))})
    sec = pd.DataFrame(rows)
    sec.to_csv(OUT / "visium_section_spearman.csv", index=False)
    pat = sec.groupby(["patient", "endpoint_id", "expected_sign"], as_index=False)["rho"].mean()
    tests = []
    for eid, g in pat.groupby("endpoint_id"):
        r = g["rho"].dropna()
        if len(r) < 5:
            stat, p = np.nan, np.nan
        else:
            stat, p = stats.wilcoxon(r, alternative="two-sided", zero_method="wilcox")
        sign = int(g["expected_sign"].iloc[0])
        tests.append({
            "endpoint_id": eid,
            "n_patients": int(len(r)),
            "median_rho": float(r.median()) if len(r) else np.nan,
            "wilcoxon_stat": float(stat) if np.isfinite(stat) else np.nan,
            "pval": float(p) if np.isfinite(p) else np.nan,
            "direction_ok": bool(len(r) and np.sign(r.median()) == sign),
        })
    tab = pd.DataFrame(tests)
    tab["qval_bh"] = bh(tab["pval"])
    tab["clears"] = tab["direction_ok"] & (tab["qval_bh"] < 0.10)
    tab.to_csv(OUT / "visium_patient_spearman_tests.csv", index=False)
    note = (
        "P1.3 is omitted: the locked outcome is M2 minus M1, and M1 is not on the stored spot table. "
        "Substituting M2 would be a different estimand."
    )
    return {
        "n_tumor_sections": int(tumor["sample"].nunique()),
        "n_patients": int(tumor["patient"].nunique()),
        "n_clear": int(tab["clears"].sum()),
        "supported": bool(tab["clears"].sum() >= 3),
        "note": note,
        "primary_recurrent_labels": "absent_from_section_names",
    }


def ivy_description() -> dict:
    ivy = pd.read_csv(Path(r"F:\GBM Arterial Territory\analysis\results\expression_ivy_patient_scores.csv"))
    rows = []
    for score in ("barrier_integrity", "barrier_leak", "barrier_pericyte", "homing_ligand"):
        a = ivy[ivy["structure"] == "LE"][["patient", score]].rename(columns={score: "le"})
        b = ivy[ivy["structure"] == "CT"][["patient", score]].rename(columns={score: "ct"})
        m = a.merge(b, on="patient").dropna()
        if len(m) < 5:
            continue
        diff = m["le"] - m["ct"]
        stat, p = stats.wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
        rows.append({
            "score": score, "n_patients": int(len(m)),
            "median_LE_minus_CT": float(diff.median()), "pval": float(p),
        })
    tab = pd.DataFrame(rows)
    if len(tab):
        tab["qval_bh"] = bh(tab["pval"])
    tab.to_csv(OUT / "ivy_barrier_LE_vs_CT.csv", index=False)
    return {"n_tests": int(len(tab)), "source": "existing arterial-territory patient scores, not a new matrix"}


def coordinate_search() -> dict:
    roots = [
        Path(r"G:\GBM Detection\data"),
        Path(r"E:\GBM Detection"),
        Path(r"F:\Radiogenomic Study"),
        Path(r"D:\GBM"),
        Path(r"F:\Glioma Radiotranscriptomics\data"),
    ]
    found = []
    for root in roots:
        if not root.exists():
            continue
        try:
            children = [p.name for p in root.iterdir()]
        except OSError:
            continue
        hits = [n for n in children if re.search(r"lumiere|recur|follow-?up|residual", n, re.I)]
        found.append({"root": str(root), "name_hits": hits})
    return {
        "identifiable": False,
        "reason": "No directory in the searched roots is named as a residual-volume or recurrence-coordinate cohort, and the Ravi section names encode histology region, not primary versus recurrence.",
        "searched": found,
    }


def main() -> None:
    assert_composition_disjoint()
    overlap_rows = []
    for eid, nt, tme, _ in P1:
        inter = sorted(set(PROGRAMS[nt]) & outcome_genes(tme))
        smaller = min(len(PROGRAMS[nt]), len(outcome_genes(tme)))
        overlap_rows.append({
            "endpoint_id": eid, "n_nt": len(PROGRAMS[nt]), "n_tme": len(outcome_genes(tme)),
            "n_overlap": len(inter), "overlap_fraction_of_smaller": len(inter) / smaller,
            "genes": ";".join(inter),
        })
    overlap = pd.DataFrame(overlap_rows)
    overlap.to_csv(OUT / "p1_gene_overlap.csv", index=False)

    dat = load_cgga()
    # Keep the analysis columns only, plus covariates. The clinical merge is wide.
    keep = [c for c in dat.columns if c.startswith("nt_") or c.startswith("tme_") or c in
            {"neuronal_composition", "myeloid_composition", "cohort", "grade_ord", "idh_mutant", "prs_type", "sample_id"}]
    dat = dat[keep]
    cohort_tabs = []
    metas = []
    for model, extra, drop, mask in (
        ("unadjusted", [], False, None),
        ("shared_genes_removed", [], True, None),
        ("composition_adjusted", ["neuronal_composition", "myeloid_composition"], False, None),
        ("composition_adjusted_idhwt_primary", ["neuronal_composition", "myeloid_composition"], False, "idhwt_primary"),
    ):
        sub = dat
        if mask == "idhwt_primary":
            sub = dat[(dat["idh_mutant"] == 0) & (dat["prs_type"].str.lower().str.contains("primary"))].copy()
        print(model, "n", len(sub), flush=True)
        tab, meta = cgga_family(sub, model, extra, drop)
        cohort_tabs.append(tab)
        metas.append(meta)
    pd.concat(cohort_tabs, ignore_index=True).to_csv(OUT / "cgga_models_by_cohort.csv", index=False)
    meta_all = pd.concat(metas, ignore_index=True)
    meta_all.to_csv(OUT / "cgga_meta.csv", index=False)

    unadj = meta_all[meta_all["model"] == "unadjusted"]
    reproduce_ok = bool((unadj["abs_diff_vs_published"] < 0.02).all()) if len(unadj) == 5 else False
    gate_a = gate_from_meta(meta_all[meta_all["model"] == "shared_genes_removed"])
    gate_b = gate_from_meta(meta_all[meta_all["model"] == "composition_adjusted"])
    gate_b_sens = gate_from_meta(meta_all[meta_all["model"] == "composition_adjusted_idhwt_primary"])

    glass = glass_analysis(load_glass_scores())
    vis = visium_gate()
    ivy = ivy_description()
    coord = coordinate_search()

    verdict = {
        "reproduction_of_published_cgga_meta": reproduce_ok,
        "max_abs_diff_vs_published": float(unadj["abs_diff_vs_published"].max()) if len(unadj) else None,
        "gate_A_shared_gene": gate_a,
        "gate_B_beyond_composition": gate_b,
        "gate_B_idhwt_primary_sensitivity": gate_b_sens,
        "gate_C_glass_cotransition": glass["gate_c"],
        "gate_D_lobe": glass["gate_d"],
        "glass_diagnostics": glass["diagnostics"],
        "gate_E_visium_colocalization": vis,
        "ivy_barrier_description": ivy,
        "recurrence_coordinate_vs_residual_volume": coord,
    }
    # Claims the numbers support. Written from the locked rules only.
    claims = {}
    claims["shared_gene_artifact"] = (not reproduce_ok) or gate_a["fail"]
    claims["two_program_association_in_cgga"] = reproduce_ok and (not gate_a["fail"])
    claims["coupling_beyond_composition"] = reproduce_ok and (not gate_b["fail"])
    claims["bulk_programs_move_together_after_therapy"] = bool(glass["gate_c"].get("supported"))
    claims["lobe_does_not_carry_the_association"] = bool(glass["gate_d"].get("location_loses"))
    claims["spots_colocalize"] = bool(vis["supported"])
    claims["recurrence_coordinate_test_was_run"] = False
    claims["prospective_cohort_opened_by_this_analysis"] = False
    verdict["claims"] = claims
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    print(json.dumps(verdict, indent=2, default=str)[:6000])


if __name__ == "__main__":
    main()
