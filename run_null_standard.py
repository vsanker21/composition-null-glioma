"""Composition null. Rules are in NULL_STANDARD_PROTOCOL.md. Do not retune after the run."""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import sparse
from scipy.stats import rankdata

sys.path.insert(0, str(Path(r"G:\James Zou Project\operative_state")))
import run_gates as rg  # noqa: E402

OUT = Path(r"G:\James Zou Project\operative_state\results\null_standard")
OUT.mkdir(parents=True, exist_ok=True)

VISIUM = Path(r"G:\GBM Detection\data\dryad\spatial_gbm\10XVisium 2")
N_PERM = 500
SEED = 11

CELL_A = ["MKI67", "TOP2A", "CDK1", "CCNB1", "BIRC5", "NUSAP1"]
CELL_B = ["PCNA", "MCM2", "TYMS", "RRM2", "UBE2C", "CENPF"]
MES = ["CD44", "S100A4", "VIM", "CHI3L1", "SERPINE1"]
NPC = ["SOX2", "ASCL1", "DLL3", "HES1"]
HYPOXIA = ["CA9", "VEGFA", "SLC2A1", "LDHA", "PGK1", "ADM", "NDRG1", "P4HA1", "PDK1", "EGLN3"]

# name -> genes. Composition scores are computed separately.
PROGRAMS = {
    "purinergic_adenosine": rg.PROGRAMS["purinergic_adenosine"],
    "gaba_receptor": rg.PROGRAMS["gaba_receptor"],
    "dopamine_receptor": rg.PROGRAMS["dopamine_receptor"],
    "glutamate_ligand": rg.PROGRAMS["glutamate_ligand"],
    "glutamate_receptor": rg.PROGRAMS["glutamate_receptor"],
    "macrophage_m2_like": rg.IMMUNE["macrophage_m2_like"],
    "macrophage_m1_like": rg.IMMUNE["macrophage_m1_like"],
    "immune_estimate_like": rg.IMMUNE["immune_estimate_like"],
    "microglia_like": rg.IMMUNE["microglia_like"],
    "cell_cycle_a": CELL_A,
    "cell_cycle_b": CELL_B,
    "mes": MES,
    "npc": NPC,
    "hypoxia": HYPOXIA,
    "neuronal_composition": rg.NEURONAL,
    "myeloid_composition": rg.MYELOID,
}

# (id, family, predictor, outcome, literature sign or None)
COUPLINGS = [
    ("P1.1", "calibration", "purinergic_adenosine", "macrophage_m2_like", 1),
    ("P1.2", "calibration", "gaba_receptor", "immune_estimate_like", -1),
    ("P1.3", "calibration", "dopamine_receptor", "m2_minus_m1", -1),
    ("P1.4", "calibration", "glutamate_ligand", "immune_estimate_like", -1),
    ("P1.5", "calibration", "glutamate_receptor", "microglia_like", -1),
    ("S1", "specificity", "cell_cycle_a", "cell_cycle_b", 1),
    ("C1", "census", "mes", "macrophage_m2_like", 1),
    ("C2", "census", "hypoxia", "mes", 1),
    ("C3", "census", "npc", "mes", -1),
]


def program_genes(name: str) -> list[str]:
    if name == "m2_minus_m1":
        return sorted(set(PROGRAMS["macrophage_m2_like"]) | set(PROGRAMS["macrophage_m1_like"]))
    return list(PROGRAMS[name])


def drop_composition_genes(genes: list[str]) -> tuple[list[str], list[str]]:
    banned = set(rg.NEURONAL) | set(rg.MYELOID)
    dropped = [g for g in genes if g in banned]
    kept = [g for g in genes if g not in banned]
    return kept, dropped


def prepare_sets() -> pd.DataFrame:
    rows = []
    for name, genes in PROGRAMS.items():
        kept, dropped = drop_composition_genes(genes) if name not in {"neuronal_composition", "myeloid_composition"} else (list(genes), [])
        PROGRAMS[name] = kept
        rows.append({"program": name, "n_defined": len(genes), "n_kept": len(kept), "dropped": ";".join(dropped)})
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "gene_sets_after_composition_drop.csv", index=False)
    return tab


def score_frame(expr: pd.DataFrame) -> pd.DataFrame:
    centered = rg.centered_ranks(expr)
    scores = {}
    for name, genes in PROGRAMS.items():
        scores[name] = rg.singscore_from_ranks(centered, genes, len(genes) if genes else 1)
    scores["m2_minus_m1"] = scores["macrophage_m2_like"] - scores["macrophage_m1_like"]
    return pd.DataFrame(scores)


def fit_one(df: pd.DataFrame, y: str, x: str, extra: list[str]) -> dict:
    cols = [y, x] + extra
    d = df[cols].apply(pd.to_numeric, errors="coerce").dropna().copy()
    if len(d) < 40:
        return {"n": int(len(d)), "error": "n<40"}
    for c in d.columns:
        if c == "idh_mutant":
            continue
        d[c] = rg.zscore(d[c])
    rhs = " + ".join([x] + extra)
    model = smf.ols(f"{y} ~ {rhs}", data=d).fit(cov_type="HC3")
    return {
        "coef": float(model.params[x]),
        "se": float(model.bse[x]),
        "pval": float(model.pvalues[x]),
        "n": int(model.nobs),
        "r2": float(model.rsquared),
    }


def load_cgga_scores() -> pd.DataFrame:
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
        scores = score_frame(expr)
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


def cgga_models(dat: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort, sub in dat.groupby("cohort"):
        for cid, family, pred, outcome, lit in COUPLINGS:
            y, x = outcome, pred
            for model, extra in (
                ("unadjusted", ["grade_ord", "idh_mutant"]),
                ("composition", ["grade_ord", "idh_mutant", "neuronal_composition", "myeloid_composition"]),
            ):
                try:
                    fit = fit_one(sub, y, x, extra)
                except Exception as exc:  # noqa: BLE001
                    fit = {"error": str(exc)}
                rows.append({
                    "coupling": cid, "family": family, "cohort": cohort, "model": model,
                    "literature_sign": lit, **fit,
                })
    tab = pd.DataFrame(rows)
    parts = []
    for (_, _), group in tab.groupby(["family", "model"]):
        group = group.copy()
        group["qval_bh"] = rg.bh(group["pval"]) if "pval" in group else np.nan
        parts.append(group)
    by_cohort = pd.concat(parts, ignore_index=True)
    by_cohort.to_csv(OUT / "cgga_by_cohort.csv", index=False)
    meta_rows = []
    ok = by_cohort.dropna(subset=["coef", "se"]) if "coef" in by_cohort else by_cohort
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
            "literature_sign": int(group["literature_sign"].iloc[0]),
        })
    meta = pd.DataFrame(meta_rows)
    parts = []
    for (_, _), group in meta.groupby(["family", "model"]):
        group = group.copy()
        group["qval_bh"] = rg.bh(group["meta_pval"])
        parts.append(group)
    meta = pd.concat(parts, ignore_index=True)
    return meta


def pair_status(meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cid, group in meta.groupby("coupling"):
        unadj = group[group["model"] == "unadjusted"].iloc[0]
        adj = group[group["model"] == "composition"].iloc[0]
        if unadj["qval_bh"] >= 0.10:
            status = "absent"
        elif np.sign(adj["meta_coef"]) != np.sign(unadj["meta_coef"]) or adj["qval_bh"] >= 0.10:
            status = "killed"
        else:
            status = "survives"
        rows.append({
            "coupling": cid,
            "family": unadj["family"],
            "unadj_coef": unadj["meta_coef"],
            "unadj_q": unadj["qval_bh"],
            "adj_coef": adj["meta_coef"],
            "adj_q": adj["qval_bh"],
            "attenuation": adj["meta_coef"] / unadj["meta_coef"] if unadj["meta_coef"] else np.nan,
            "literature_sign": int(unadj["literature_sign"]),
            "literature_concordant_unadj": bool(np.sign(unadj["meta_coef"]) == unadj["literature_sign"]),
            "status": status,
            "n_total": int(unadj["n_total"]),
        })
    return pd.DataFrame(rows)


def bulk_valid(status: pd.DataFrame) -> dict:
    calib = status[status["family"] == "calibration"].set_index("coupling")
    killed = [cid for cid in ("P1.2", "P1.3", "P1.4", "P1.5") if calib.loc[cid, "status"] == "killed"]
    spec = status[status["coupling"] == "S1"].iloc[0]
    specificity_ok = bool(spec["status"] == "survives" and spec["adj_coef"] > 0)
    sensitivity_ok = len(killed) >= 3
    return {
        "sensitivity_ok": sensitivity_ok,
        "n_of_p12_to_p15_killed": len(killed),
        "killed": killed,
        "specificity_ok": specificity_ok,
        "cell_cycle_status": spec["status"],
        "cell_cycle_adj_coef": float(spec["adj_coef"]),
        "procedure_valid": bool(sensitivity_ok and specificity_ok),
    }


def load_glass_wide() -> pd.DataFrame:
    print("GLASS", flush=True)
    expr = rg.load_expression(Path(r"G:\GBM Detection\data\natcomms\glass\glass_gene_tpm_matrix_all_samples.tsv"))
    parsed = [rg.parse_glass_column(c) for c in expr.columns]
    keep = [i for i, (case, time) in enumerate(parsed) if case and time in {"TP", "R1"}]
    expr = expr.iloc[:, keep]
    scores = score_frame(expr)
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
    primary = wide[(wide["idh_wt"] == True) & (wide["gbm"] == True)].copy()  # noqa: E712
    return primary


def glass_models(primary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if len(primary) < 30:
        return pd.DataFrame()
    for cid, family, pred, outcome, lit in COUPLINGS:
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
                "r2": float(fit.rsquared), "literature_sign": lit,
                "median_delta_predictor": float(d["dx"].median()),
                "median_delta_outcome": float(d["dy"].median()),
            })
    tab = pd.DataFrame(rows)
    parts = []
    for (_, _), group in tab.groupby(["family", "model"]):
        group = group.copy()
        group["qval_bh"] = rg.bh(group["meta_pval"])
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def read_visium_counts(path: Path) -> tuple[np.ndarray, list[str]]:
    with h5py.File(path, "r") as handle:
        matrix = handle["matrix"]
        shape = matrix["shape"][:].astype(int)
        data = matrix["data"][:]
        indices = matrix["indices"][:]
        indptr = matrix["indptr"][:]
        names = np.array([n.decode() if isinstance(n, bytes) else str(n) for n in matrix["features"]["name"][:]])
    counts = sparse.csc_matrix((data, indices, indptr), shape=tuple(shape)).toarray().astype(np.float64)
    # 10x feature names are not unique. Sum counts within a symbol before ranking.
    if len(names) != len(set(names.tolist())):
        unique, inverse = np.unique(names, return_inverse=True)
        collapsed = np.zeros((len(unique), counts.shape[1]), dtype=np.float64)
        np.add.at(collapsed, inverse, counts)
        counts = collapsed
        names = unique
    return counts, names.tolist()


def singscore_spots(counts: np.ndarray, gene_names: list[str]) -> pd.DataFrame:
    ranks = rankdata(counts, axis=0, method="average")
    n_genes = ranks.shape[0]
    centered = (ranks - (n_genes + 1) / 2.0) / n_genes
    index = pd.Index(gene_names)
    out = {}
    for name, genes in PROGRAMS.items():
        present = [g for g in genes if g in index]
        if len(present) < max(3, int(0.3 * max(len(genes), 1))):
            out[name] = np.full(counts.shape[1], np.nan)
            continue
        positions = index.get_indexer(present)
        out[name] = centered[positions].mean(axis=0)
    out["m2_minus_m1"] = out["macrophage_m2_like"] - out["macrophage_m1_like"]
    return pd.DataFrame(out)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = rankdata(a, method="average")
    rb = rankdata(b, method="average")
    a0 = ra - ra.mean()
    b0 = rb - rb.mean()
    denom = np.sqrt(np.dot(a0, a0) * np.dot(b0, b0))
    if denom == 0:
        return np.nan
    return float(np.dot(a0, b0) / denom)


def stratified_draws(predictor: np.ndarray, outcome: np.ndarray, neuronal: np.ndarray, myeloid: np.ndarray, rng: np.random.Generator) -> tuple[float, np.ndarray]:
    """Observed Spearman and the composition-bin permutation distribution."""
    observed = spearman(predictor, outcome)
    frame = pd.DataFrame({"n": neuronal, "m": myeloid})
    try:
        frame["nq"] = pd.qcut(frame["n"], 4, labels=False, duplicates="drop")
        frame["mq"] = pd.qcut(frame["m"], 4, labels=False, duplicates="drop")
    except ValueError:
        return observed, np.full(N_PERM, np.nan)
    frame["bin"] = frame["nq"].astype(str) + "_" + frame["mq"].astype(str)
    groups = [np.flatnonzero(frame["bin"].to_numpy() == key) for key in frame["bin"].unique()]
    groups = [positions for positions in groups if len(positions) >= 5]
    null = np.empty(N_PERM)
    order = np.arange(len(predictor))
    for draw in range(N_PERM):
        perm = order.copy()
        for positions in groups:
            perm[positions] = rng.permutation(perm[positions])
        null[draw] = spearman(predictor[perm], outcome)
    return observed, null


def spatial_layer(rng: np.random.Generator) -> pd.DataFrame:
    """Cohort p-value is the permutation distribution of the mean across sections."""
    section_rows = []
    draws: dict[str, list[np.ndarray]] = {cid: [] for cid, *_rest in COUPLINGS}
    observed: dict[str, list[float]] = {cid: [] for cid, *_rest in COUPLINGS}
    families = {cid: family for cid, family, *_rest in COUPLINGS}
    folders = sorted(
        p for p in VISIUM.iterdir()
        if p.is_dir() and "_T_ST" in p.name and "_TI_" not in p.name and "_TC_" not in p.name and "_C_ST" not in p.name
    )
    print(f"Visium tumor sections {len(folders)}", flush=True)
    for folder in folders:
        h5_path = folder / "outs" / "filtered_feature_bc_matrix.h5"
        if not h5_path.exists():
            continue
        print(folder.name, flush=True)
        counts, names = read_visium_counts(h5_path)
        scores = singscore_spots(counts, names)
        del counts
        keep = scores[["neuronal_composition", "myeloid_composition"]].notna().all(axis=1)
        scores = scores.loc[keep]
        if len(scores) < 100:
            continue
        sample = folder.name.lstrip("#")
        neuronal = scores["neuronal_composition"].to_numpy()
        myeloid = scores["myeloid_composition"].to_numpy()
        for cid, _family, pred, outcome, _lit in COUPLINGS:
            mask = (
                np.isfinite(scores[pred].to_numpy())
                & np.isfinite(scores[outcome].to_numpy())
                & np.isfinite(neuronal)
                & np.isfinite(myeloid)
            )
            if int(mask.sum()) < 100:
                continue
            rho, null = stratified_draws(
                scores[pred].to_numpy()[mask], scores[outcome].to_numpy()[mask],
                neuronal[mask], myeloid[mask], rng,
            )
            if not np.isfinite(rho):
                continue
            observed[cid].append(rho)
            draws[cid].append(null)
            section_rows.append({
                "sample": sample, "coupling": cid, "n_spots": int(len(scores)),
                "rho": rho, "null_median": float(np.nanmedian(null)),
            })
    pd.DataFrame(section_rows).to_csv(OUT / "visium_section_permutation.csv", index=False)
    rows = []
    for cid, rhos in observed.items():
        if not rhos:
            continue
        obs_mean = float(np.mean(rhos))
        # Each section's draw i is an independent permutation. The mean of draw i
        # across sections is one null replicate of the cross-section mean.
        stacked = np.vstack(draws[cid])
        null_means = np.nanmean(stacked, axis=0)
        pvalue = (1 + np.sum(np.abs(null_means) >= abs(obs_mean))) / (N_PERM + 1)
        rows.append({
            "coupling": cid,
            "family": families[cid],
            "n_sections": len(rhos),
            "mean_rho": obs_mean,
            "median_rho": float(np.median(rhos)),
            "null_mean_median": float(np.median(null_means)),
            "perm_p": float(pvalue),
        })
    summary = pd.DataFrame(rows)
    parts = []
    for _family, group in summary.groupby("family"):
        group = group.copy()
        group["qval_bh"] = rg.bh(group["perm_p"])
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def spatial_status(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in summary.itertuples(index=False):
        if row.qval_bh < 0.10:
            status = "survives"
        elif abs(row.mean_rho) >= 0.05:
            status = "killed"
        else:
            status = "absent"
        rows.append({**row._asdict(), "status": status})
    return pd.DataFrame(rows)


def main() -> None:
    prepare_sets()
    # Composition sets must still be the full locked lists.
    assert PROGRAMS["neuronal_composition"] == rg.NEURONAL
    assert PROGRAMS["myeloid_composition"] == rg.MYELOID

    cgga = load_cgga_scores()
    meta = cgga_models(cgga)
    meta.to_csv(OUT / "cgga_meta.csv", index=False)
    status = pair_status(meta)
    status.to_csv(OUT / "cgga_status.csv", index=False)
    validity = bulk_valid(status)

    glass = load_glass_wide()
    glass_tab = glass_models(glass)
    glass_tab.to_csv(OUT / "glass_models.csv", index=False)
    glass_status = pair_status(glass_tab) if len(glass_tab) else pd.DataFrame()
    if len(glass_status):
        glass_status.to_csv(OUT / "glass_status.csv", index=False)

    rng = np.random.default_rng(SEED)
    spatial = spatial_status(spatial_layer(rng))
    spatial.to_csv(OUT / "visium_status.csv", index=False)
    spec_spatial = spatial[spatial["coupling"] == "S1"].iloc[0]
    spatial_valid = bool(spec_spatial["status"] == "survives" and spec_spatial["mean_rho"] > 0)

    census = status[status["family"] == "census"][["coupling", "status", "unadj_coef", "adj_coef", "unadj_q", "adj_q"]].rename(
        columns={"status": "cgga_status"}
    )
    if len(glass_status):
        census = census.merge(
            glass_status[["coupling", "status", "unadj_coef", "adj_coef"]].rename(
                columns={"status": "glass_status", "unadj_coef": "glass_unadj", "adj_coef": "glass_adj"}
            ),
            on="coupling", how="left",
        )
    census = census.merge(
        spatial[spatial["family"] == "census"][["coupling", "status", "mean_rho", "qval_bh"]].rename(
            columns={"status": "spatial_status", "qval_bh": "spatial_q"}
        ),
        on="coupling", how="left",
    )
    if validity["procedure_valid"] and spatial_valid:
        census["full_lock"] = (
            (census["cgga_status"] == "survives")
            & (census["glass_status"] == "survives")
            & (census["spatial_status"] == "survives")
        )
    else:
        census["full_lock"] = pd.NA
    census.to_csv(OUT / "census.csv", index=False)

    verdict = {
        "bulk_validity": validity,
        "n_cgga": int(len(cgga)),
        "n_glass_pairs": int(len(glass)),
        "spatial_layer_valid": spatial_valid,
        "spatial_cell_cycle_status": spec_spatial["status"],
        "spatial_cell_cycle_mean_rho": float(spec_spatial["mean_rho"]),
        "spatial_cell_cycle_q": float(spec_spatial["qval_bh"]),
        "census_interpretable": bool(validity["procedure_valid"]),
        "full_lock_interpretable": bool(validity["procedure_valid"] and spatial_valid),
        "n_census_full_lock": int(census["full_lock"].fillna(False).sum()) if census["full_lock"].notna().any() else 0,
    }
    import json
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(census.to_string(index=False))
    print(verdict)


if __name__ == "__main__":
    main()
