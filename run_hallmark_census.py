"""Hallmark-by-Hallmark census. Rules are in FIELD_PROTOCOL.md.

Survivors are written with claim=False. Do not add a pair to the biological result.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm, rankdata

sys.path.insert(0, str(Path(r"G:\James Zou Project\operative_state")))
import run_gates as rg  # noqa: E402
import run_null_standard as ns  # noqa: E402

OUT = Path(r"G:\James Zou Project\operative_state\results\hallmark_census")
OUT.mkdir(parents=True, exist_ok=True)
GMT = Path(r"G:\GBM Detection\data\references\gene_sets\msigdb\h.all.v2023.2.Hs.symbols.gmt")

MIN_GENES = 20
COEF_FLOOR = 0.20
EXCESS_FLOOR = 0.10
P_FLOOR = 0.10
MIN_SPOTS = 100
MIN_SECTIONS = 8
COMP = set(rg.NEURONAL) | set(rg.MYELOID)


def hc3_matches_statsmodels() -> None:
    rng = np.random.default_rng(0)
    n = 80
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = 0.4 * x - 0.2 * z + rng.normal(size=n)
    frame = pd.DataFrame({"y": y, "x": x, "z": z})
    fit = smf.ols("y ~ x + z", data=frame).fit(cov_type="HC3")
    design = np.column_stack([np.ones(n), x, z])
    beta, se = ols_hc3(y, design)
    if abs(beta[1] - fit.params["x"]) > 1e-8 or abs(se[1] - fit.bse["x"]) > 1e-6:
        raise SystemExit("HC3 does not match statsmodels. Census not run.")


def ols_hc3(y: np.ndarray, design: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xtx = design.T @ design
    beta = np.linalg.solve(xtx, design.T @ y)
    resid = y - design @ beta
    xtx_inv = np.linalg.inv(xtx)
    leverage = np.sum((design @ xtx_inv) * design, axis=1)
    leverage = np.clip(leverage, 0.0, 1.0 - 1e-12)
    weight = np.square(resid / (1.0 - leverage))
    meat = design.T @ (design * weight[:, None])
    cov = xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return beta, se


def load_hallmarks() -> dict[str, list[str]]:
    sets = {}
    for line in GMT.read_text(encoding="utf-8").splitlines():
        parts = line.rstrip("\n").split("\t")
        if parts[0].startswith("HALLMARK_"):
            sets[parts[0]] = parts[2:]
    if len(sets) != 50:
        raise SystemExit(f"Expected 50 Hallmark sets, found {len(sets)}")
    return dict(sorted(sets.items()))


def reduced_lists(sets: dict[str, list[str]]) -> list[dict]:
    names = list(sets)
    rows = []
    for left, right in combinations(names, 2):
        share = set(sets[left]) & set(sets[right])
        a = [g for g in sets[left] if g not in share and g not in COMP]
        b = [g for g in sets[right] if g not in share and g not in COMP]
        rows.append({
            "predictor": left,
            "outcome": right,
            "predictor_genes": a,
            "outcome_genes": b,
            "n_shared_removed": len(share),
            "eligible": len(a) >= MIN_GENES and len(b) >= MIN_GENES,
        })
    return rows


def program_values(centered: np.ndarray, index: pd.Index, genes: list[str]) -> np.ndarray | None:
    present = index.get_indexer([g for g in genes if g in index])
    present = present[present >= 0]
    if len(present) < max(3, int(0.3 * len(genes))):
        return None
    return centered[present].mean(axis=0)


def zscore(values: np.ndarray) -> np.ndarray:
    sd = float(values.std())
    if sd <= 0 or not np.isfinite(sd):
        return np.full(len(values), np.nan)
    return (values - values.mean()) / sd


def two_sided(coef: float, se: float) -> float:
    if not np.isfinite(se) or se <= 0:
        return np.nan
    return float(2 * norm.sf(abs(coef / se)))


def fit_pair(y: np.ndarray, x: np.ndarray, grade: np.ndarray, idh: np.ndarray, neuronal: np.ndarray, myeloid: np.ndarray, composition: bool) -> dict:
    columns = [y, x, grade, idh] + ([neuronal, myeloid] if composition else [])
    stacked = np.column_stack(columns)
    ok = np.isfinite(stacked).all(axis=1)
    stacked = stacked[ok]
    if len(stacked) < 40:
        return {"n": int(len(stacked)), "error": "n<40"}
    yy = zscore(stacked[:, 0])
    xx = zscore(stacked[:, 1])
    gg = zscore(stacked[:, 2])
    ii = stacked[:, 3]
    design_cols = [np.ones(len(stacked)), xx, gg, ii]
    if composition:
        design_cols.extend([zscore(stacked[:, 4]), zscore(stacked[:, 5])])
    design = np.column_stack(design_cols)
    if not np.isfinite(design).all() or not np.isfinite(yy).all():
        return {"n": int(len(stacked)), "error": "nonfinite"}
    beta, se = ols_hc3(yy, design)
    return {"coef": float(beta[1]), "se": float(se[1]), "pval": two_sided(float(beta[1]), float(se[1])), "n": int(len(stacked))}


def load_cgga_bundle(cohort: str, expr_path: Path, clin_path: Path, sets: dict[str, list[str]]) -> dict:
    print(f"CGGA {cohort}", flush=True)
    expr = rg.load_expression(expr_path)
    clin = pd.read_csv(clin_path, sep="\t")
    colmap = {c.lower(): c for c in clin.columns}
    id_col = next(colmap[k] for k in ("cgga_id", "sample", "sampleid", "id") if k in colmap)
    clin["sample_id"] = clin[id_col].astype(str).str.strip()
    expr = expr.rename(columns={c: str(c).strip() for c in expr.columns})
    shared = [c for c in expr.columns if c in set(clin["sample_id"])]
    expr = expr[shared]
    clin = clin.drop_duplicates("sample_id").set_index("sample_id").loc[shared]
    centered = expr.rank(axis=0, method="average").to_numpy()
    n_genes = centered.shape[0]
    centered = (centered - (n_genes + 1) / 2.0) / n_genes
    index = pd.Index(expr.index.astype(str))
    del expr
    roman = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "1": 1, "2": 2, "3": 3, "4": 4}
    gcol = next((colmap[k] for k in ("grade", "who_grade") if k in colmap), None)
    if gcol:
        grade = (
            clin[gcol].astype(str).str.lower().str.replace("who", "", regex=False)
            .str.replace("grade", "", regex=False).str.strip()
            .str.replace(r"[^ivx0-9]", "", regex=True)
        )
        grade_ord = grade.map(roman).astype(float).to_numpy()
    else:
        grade_ord = np.full(len(shared), np.nan)
    idh_col = next((colmap[k] for k in ("idh_mutation_status", "idh_status", "idh_mutation") if k in colmap), None)
    if idh_col:
        low = clin[idh_col].astype(str).str.lower()
        idh = np.where(low.str.contains("wild|wt", regex=True), 0.0, np.where(low.str.contains("mut"), 1.0, np.nan))
    else:
        idh = np.full(len(shared), np.nan)
    neuronal = program_values(centered, index, list(rg.NEURONAL))
    myeloid = program_values(centered, index, list(rg.MYELOID))
    if neuronal is None or myeloid is None:
        raise SystemExit(f"Composition genes missing in {cohort}")
    return {
        "cohort": cohort,
        "centered": centered,
        "index": index,
        "grade": grade_ord,
        "idh": idh.astype(float),
        "neuronal": neuronal,
        "myeloid": myeloid,
        "sets": sets,
    }


def bulk_census(pairs: list[dict], bundles: list[dict]) -> pd.DataFrame:
    per_cohort = []
    for bundle in bundles:
        rows = []
        for pair in pairs:
            if not pair["eligible"]:
                continue
            predictor = program_values(bundle["centered"], bundle["index"], pair["predictor_genes"])
            outcome = program_values(bundle["centered"], bundle["index"], pair["outcome_genes"])
            if predictor is None or outcome is None:
                rows.append({"predictor": pair["predictor"], "outcome": pair["outcome"], "model": "composition", "error": "genes"})
                continue
            for model, composition in (("unadjusted", False), ("composition", True)):
                fit = fit_pair(outcome, predictor, bundle["grade"], bundle["idh"], bundle["neuronal"], bundle["myeloid"], composition)
                rows.append({
                    "predictor": pair["predictor"], "outcome": pair["outcome"],
                    "cohort": bundle["cohort"], "model": model, **fit,
                })
        per_cohort.append(pd.DataFrame(rows))
        print(bundle["cohort"], "pairs fit", flush=True)
    detail = pd.concat(per_cohort, ignore_index=True)
    detail.to_csv(OUT / "cgga_by_cohort.csv", index=False)
    meta_rows = []
    ok = detail.dropna(subset=["coef", "se"]) if "coef" in detail.columns else detail.iloc[0:0]
    for (predictor, outcome, model), group in ok.groupby(["predictor", "outcome", "model"]):
        if group["cohort"].nunique() < 2:
            continue
        coef, se, pval = rg.meta_iv(group["coef"].to_numpy(), group["se"].to_numpy())
        meta_rows.append({
            "predictor": predictor, "outcome": outcome, "model": model,
            "meta_coef": coef, "meta_se": se, "meta_pval": pval, "n_total": int(group["n"].sum()),
        })
    return pd.DataFrame(meta_rows)


def sections() -> list[Path]:
    return sorted(
        p for p in ns.VISIUM.iterdir()
        if p.is_dir() and "_T_ST" in p.name and "_TI_" not in p.name and "_TC_" not in p.name and "_C_ST" not in p.name
    )


def bin_groups(neuronal: np.ndarray, myeloid: np.ndarray) -> list[np.ndarray]:
    frame = pd.DataFrame({"n": neuronal, "m": myeloid})
    frame["nq"] = pd.qcut(frame["n"], 4, labels=False, duplicates="drop")
    frame["mq"] = pd.qcut(frame["m"], 4, labels=False, duplicates="drop")
    frame["bin"] = frame["nq"].astype(str) + "_" + frame["mq"].astype(str)
    groups = []
    for key in frame["bin"].unique():
        positions = np.flatnonzero(frame["bin"].to_numpy() == key)
        if len(positions) >= 5:
            groups.append(positions)
    return groups


def centered_ranks(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ranks = rankdata(values, axis=0, method="average")
    ranks = ranks - ranks.mean(axis=0)
    norms = np.sqrt(np.sum(ranks * ranks, axis=0))
    return ranks, norms


def spatial_census(pairs: list[dict]) -> pd.DataFrame:
    eligible = [p for p in pairs if p["eligible"]]
    n_pairs = len(eligible)
    obs_sections: list[np.ndarray] = []
    null_sections: list[np.ndarray] = []
    n_spot_sections: list[np.ndarray] = []
    for section_index, folder in enumerate(sections()):
        h5_path = folder / "outs" / "filtered_feature_bc_matrix.h5"
        counts, names = ns.read_visium_counts(h5_path)
        ranks = rankdata(counts, axis=0, method="average")
        n_genes = ranks.shape[0]
        centered = (ranks - (n_genes + 1) / 2.0) / n_genes
        index = pd.Index(names)
        del counts, ranks
        neuronal = program_values(centered, index, list(rg.NEURONAL))
        myeloid = program_values(centered, index, list(rg.MYELOID))
        if neuronal is None or myeloid is None:
            raise SystemExit(f"Composition genes missing in {folder.name}")
        keep = np.isfinite(neuronal) & np.isfinite(myeloid)
        predictor = np.zeros((int(keep.sum()), n_pairs))
        outcome = np.zeros((int(keep.sum()), n_pairs))
        usable = np.zeros(n_pairs, dtype=bool)
        for i, pair in enumerate(eligible):
            pred = program_values(centered, index, pair["predictor_genes"])
            out = program_values(centered, index, pair["outcome_genes"])
            if pred is None or out is None:
                continue
            predictor[:, i] = pred[keep]
            outcome[:, i] = out[keep]
            usable[i] = True
        n_spots = int(keep.sum())
        print(folder.name, "spots", n_spots, "pairs", int(usable.sum()), flush=True)
        if n_spots < MIN_SPOTS:
            continue
        pred_rank, pred_norm = centered_ranks(predictor)
        out_rank, out_norm = centered_ranks(outcome)
        denom = pred_norm * out_norm
        observed = np.divide(
            np.sum(pred_rank * out_rank, axis=0), denom,
            out=np.full(n_pairs, np.nan), where=np.isfinite(denom) & (denom > 0),
        )
        observed[~usable] = np.nan
        groups = bin_groups(neuronal[keep], myeloid[keep])
        rng = np.random.default_rng(np.random.SeedSequence([ns.SEED, section_index]))
        null = np.empty((ns.N_PERM, n_pairs))
        order = np.arange(n_spots)
        for draw in range(ns.N_PERM):
            perm = order.copy()
            for positions in groups:
                perm[positions] = rng.permutation(perm[positions])
            null[draw] = np.divide(
                np.sum(pred_rank[perm] * out_rank, axis=0), denom,
                out=np.full(n_pairs, np.nan), where=np.isfinite(denom) & (denom > 0),
            )
        null[:, ~usable] = np.nan
        obs_sections.append(observed)
        null_sections.append(null)
        n_spot_sections.append(np.full(n_pairs, n_spots))
    obs_mat = np.vstack(obs_sections)
    # null_sections[s] is (perm, pair). Cohort null draw is the mean across sections.
    null_cube = np.stack(null_sections, axis=0)  # section, perm, pair
    rows = []
    for i, pair in enumerate(eligible):
        finite = np.isfinite(obs_mat[:, i])
        n_sec = int(finite.sum())
        if n_sec < MIN_SECTIONS:
            rows.append({
                "predictor": pair["predictor"], "outcome": pair["outcome"],
                "n_sections": n_sec, "interpretable": False,
            })
            continue
        obs_mean = float(np.nanmean(obs_mat[:, i]))
        null_means = np.nanmean(null_cube[:, :, i], axis=0)
        center = float(np.median(null_means))
        pvalue = (1 + np.sum(np.abs(null_means) >= abs(obs_mean))) / (ns.N_PERM + 1)
        rows.append({
            "predictor": pair["predictor"], "outcome": pair["outcome"],
            "n_sections": n_sec, "interpretable": True,
            "mean_rho": obs_mean, "null_center": center,
            "excess": obs_mean - center, "perm_p": float(pvalue),
        })
    return pd.DataFrame(rows)


def classify(meta: pd.DataFrame, spatial: pd.DataFrame, pairs: list[dict]) -> pd.DataFrame:
    unadj = meta.loc[meta["model"] == "unadjusted"].set_index(["predictor", "outcome"])
    comp = meta.loc[meta["model"] == "composition"].set_index(["predictor", "outcome"])
    space = spatial.set_index(["predictor", "outcome"])
    rows = []
    for pair in pairs:
        key = (pair["predictor"], pair["outcome"])
        row = {
            "predictor": pair["predictor"], "outcome": pair["outcome"],
            "n_predictor_genes": len(pair["predictor_genes"]),
            "n_outcome_genes": len(pair["outcome_genes"]),
            "n_shared_removed": pair["n_shared_removed"],
            "eligible": pair["eligible"],
            "claim": False,
        }
        if not pair["eligible"]:
            row["bulk_status"] = "ineligible"
            row["spatial_status"] = "ineligible"
        else:
            if key not in comp.index or key not in unadj.index:
                row["bulk_status"] = "uninterpretable"
            else:
                coef = float(comp.loc[key, "meta_coef"])
                pval = float(comp.loc[key, "meta_pval"])
                base = float(unadj.loc[key, "meta_coef"])
                row["bulk_unadjusted"] = base
                row["bulk_adjusted"] = coef
                row["bulk_p"] = pval
                row["bulk_n"] = int(comp.loc[key, "n_total"])
                same_sign = np.sign(coef) == np.sign(base) and base != 0
                row["bulk_status"] = "survives" if same_sign and abs(coef) >= COEF_FLOOR and pval < P_FLOOR else "dies"
            if key not in space.index or not bool(space.loc[key, "interpretable"]):
                row["spatial_status"] = "uninterpretable"
            else:
                excess = float(space.loc[key, "excess"])
                pval = float(space.loc[key, "perm_p"])
                row["spatial_excess"] = excess
                row["spatial_mean_rho"] = float(space.loc[key, "mean_rho"])
                row["spatial_null_center"] = float(space.loc[key, "null_center"])
                row["spatial_p"] = pval
                row["spatial_n_sections"] = int(space.loc[key, "n_sections"])
                row["spatial_status"] = "survives" if abs(excess) >= EXCESS_FLOOR and pval < P_FLOOR else "dies"
        rows.append(row)
    return pd.DataFrame(rows)


def fraction(status: pd.Series, value: str, among: pd.Series) -> float:
    base = status.loc[among]
    if len(base) == 0:
        return np.nan
    return float((base == value).mean())


def main() -> None:
    hc3_matches_statsmodels()
    sets = load_hallmarks()
    pairs = reduced_lists(sets)
    pd.DataFrame([{
        "predictor": p["predictor"], "outcome": p["outcome"],
        "n_predictor_genes": len(p["predictor_genes"]),
        "n_outcome_genes": len(p["outcome_genes"]),
        "n_shared_removed": p["n_shared_removed"],
        "eligible": p["eligible"],
    } for p in pairs]).to_csv(OUT / "pairs.csv", index=False)
    base = Path(r"E:\Glycocalyx Immune Escape\data\raw\cgga")
    bundles = [
        load_cgga_bundle(
            "mRNAseq_325",
            base / "mRNAseq_325" / "CGGA.mRNAseq_325.RSEM-genes.20200506.txt",
            base / "mRNAseq_325" / "CGGA.mRNAseq_325_clinical.20200506.txt",
            sets,
        ),
        load_cgga_bundle(
            "mRNAseq_693",
            base / "mRNAseq_693" / "CGGA.mRNAseq_693.RSEM-genes.20200506.txt",
            base / "mRNAseq_693" / "CGGA.mRNAseq_693_clinical.20200506.txt",
            sets,
        ),
    ]
    meta = bulk_census(pairs, bundles)
    meta.to_csv(OUT / "cgga_meta.csv", index=False)
    for bundle in bundles:
        del bundle["centered"]
    spatial = spatial_census(pairs)
    spatial.to_csv(OUT / "visium_excess.csv", index=False)
    table = classify(meta, spatial, pairs)
    table.to_csv(OUT / "census.csv", index=False)
    eligible = table["eligible"] == True  # noqa: E712
    bulk_known = table["bulk_status"].isin(["survives", "dies"])
    space_known = table["spatial_status"].isin(["survives", "dies"])
    both_known = bulk_known & space_known
    either_dies = both_known & ((table["bulk_status"] == "dies") | (table["spatial_status"] == "dies"))
    both_survive = both_known & (table["bulk_status"] == "survives") & (table["spatial_status"] == "survives")
    summary = {
        "n_pairs": int(len(table)),
        "n_eligible": int(eligible.sum()),
        "bulk_die_fraction": fraction(table["bulk_status"], "dies", eligible & bulk_known),
        "bulk_survive_fraction": fraction(table["bulk_status"], "survives", eligible & bulk_known),
        "n_bulk_interpretable": int((eligible & bulk_known).sum()),
        "spatial_die_fraction": fraction(table["spatial_status"], "dies", eligible & space_known),
        "spatial_survive_fraction": fraction(table["spatial_status"], "survives", eligible & space_known),
        "n_spatial_interpretable": int((eligible & space_known).sum()),
        "either_layer_die_fraction": float(either_dies.sum() / both_known.sum()) if both_known.any() else None,
        "both_layers_survive_fraction": float(both_survive.sum() / both_known.sum()) if both_known.any() else None,
        "n_both_interpretable": int(both_known.sum()),
        "n_survivors_not_claims": int(both_survive.sum()),
        "claims_added": 0,
        "recording_study_opened": False,
    }
    (OUT / "verdict.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
