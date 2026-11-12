"""Smooth composition null for the locked F1 spatial claim.

Rules are in FIELD_PROTOCOL.md. Do not change the basis or the floor after reading the excess.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from patsy import dmatrix

sys.path.insert(0, str(Path(r"G:\James Zou Project\operative_state")))
import run_full_list as fl  # noqa: E402
import run_null_standard as ns  # noqa: E402

OUT = Path(r"G:\James Zou Project\operative_state\results\smooth_null")
OUT.mkdir(parents=True, exist_ok=True)
LABELS = Path(r"G:\James Zou Project\operative_state\results\niche_visium\spot_labels.csv")

EXCESS_FLOOR = 0.10
P_FLOOR = 0.10
MIN_SPOTS = 100
MIN_SECTIONS = 8
# Locked observed mean on primary spots. A mismatch means the spot join is wrong.
LOCKED_PRIMARY_MEAN_RHO = 0.5888787494211498


def sections() -> list[Path]:
    return sorted(
        p for p in ns.VISIUM.iterdir()
        if p.is_dir() and "_T_ST" in p.name and "_TI_" not in p.name and "_TC_" not in p.name and "_C_ST" not in p.name
    )


def residualize(y: np.ndarray, design: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ beta


def design_smooth(neuronal: np.ndarray, myeloid: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame({"n": neuronal, "m": myeloid})
    built = dmatrix("bs(n, df=4):bs(m, df=4)", frame, return_type="dataframe")
    if built.shape[1] != 17:
        raise SystemExit(f"Spline basis has {built.shape[1]} columns, not the locked 17")
    return np.asarray(built, dtype=float)


def design_linear(neuronal: np.ndarray, myeloid: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(neuronal)), neuronal, myeloid])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return ns.spearman(a, b)


def residual_permutation(hypoxia: np.ndarray, mes: np.ndarray, design: np.ndarray, rng: np.random.Generator) -> tuple[float, np.ndarray]:
    hx = residualize(hypoxia, design)
    my = residualize(mes, design)
    observed = spearman(hx, my)
    null = np.empty(ns.N_PERM)
    for draw in range(ns.N_PERM):
        null[draw] = spearman(rng.permutation(hx), my)
    return observed, null


def cohort(rows: list[dict], key: str) -> dict:
    part = [r for r in rows if r["subset"] == key and np.isfinite(r["rho"]) and np.isfinite(r["null_median"])]
    if len(part) < MIN_SECTIONS:
        return {"subset": key, "n_sections": len(part), "interpretable": False}
    obs = np.array([r["rho"] for r in part])
    # Rebuild the cohort null from stored draws.
    return {"subset": key, "n_sections": len(part), "interpretable": True, "obs": obs}


def main() -> None:
    gene_col = pd.read_csv(
        Path(r"E:\Glycocalyx Immune Escape\data\raw\cgga\mRNAseq_325\CGGA.mRNAseq_325.RSEM-genes.20200506.txt"),
        sep="\t", usecols=[0],
    ).iloc[:, 0].astype(str)
    previous = fl.OUT
    fl.OUT = OUT
    fl.build_sets(set(gene_col))
    fl.OUT = previous

    labels = pd.read_csv(LABELS, usecols=["barcode", "sample", "role"])
    rng = np.random.default_rng(ns.SEED)
    section_rows = []
    draws: dict[str, list[np.ndarray]] = {"smooth_primary": [], "linear_primary": [], "smooth_stained": []}
    observed: dict[str, list[float]] = {key: [] for key in draws}
    primary_rhos = []
    for folder in sections():
        sample = folder.name.lstrip("#")
        h5_path = folder / "outs" / "filtered_feature_bc_matrix.h5"
        counts, names = ns.read_visium_counts(h5_path)
        with h5py.File(h5_path, "r") as handle:
            barcodes = [b.decode() if isinstance(b, bytes) else str(b) for b in handle["matrix"]["barcodes"][:]]
        scores = fl.score_spots(counts, names)
        del counts
        scores.insert(0, "barcode", barcodes)
        lab = labels.loc[labels["sample"] == sample, ["barcode", "role"]]
        merged = scores.merge(lab, on="barcode", how="inner")
        finite = (
            np.isfinite(merged["hypoxia_f1"])
            & np.isfinite(merged["mes1_f1"])
            & np.isfinite(merged["neuronal_composition"])
            & np.isfinite(merged["myeloid_composition"])
        )
        primary = merged.loc[(merged["role"] == "primary") & finite]
        stained = merged.loc[merged["role"].isin(["primary", "rim", "necrosis"]) & finite]
        if len(primary) >= MIN_SPOTS:
            primary_rhos.append(ns.spearman(primary["hypoxia_f1"].to_numpy(), primary["mes1_f1"].to_numpy()))
        subsets = {
            "smooth_primary": (primary, design_smooth),
            "linear_primary": (primary, design_linear),
            "smooth_stained": (stained, design_smooth),
        }
        for key, (part, builder) in subsets.items():
            if len(part) < MIN_SPOTS:
                section_rows.append({"sample": sample, "subset": key, "n_spots": int(len(part)), "rho": np.nan, "null_median": np.nan})
                continue
            design = builder(part["neuronal_composition"].to_numpy(), part["myeloid_composition"].to_numpy())
            rho, null = residual_permutation(
                part["hypoxia_f1"].to_numpy(),
                part["mes1_f1"].to_numpy(),
                design,
                rng,
            )
            section_rows.append({
                "sample": sample, "subset": key, "n_spots": int(len(part)),
                "rho": rho, "null_median": float(np.median(null)),
            })
            observed[key].append(rho)
            draws[key].append(null)
        print(sample, "residualized", len(primary), flush=True)

    mean_primary = float(np.mean(primary_rhos)) if primary_rhos else np.nan
    if not np.isfinite(mean_primary) or abs(mean_primary - LOCKED_PRIMARY_MEAN_RHO) > 1e-6:
        raise SystemExit(f"Primary Spearman mean {mean_primary} does not match the locked niche result {LOCKED_PRIMARY_MEAN_RHO}")

    summary_rows = []
    for key, rhos in observed.items():
        if len(rhos) < MIN_SECTIONS:
            summary_rows.append({"subset": key, "n_sections": len(rhos), "interpretable": False})
            continue
        obs_mean = float(np.mean(rhos))
        null_means = np.vstack(draws[key]).mean(axis=0)
        center = float(np.median(null_means))
        pvalue = (1 + np.sum(np.abs(null_means) >= abs(obs_mean))) / (ns.N_PERM + 1)
        summary_rows.append({
            "subset": key,
            "n_sections": len(rhos),
            "interpretable": True,
            "mean_rho": obs_mean,
            "null_center": center,
            "excess": obs_mean - center,
            "perm_p": float(pvalue),
        })
    summary = pd.DataFrame(summary_rows)
    pd.DataFrame(section_rows).to_csv(OUT / "section_residuals.csv", index=False)
    summary.to_csv(OUT / "excess.csv", index=False)
    decision = summary.loc[summary["subset"] == "smooth_primary"].iloc[0]
    if not bool(decision["interpretable"]):
        verdict = {
            "smooth_null": "uninterpretable",
            "reason": "fewer than 8 sections with 100 primary spots",
            "spatial_claim_stops": False,
            "recording_study_opened": False,
        }
    else:
        holds = bool(decision["excess"] >= EXCESS_FLOOR and decision["perm_p"] < P_FLOOR)
        verdict = {
            "smooth_null": "holds" if holds else "spatial_claim_stops",
            "excess": float(decision["excess"]),
            "mean_rho": float(decision["mean_rho"]),
            "null_center": float(decision["null_center"]),
            "perm_p": float(decision["perm_p"]),
            "n_sections": int(decision["n_sections"]),
            "spatial_claim_stops": not holds,
            "mixture_bound": 0.0625,
            "excess_above_perfect_mixture_bound": bool(decision["excess"] > 1.0 / 16.0),
            "recording_study_opened": False,
        }
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(json.dumps(verdict, indent=2), flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
