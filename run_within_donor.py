"""Within-donor partial correlation. Rules are in FIELD_PROTOCOL.md.

Cells are not pooled. The decision is the unweighted mean of partial Spearmans.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

CELLS = Path(r"G:\James Zou Project\operative_state\results\malignant_cells\cell_scores.csv")
OUT = Path(r"G:\James Zou Project\operative_state\results\within_donor")
OUT.mkdir(parents=True, exist_ok=True)

MIN_CELLS = 20
MIN_DONORS = 8
FLOOR = 0.20
DENOM_FLOOR = 1e-8


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.unique(a).size < 2 or np.unique(b).size < 2:
        return np.nan
    rho = spearmanr(a, b).statistic
    return float(rho)


def partial_spearman(hypoxia: np.ndarray, mes: np.ndarray, cycle: np.ndarray) -> tuple[float, float]:
    raw = safe_spearman(hypoxia, mes)
    r_hc = safe_spearman(hypoxia, cycle)
    r_mc = safe_spearman(mes, cycle)
    if not np.isfinite(raw) or not np.isfinite(r_hc) or not np.isfinite(r_mc):
        return raw, np.nan
    denom = np.sqrt((1.0 - r_hc * r_hc) * (1.0 - r_mc * r_mc))
    if not np.isfinite(denom) or denom < DENOM_FLOOR:
        return raw, np.nan
    return raw, float((raw - r_hc * r_mc) / denom)


def summarize(frame: pd.DataFrame) -> dict:
    values = frame["partial"].to_numpy()
    n = int(np.isfinite(values).sum())
    if n < MIN_DONORS:
        return {"n_donors": n, "interpretable": False, "mean_partial": None}
    used = values[np.isfinite(values)]
    sd = float(np.std(used, ddof=1)) if n > 1 else np.nan
    return {
        "n_donors": n,
        "interpretable": True,
        "mean_partial": float(np.mean(used)),
        "se_mean": float(sd / np.sqrt(n)) if np.isfinite(sd) else None,
        "median_partial": float(np.median(used)),
        "fraction_at_least_0.20": float(np.mean(used >= FLOOR)),
        "mean_raw_spearman": float(frame.loc[np.isfinite(values), "raw_spearman"].mean()),
    }


def main() -> None:
    cells = pd.read_csv(CELLS, usecols=["hypoxia_f1", "mes1_f1", "cell_cycle", "donor", "author"])
    rows = []
    for (donor, author), sub in cells.groupby(["donor", "author"], sort=False):
        sub = sub.dropna()
        if len(sub) < MIN_CELLS:
            continue
        raw, partial = partial_spearman(
            sub["hypoxia_f1"].to_numpy(),
            sub["mes1_f1"].to_numpy(),
            sub["cell_cycle"].to_numpy(),
        )
        rows.append({
            "donor": donor,
            "author": author,
            "n_cells": int(len(sub)),
            "raw_spearman": raw,
            "partial": partial,
        })
    donors = pd.DataFrame(rows)
    donors.to_csv(OUT / "donor_correlations.csv", index=False)
    entered = donors.dropna(subset=["partial"])
    all_donors = summarize(entered)
    without = summarize(entered.loc[entered["author"] != "Neftel2019"])
    if not all_donors["interpretable"] or not without["interpretable"]:
        verdict = {
            "within_donor": "uninterpretable",
            "all_donors": all_donors,
            "without_neftel2019": without,
            "cell_intrinsic_upgrade": False,
            "recording_study_opened": False,
            "further_analyses_added": False,
        }
    else:
        all_mean = all_donors["mean_partial"]
        leave_mean = without["mean_partial"]
        if all_mean < FLOOR:
            status = "two_malignant_states"
            upgrade = False
        elif leave_mean < FLOOR:
            status = "upgrade_withdrawn"
            upgrade = False
        else:
            status = "holds"
            upgrade = True
        verdict = {
            "within_donor": status,
            "mean_partial": all_mean,
            "mean_partial_without_neftel": leave_mean,
            "all_donors": all_donors,
            "without_neftel2019": without,
            "cell_intrinsic_upgrade": upgrade,
            "sentence_changes_to_two_malignant_states": status == "two_malignant_states",
            "recording_study_opened": False,
            "further_analyses_added": False,
            "hallmark_census_survivors_are_claims": False,
        }
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
