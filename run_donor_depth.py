# -*- coding: utf-8 -*-
"""Donor-level coefficient with detected genes added.

Written before that coefficient was read. Does not overwrite
results/malignant_cells/verdict.json or tumor_means.csv.

The existing row is cell cycle only. The third covariate is the donor mean
of the natural log of one plus the number of genes detected in each cell,
the same cell-level covariate the within-donor partial uses, averaged as the
cell-cycle score is averaged. Every term is standardized inside the fit.
Ordinary least squares, HC3, the same 81 donors and the same leave-Neftel fit.

Reading, fixed before the coefficient is known. A fit clears the bar only
when its coefficient is at least 0.20, its p-value is below 0.10, and its
robust 95% interval lies entirely above 0.20. The result holds only when
both the 81-donor fit and the leave-Neftel fit clear that bar. Otherwise
the cell-cycle-only row is not the decision. The within-donor partials are
not recomputed.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import run_gates as rg

H5 = Path(r"G:\James Zou Project\operative_state\data_neftel\gbmap_core.h5ad")
TUMORS = Path(r"G:\James Zou Project\operative_state\results\malignant_cells\tumor_means.csv")
OUT = Path(r"G:\James Zou Project\operative_state\results\malignant_cells")
LOCKED = 0.9344452944572189
LOCKED_LEAVE = 0.9104361688220369


def categorical(group):
    categories = [x.decode() if isinstance(x, bytes) else str(x) for x in group["categories"][:]]
    return categories, np.asarray(group["codes"])


def fit(frame: pd.DataFrame) -> dict:
    d = frame[["mes1_f1", "hypoxia_f1", "cell_cycle", "log_detected"]].dropna().copy()
    for col in d.columns:
        d[col] = rg.zscore(d[col])
    adj = smf.ols("mes1_f1 ~ hypoxia_f1 + cell_cycle + log_detected", data=d).fit(cov_type="HC3")
    low, high = (float(x) for x in adj.conf_int().loc["hypoxia_f1"].tolist())
    coef = float(adj.params["hypoxia_f1"])
    pval = float(adj.pvalues["hypoxia_f1"])
    clears = bool(coef >= 0.20 and pval < 0.10 and low > 0.20)
    return {
        "n": int(adj.nobs),
        "coefficient": coef,
        "se": float(adj.bse["hypoxia_f1"]),
        "p": pval,
        "ci_low": low,
        "ci_high": high,
        "depth_coefficient": float(adj.params["log_detected"]),
        "clears_bar": clears,
    }


def main() -> None:
    tumors = pd.read_csv(TUMORS)
    with h5py.File(H5, "r") as handle:
        types, type_codes = categorical(handle["obs"]["cell_type"])
        donors, donor_codes = categorical(handle["obs"]["donor_id"])
        authors, author_codes = categorical(handle["obs"]["author"])
        malignant = np.flatnonzero(type_codes == types.index("malignant cell"))
        indptr = np.asarray(handle["raw"]["X"]["indptr"])
        detected = indptr[malignant + 1] - indptr[malignant]
    depth = pd.DataFrame({
        "donor": [donors[int(donor_codes[i])] for i in malignant],
        "author": [authors[int(author_codes[i])] for i in malignant],
        "log_detected": np.log1p(detected.astype(np.float64)),
    })
    means = depth.groupby(["donor", "author"], as_index=False)["log_detected"].mean()
    merged = tumors.merge(means, on=["donor", "author"], how="left", validate="one_to_one")
    if merged["log_detected"].isna().any():
        raise SystemExit("a donor has no detected-gene mean")
    check = smf.ols(
        "mes1_f1 ~ hypoxia_f1 + cell_cycle",
        data=merged.assign(**{c: rg.zscore(merged[c]) for c in ("mes1_f1", "hypoxia_f1", "cell_cycle")}),
    ).fit(cov_type="HC3")
    reproduced = float(check.params["hypoxia_f1"])
    if abs(reproduced - LOCKED) > 1e-8:
        raise SystemExit(f"cell-cycle-only coefficient reproduced as {reproduced}")
    primary = fit(merged)
    leave = fit(merged.loc[merged["author"] != "Neftel2019"])
    leave_check = smf.ols(
        "mes1_f1 ~ hypoxia_f1 + cell_cycle",
        data=merged.loc[merged["author"] != "Neftel2019"].assign(
            **{c: rg.zscore(merged.loc[merged["author"] != "Neftel2019", c]) for c in ("mes1_f1", "hypoxia_f1", "cell_cycle")}
        ),
    ).fit(cov_type="HC3")
    if abs(float(leave_check.params["hypoxia_f1"]) - LOCKED_LEAVE) > 1e-8:
        raise SystemExit("leave-Neftel cell-cycle-only coefficient was not reproduced")
    holds = bool(primary["clears_bar"] and leave["clears_bar"])
    verdict = {
        "covariate": "donor mean of log1p(genes detected)",
        "original_verdict_overwritten": False,
        "cell_cycle_only_reproduced": reproduced,
        "primary": primary,
        "without_neftel2019": leave,
        "holds": holds,
    }
    (OUT / "depth_adjusted.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    merged.to_csv(OUT / "tumor_means_depth.csv", index=False)
    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
