"""Spot-level niche test. Rules are in NICHE_PROTOCOL.md. Labels are written before any correlation."""

from __future__ import annotations

import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(r"G:\James Zou Project\operative_state")))
import run_full_list as fl  # noqa: E402
import run_gates as rg  # noqa: E402
import run_null_standard as ns  # noqa: E402

OUT = Path(r"G:\James Zou Project\operative_state\results\niche_visium")
IMG = Path(r"G:\James Zou Project\operative_state\data_visium_hires")
OUT.mkdir(parents=True, exist_ok=True)

H_VEC = np.array([0.65, 0.70, 0.29], dtype=np.float64)
E_VEC = np.array([0.07, 0.99, 0.11], dtype=np.float64)
H_VEC /= np.linalg.norm(H_VEC)
E_VEC /= np.linalg.norm(E_VEC)
EXCESS_FLOOR = 0.10
P_FLOOR = 0.10
MIN_SECTIONS = 8
MIN_SPOTS = 100


def sections() -> list[Path]:
    folders = []
    for folder in sorted(ns.VISIUM.iterdir()):
        if not folder.is_dir():
            continue
        name = folder.name
        if "_T_ST" in name and "_TI_" not in name and "_TC_" not in name and "_C_ST" not in name:
            folders.append(folder)
    return folders


def stain_image(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    optical = -np.log10((rgb + 1.0) / 256.0)
    return optical @ H_VEC, optical @ E_VEC


def positions(folder: Path) -> pd.DataFrame:
    table = pd.read_csv(
        folder / "outs" / "spatial" / "tissue_positions_list.csv",
        header=None,
        names=["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"],
    )
    return table


def measure_spots(folder: Path) -> pd.DataFrame:
    sample = folder.name.lstrip("#")
    scale_path = folder / "outs" / "spatial" / "scalefactors_json.json"
    scale = json.loads(scale_path.read_text(encoding="utf-8"))
    factor = float(scale["tissue_hires_scalef"])
    radius = float(scale["spot_diameter_fullres"]) * factor / 2.0
    hematoxylin, eosin = stain_image(IMG / sample / "tissue_hires_image.png")
    height, width = hematoxylin.shape
    spots = positions(folder)
    h_values = np.full(len(spots), np.nan)
    e_values = np.full(len(spots), np.nan)
    n_pixels = np.zeros(len(spots), dtype=np.int32)
    radius2 = radius * radius
    yy_full = None
    for i, row in enumerate(spots.itertuples(index=False)):
        if int(row.in_tissue) != 1:
            continue
        y = float(row.pxl_row) * factor
        x = float(row.pxl_col) * factor
        y0 = max(0, int(np.floor(y - radius)))
        y1 = min(height, int(np.ceil(y + radius)) + 1)
        x0 = max(0, int(np.floor(x - radius)))
        x1 = min(width, int(np.ceil(x + radius)) + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (yy - y) ** 2 + (xx - x) ** 2 <= radius2
        count = int(mask.sum())
        n_pixels[i] = count
        if count < 10:
            continue
        h_values[i] = float(hematoxylin[y0:y1, x0:x1][mask].mean())
        e_values[i] = float(eosin[y0:y1, x0:x1][mask].mean())
    spots = spots.copy()
    spots["H"] = h_values
    spots["E"] = e_values
    spots["n_pixels"] = n_pixels
    spots["sample"] = sample
    return spots


def label_spots(spots: pd.DataFrame) -> pd.DataFrame:
    spots = spots.copy()
    spots["role"] = "unlabeled"
    stained = spots["H"].notna() & (spots["in_tissue"] == 1)
    if int(stained.sum()) < 50:
        return spots
    total = spots.loc[stained, "H"] + spots.loc[stained, "E"]
    background_cut = float(np.quantile(total, 0.05))
    background = stained & ((spots["H"] + spots["E"]) <= background_cut)
    spots.loc[background, "role"] = "background"
    remaining = stained & ~background
    h_cut = float(np.quantile(spots.loc[remaining, "H"], 0.20))
    e_cut = float(np.median(spots.loc[remaining, "E"]))
    # Addendum in NICHE_PROTOCOL.md: absolute eosin is collinear with hematoxylin
    # on these slides, so the eosin gate is not applied.
    candidate = remaining & (spots["H"] <= h_cut)
    return _label_positional(spots, stained, background, candidate, h_cut, e_cut)


def _label_positional(spots, stained, background, candidate, h_cut, e_cut) -> pd.DataFrame:
    spots = spots.copy()
    n = len(spots)
    in_graph = (spots["in_tissue"].to_numpy() == 1) & np.isfinite(spots["H"].to_numpy())
    rows = spots["array_row"].to_numpy()
    cols = spots["array_col"].to_numpy()
    coord_to_i = {(int(rows[i]), int(cols[i])): i for i in range(n) if in_graph[i]}
    cand = candidate.to_numpy()
    parent = np.arange(n)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    offsets = [(dr, dc) for dr in range(-2, 3) for dc in range(-2, 3) if dr * dr + dc * dc <= 4 and (dr, dc) != (0, 0)]
    for i in range(n):
        if not cand[i]:
            continue
        r, c = int(rows[i]), int(cols[i])
        for dr, dc in offsets:
            j = coord_to_i.get((r + dr, c + dc))
            if j is not None and cand[j] and j > i:
                union(i, j)
    components: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        if cand[i]:
            components[find(i)].append(i)
    necrosis = np.zeros(n, dtype=bool)
    for members in components.values():
        if len(members) >= 30:
            necrosis[members] = True
    # Distance on the in-tissue stained graph.
    neighbors: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        if not in_graph[i]:
            continue
        r, c = int(rows[i]), int(cols[i])
        for dr, dc in offsets:
            j = coord_to_i.get((r + dr, c + dc))
            if j is not None and j > i:
                neighbors[i].append(j)
                neighbors[j].append(i)
    distance = np.full(n, -1, dtype=np.int16)
    queue = deque()
    for i in np.flatnonzero(necrosis):
        distance[i] = 0
        queue.append(i)
    while queue:
        i = queue.popleft()
        if distance[i] == 2:
            continue
        for j in neighbors[i]:
            if distance[j] == -1:
                distance[j] = distance[i] + 1
                queue.append(j)
    role = np.array(["unlabeled"] * n, dtype=object)
    role[stained.to_numpy() & background.to_numpy()] = "background"
    remaining = stained.to_numpy() & ~background.to_numpy()
    role[remaining & necrosis] = "necrosis"
    role[remaining & ~necrosis & (distance >= 1) & (distance <= 2)] = "rim"
    role[remaining & ~necrosis & ~((distance >= 1) & (distance <= 2))] = "primary"
    spots["role"] = role
    spots["h_cut"] = h_cut
    spots["e_cut"] = e_cut
    return spots


def cohort_excess(observed: dict[str, list[float]], draws: dict[str, list[np.ndarray]]) -> pd.DataFrame:
    rows = []
    for key, rhos in observed.items():
        if not rhos:
            continue
        obs_mean = float(np.mean(rhos))
        null_means = np.nanmean(np.vstack(draws[key]), axis=0)
        center = float(np.median(null_means))
        pvalue = (1 + np.sum(np.abs(null_means) >= abs(obs_mean))) / (ns.N_PERM + 1)
        rows.append({
            "subset": key,
            "n_sections": len(rhos),
            "mean_rho": obs_mean,
            "null_center": center,
            "excess": obs_mean - center,
            "perm_p": float(pvalue),
        })
    return pd.DataFrame(rows)


def main() -> None:
    label_frames = []
    for folder in sections():
        measured = measure_spots(folder)
        labeled = label_spots(measured)
        counts = labeled["role"].value_counts().to_dict()
        in_tissue = int((labeled["in_tissue"] == 1).sum())
        stained = int(labeled["H"].notna().sum())
        print(folder.name, "in_tissue", in_tissue, "stained", stained, counts, flush=True)
        if in_tissue and stained / in_tissue < 0.5:
            raise SystemExit(f"Coordinate failure in {folder.name}: most in-tissue spots have no stain disk")
        label_frames.append(labeled)
    labels = pd.concat(label_frames, ignore_index=True)
    labels.to_csv(OUT / "spot_labels.csv", index=False)
    print("labels written", len(labels), flush=True)

    gene_col = pd.read_csv(
        Path(r"E:\Glycocalyx Immune Escape\data\raw\cgga\mRNAseq_325\CGGA.mRNAseq_325.RSEM-genes.20200506.txt"),
        sep="\t", usecols=[0],
    ).iloc[:, 0].astype(str)
    previous = fl.OUT
    fl.OUT = OUT
    fl.build_sets(set(gene_col))
    fl.OUT = previous

    rng = np.random.default_rng(ns.SEED)
    observed = {"primary": [], "sensitivity": [], "rim": [], "necrosis": []}
    draws = {key: [] for key in observed}
    section_rows = []
    for folder in sections():
        sample = folder.name.lstrip("#")
        h5_path = folder / "outs" / "filtered_feature_bc_matrix.h5"
        counts, names = ns.read_visium_counts(h5_path)
        with h5py.File(h5_path, "r") as handle:
            barcodes = [b.decode() if isinstance(b, bytes) else str(b) for b in handle["matrix"]["barcodes"][:]]
        # Duplicate symbols were summed inside read_visium_counts, which can reorder names
        # but not barcodes: barcodes stay in the original spot order, and summing genes
        # does not change column order.
        scores = fl.score_spots(counts, names)
        del counts
        scores.insert(0, "barcode", barcodes)
        lab = labels.loc[labels["sample"] == sample, ["barcode", "role", "H", "E"]]
        merged = scores.merge(lab, on="barcode", how="inner")
        non_background = merged["role"].isin(["primary", "rim", "necrosis"])
        h_median = float(merged.loc[non_background, "H"].median()) if non_background.any() else np.nan
        subsets = {
            "primary": merged["role"] == "primary",
            "sensitivity": (merged["role"] == "primary") & (merged["H"] >= h_median),
            "rim": merged["role"] == "rim",
            "necrosis": merged["role"] == "necrosis",
        }
        for key, mask in subsets.items():
            part = merged.loc[mask]
            ok = (
                np.isfinite(part["hypoxia_f1"])
                & np.isfinite(part["mes1_f1"])
                & np.isfinite(part["neuronal_composition"])
                & np.isfinite(part["myeloid_composition"])
            )
            part = part.loc[ok]
            if len(part) < MIN_SPOTS:
                section_rows.append({"sample": sample, "subset": key, "n_spots": int(len(part)), "rho": np.nan})
                continue
            rho, null = ns.stratified_draws(
                part["hypoxia_f1"].to_numpy(),
                part["mes1_f1"].to_numpy(),
                part["neuronal_composition"].to_numpy(),
                part["myeloid_composition"].to_numpy(),
                rng,
            )
            section_rows.append({
                "sample": sample, "subset": key, "n_spots": int(len(part)),
                "rho": rho, "null_median": float(np.nanmedian(null)),
            })
            if np.isfinite(rho) and np.isfinite(null).all():
                observed[key].append(rho)
                draws[key].append(null)
        print(sample, "permuted", flush=True)
    pd.DataFrame(section_rows).to_csv(OUT / "section_permutation.csv", index=False)
    summary = cohort_excess(observed, draws)
    summary.to_csv(OUT / "excess.csv", index=False)
    primary = summary[summary["subset"] == "primary"]
    if primary.empty or int(primary.iloc[0]["n_sections"]) < MIN_SECTIONS:
        verdict = {
            "within_niche": "uninterpretable",
            "reason": "fewer than 8 sections with 100 primary spots",
            "n_sections": 0 if primary.empty else int(primary.iloc[0]["n_sections"]),
            "recording_study_opened": False,
        }
    else:
        row = primary.iloc[0]
        holds = bool(row["excess"] >= EXCESS_FLOOR and row["perm_p"] < P_FLOOR)
        verdict = {
            "within_niche": "holds" if holds else "niche_cooccupancy",
            "excess": float(row["excess"]),
            "mean_rho": float(row["mean_rho"]),
            "null_center": float(row["null_center"]),
            "perm_p": float(row["perm_p"]),
            "n_sections": int(row["n_sections"]),
            "recording_study_opened": False,
        }
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(json.dumps(verdict, indent=2), flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
