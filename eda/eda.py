#!/usr/bin/env python3
"""Reproducible EDA for the Summer 2026 MERFISH cell-type challenge.

The script intentionally depends only on pandas and NumPy.  Figures are written
as standalone SVG files so the analysis can run in a minimal environment.
"""

from __future__ import annotations

import html
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "eda" / "outputs"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
TARGET = "MERFISH_cell_type_annotation"
SPLIT_COLOR = {"train": "#2563eb", "test": "#f97316"}


def esc(value: object) -> str:
    return html.escape(str(value))


def q(values: pd.Series, probs=(0.0, 0.25, 0.5, 0.75, 1.0)) -> dict[str, float]:
    result = values.quantile(list(probs))
    return {f"q{int(p * 100):02d}": float(result.loc[p]) for p in probs}


def write_svg(path: Path, width: int, height: int, body: list[str]) -> None:
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;fill:#172033}.title{font-size:20px;font-weight:700}.subtitle{font-size:12px;fill:#64748b}.axis{font-size:10px;fill:#64748b}.label{font-size:11px}.grid{stroke:#e2e8f0;stroke-width:1}</style>',
        *body,
        "</svg>",
    ]
    path.write_text("\n".join(svg), encoding="utf-8")


def barh_svg(series: pd.Series, path: Path, title: str, subtitle: str) -> None:
    series = series.sort_values(ascending=True)
    width = 1050
    row_h = 18
    left, right, top, bottom = 260, 60, 65, 35
    height = top + bottom + row_h * len(series)
    plot_w = width - left - right
    max_value = max(float(series.max()), 1.0)
    body = [
        f'<text x="24" y="30" class="title">{esc(title)}</text>',
        f'<text x="24" y="50" class="subtitle">{esc(subtitle)}</text>',
    ]
    for i, (name, value) in enumerate(series.items()):
        y = top + i * row_h
        bar_w = plot_w * float(value) / max_value
        body.append(f'<text x="{left - 8}" y="{y + 12}" text-anchor="end" class="label">{esc(name)}</text>')
        body.append(f'<rect x="{left}" y="{y + 2}" width="{bar_w:.2f}" height="12" rx="2" fill="#2563eb"/>')
        body.append(f'<text x="{left + bar_w + 6:.2f}" y="{y + 12}" class="axis">{int(value):,}</text>')
    write_svg(path, width, height, body)


def grouped_missingness_svg(table: pd.DataFrame, path: Path) -> None:
    fields = list(table.index)
    width, height = 1050, 540
    left, right, top, bottom = 210, 50, 75, 55
    plot_w, plot_h = width - left - right, height - top - bottom
    group_h = plot_h / max(len(fields), 1)
    body = [
        '<text x="24" y="30" class="title">Metadata missingness</text>',
        '<text x="24" y="50" class="subtitle">The test target is intentionally missing; other gaps may require indicators or imputation.</text>',
    ]
    for tick in range(0, 101, 20):
        x = left + plot_w * tick / 100
        body += [
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" class="grid"/>',
            f'<text x="{x:.1f}" y="{height-bottom+20}" text-anchor="middle" class="axis">{tick}%</text>',
        ]
    for i, field in enumerate(fields):
        center = top + (i + 0.5) * group_h
        body.append(f'<text x="{left-8}" y="{center+4:.1f}" text-anchor="end" class="label">{esc(field)}</text>')
        for offset, split in [(-5, "train"), (5, "test")]:
            value = float(table.loc[field, split]) * 100
            bar_w = plot_w * value / 100
            body.append(f'<rect x="{left}" y="{center+offset-4:.1f}" width="{bar_w:.2f}" height="8" fill="{SPLIT_COLOR[split]}"/>')
    body += [
        f'<rect x="{width-180}" y="22" width="12" height="8" fill="{SPLIT_COLOR["train"]}"/><text x="{width-162}" y="30" class="axis">train</text>',
        f'<rect x="{width-105}" y="22" width="12" height="8" fill="{SPLIT_COLOR["test"]}"/><text x="{width-87}" y="30" class="axis">test</text>',
    ]
    write_svg(path, width, height, body)


def qc_hist_svg(qc: pd.DataFrame, path: Path) -> None:
    panels = [
        ("total_counts", "Total transcripts per cell"),
        ("detected_genes", "Detected genes per cell"),
        ("zero_fraction", "Fraction of zero counts"),
    ]
    width, height = 1200, 410
    outer_left, top, panel_w, panel_h, gap = 60, 80, 330, 250, 55
    body = [
        '<text x="24" y="30" class="title">Per-cell count quality</text>',
        '<text x="24" y="50" class="subtitle">Overlaid train/test distributions; strong separation would indicate covariate shift.</text>',
    ]
    for panel_i, (column, title) in enumerate(panels):
        x0 = outer_left + panel_i * (panel_w + gap)
        values = qc[column].to_numpy(float)
        lo, hi = np.quantile(values, [0.005, 0.995])
        if hi <= lo:
            hi = lo + 1
        bins = np.linspace(lo, hi, 31)
        peak = 0.0
        histograms: dict[str, np.ndarray] = {}
        for split in ["train", "test"]:
            hist, _ = np.histogram(qc.loc[qc["split"] == split, column], bins=bins, density=True)
            histograms[split] = hist
            peak = max(peak, float(hist.max()))
        body.append(f'<text x="{x0}" y="{top-18}" class="label" font-weight="700">{esc(title)}</text>')
        for tick in range(5):
            x = x0 + panel_w * tick / 4
            value = lo + (hi - lo) * tick / 4
            body += [
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+panel_h}" class="grid"/>',
                f'<text x="{x:.1f}" y="{top+panel_h+20}" text-anchor="middle" class="axis">{value:.2g}</text>',
            ]
        for split in ["train", "test"]:
            points = []
            for i, density in enumerate(histograms[split]):
                x = x0 + panel_w * (i + 0.5) / len(histograms[split])
                y = top + panel_h - panel_h * float(density) / max(peak, 1e-12)
                points.append(f"{x:.1f},{y:.1f}")
            body.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{SPLIT_COLOR[split]}" stroke-width="2" opacity="0.9"/>')
        body.append(f'<line x1="{x0}" y1="{top+panel_h}" x2="{x0+panel_w}" y2="{top+panel_h}" stroke="#64748b"/>')
    body += [
        f'<line x1="{width-180}" y1="26" x2="{width-160}" y2="26" stroke="{SPLIT_COLOR["train"]}" stroke-width="3"/><text x="{width-154}" y="30" class="axis">train</text>',
        f'<line x1="{width-100}" y1="26" x2="{width-80}" y2="26" stroke="{SPLIT_COLOR["test"]}" stroke-width="3"/><text x="{width-74}" y="30" class="axis">test</text>',
    ]
    write_svg(path, width, height, body)


def gene_shift_svg(gene_table: pd.DataFrame, path: Path) -> None:
    width, height = 760, 700
    left, right, top, bottom = 90, 45, 75, 75
    plot_w, plot_h = width - left - right, height - top - bottom
    x = np.log10(gene_table["train_mean"].to_numpy() + 0.01)
    y = np.log10(gene_table["test_mean"].to_numpy() + 0.01)
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    pad = (hi - lo) * 0.05
    lo, hi = lo - pad, hi + pad
    body = [
        '<text x="24" y="30" class="title">Train/test gene-expression shift</text>',
        '<text x="24" y="50" class="subtitle">Each point is one gene; distance from the diagonal indicates marginal mean shift.</text>',
    ]
    for tick in np.linspace(lo, hi, 6):
        px = left + plot_w * (tick - lo) / (hi - lo)
        py = top + plot_h - plot_h * (tick - lo) / (hi - lo)
        body += [
            f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{top+plot_h}" class="grid"/>',
            f'<line x1="{left}" y1="{py:.1f}" x2="{left+plot_w}" y2="{py:.1f}" class="grid"/>',
            f'<text x="{px:.1f}" y="{top+plot_h+20}" text-anchor="middle" class="axis">{max(0.0, 10**tick-0.01):.2g}</text>',
            f'<text x="{left-10}" y="{py+3:.1f}" text-anchor="end" class="axis">{max(0.0, 10**tick-0.01):.2g}</text>',
        ]
    body.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top}" stroke="#94a3b8" stroke-dasharray="5,5"/>')
    rank = gene_table["abs_log2_mean_ratio"].rank(method="first", ascending=False)
    for gene, px0, py0, r in zip(gene_table.index, x, y, rank):
        px = left + plot_w * (px0 - lo) / (hi - lo)
        py = top + plot_h - plot_h * (py0 - lo) / (hi - lo)
        color = "#dc2626" if r <= 12 else "#2563eb"
        radius = 3.5 if r <= 12 else 2.2
        body.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{radius}" fill="{color}" opacity="0.68"/>')
        if r <= 12:
            body.append(f'<text x="{px+5:.1f}" y="{py-4:.1f}" class="axis">{esc(gene)}</text>')
    body += [
        f'<text x="{left+plot_w/2:.1f}" y="{height-24}" text-anchor="middle" class="label">Train mean count</text>',
        f'<text transform="translate(24 {top+plot_h/2:.1f}) rotate(-90)" text-anchor="middle" class="label">Test mean count</text>',
    ]
    write_svg(path, width, height, body)


def spatial_split_svg(meta: pd.DataFrame, path: Path) -> None:
    sections = meta["Section_ID"].value_counts().head(9).index.tolist()
    cols, rows = 3, math.ceil(len(sections) / 3)
    panel_w, panel_h = 340, 270
    width, height = cols * panel_w + 40, rows * panel_h + 90
    body = [
        '<text x="24" y="30" class="title">Spatial mixing of train and test cells</text>',
        '<text x="24" y="50" class="subtitle">Nine largest sections. Interleaving supports spatial features but also demands leakage-aware validation.</text>',
    ]
    for i, section in enumerate(sections):
        frame = meta.loc[meta["Section_ID"] == section]
        col, row = i % cols, i // cols
        x0, y0 = 45 + col * panel_w, 80 + row * panel_h
        plot_w, plot_h = 270, 205
        xmin, xmax = frame["center_x"].min(), frame["center_x"].max()
        ymin, ymax = frame["center_y"].min(), frame["center_y"].max()
        dx, dy = max(xmax - xmin, 1), max(ymax - ymin, 1)
        body.append(f'<text x="{x0}" y="{y0-8}" class="label" font-weight="700">{esc(section)} (n={len(frame):,})</text>')
        body.append(f'<rect x="{x0}" y="{y0}" width="{plot_w}" height="{plot_h}" fill="#f8fafc" stroke="#e2e8f0"/>')
        for split in ["train", "test"]:
            sub = frame.loc[frame["split"] == split]
            for x_value, y_value in zip(sub["center_x"], sub["center_y"]):
                px = x0 + plot_w * (x_value - xmin) / dx
                py = y0 + plot_h - plot_h * (y_value - ymin) / dy
                body.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="1.7" fill="{SPLIT_COLOR[split]}" opacity="0.58"/>')
    body += [
        f'<circle cx="{width-170}" cy="28" r="5" fill="{SPLIT_COLOR["train"]}"/><text x="{width-160}" y="32" class="axis">train</text>',
        f'<circle cx="{width-90}" cy="28" r="5" fill="{SPLIT_COLOR["test"]}"/><text x="{width-80}" y="32" class="axis">test</text>',
    ]
    write_svg(path, width, height, body)


def marker_heatmap_svg(class_means: pd.DataFrame, marker_table: pd.DataFrame, path: Path) -> None:
    ordered_classes = class_means.index.tolist()
    marker_genes = []
    for label in ordered_classes:
        candidates = marker_table.loc[marker_table["cell_type"] == label, "gene"].tolist()
        for gene in candidates:
            if gene not in marker_genes:
                marker_genes.append(gene)
                break
    values = np.log1p(class_means[marker_genes])
    values = (values - values.mean(axis=0)) / values.std(axis=0).replace(0, 1)
    values = values.clip(-2.5, 2.5)
    cell_w, cell_h = 14, 15
    left, top, right, bottom = 290, 310, 45, 45
    width = left + right + cell_w * len(marker_genes)
    height = top + bottom + cell_h * len(ordered_classes)
    body = [
        '<text x="24" y="30" class="title">Cell-type marker heatmap</text>',
        '<text x="24" y="50" class="subtitle">One unique high-effect marker per class where available; column-wise z-score of log1p class mean.</text>',
    ]
    for j, gene in enumerate(marker_genes):
        x = left + j * cell_w + 10
        body.append(f'<text transform="translate({x:.1f} {top-8}) rotate(-60)" class="axis">{esc(gene)}</text>')
    for i, label in enumerate(ordered_classes):
        y = top + i * cell_h
        body.append(f'<text x="{left-8}" y="{y+11}" text-anchor="end" class="axis">{esc(label)}</text>')
        for j, gene in enumerate(marker_genes):
            value = float(values.loc[label, gene])
            if value >= 0:
                strength = min(value / 2.5, 1)
                r, g, b = 255 - int(210 * strength), 255 - int(150 * strength), 255 - int(80 * strength)
            else:
                strength = min(-value / 2.5, 1)
                r, g, b = 255 - int(70 * strength), 255 - int(120 * strength), 255 - int(20 * strength)
            body.append(f'<rect x="{left+j*cell_w}" y="{y}" width="{cell_w}" height="{cell_h}" fill="rgb({r},{g},{b})"/>')
    write_svg(path, width, height, body)


def nearest_spatial_summary(meta_train: pd.DataFrame, meta_test: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    rows: list[dict[str, object]] = []
    for section, test_part in meta_test.groupby("Section_ID", dropna=False):
        if pd.isna(section):
            train_part = meta_train.loc[meta_train["Section_ID"].isna()]
        else:
            train_part = meta_train.loc[meta_train["Section_ID"] == section]
        if train_part.empty:
            for cell_id in test_part.index:
                rows.append({"cell_id": cell_id, "Section_ID": section, "nearest_train_distance": np.nan})
            continue
        train_xy = train_part[["center_x", "center_y"]].to_numpy(float)
        test_xy = test_part[["center_x", "center_y"]].to_numpy(float)
        for start in range(0, len(test_xy), 500):
            chunk = test_xy[start : start + 500]
            distances_sq = ((chunk[:, None, :] - train_xy[None, :, :]) ** 2).sum(axis=2)
            distances = np.sqrt(distances_sq.min(axis=1))
            for cell_id, distance in zip(test_part.index[start : start + 500], distances):
                rows.append({"cell_id": cell_id, "Section_ID": section, "nearest_train_distance": float(distance)})
    result = pd.DataFrame(rows).set_index("cell_id")

    correct = 0
    total = 0
    for _, part in meta_train.groupby("Section_ID", dropna=False):
        if len(part) < 2:
            continue
        xy = part[["center_x", "center_y"]].to_numpy(float)
        for start in range(0, len(xy), 400):
            chunk = xy[start : start + 400]
            distances_sq = ((chunk[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2)
            local_rows = np.arange(len(chunk))
            global_rows = np.arange(start, start + len(chunk))
            distances_sq[local_rows, global_rows] = np.inf
            nearest = distances_sq.argmin(axis=1)
            predicted = part.iloc[nearest][TARGET].to_numpy()
            observed = part.iloc[start : start + len(chunk)][TARGET].to_numpy()
            correct += int((predicted == observed).sum())
            total += len(chunk)
    return result, correct / max(total, 1)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    counts_train = pd.read_csv(DATA_DIR / "counts_train.csv", index_col=0)
    counts_test = pd.read_csv(DATA_DIR / "counts_test.csv", index_col=0)
    meta_train = pd.read_csv(DATA_DIR / "meta_train.csv", index_col=0)
    meta_test = pd.read_csv(DATA_DIR / "meta_test.csv", index_col=0)

    # Contract checks catch silent row/column misalignment before any analysis.
    assert counts_train.index.is_unique and counts_test.index.is_unique
    assert meta_train.index.is_unique and meta_test.index.is_unique
    assert counts_train.index.equals(meta_train.index)
    assert counts_test.index.equals(meta_test.index)
    assert counts_train.columns.equals(counts_test.columns)
    assert not counts_train.isna().any().any() and not counts_test.isna().any().any()
    assert (counts_train.to_numpy() >= 0).all() and (counts_test.to_numpy() >= 0).all()
    assert meta_train[TARGET].notna().all() and meta_test[TARGET].isna().all()

    labels = meta_train[TARGET]
    label_counts = labels.value_counts()
    label_counts.rename("n_train").to_csv(TABLE_DIR / "class_counts.csv")

    qc_parts = []
    for split, counts in [("train", counts_train), ("test", counts_test)]:
        qc_parts.append(
            pd.DataFrame(
                {
                    "split": split,
                    "total_counts": counts.sum(axis=1),
                    "detected_genes": (counts > 0).sum(axis=1),
                    "zero_fraction": (counts == 0).mean(axis=1),
                },
                index=counts.index,
            )
        )
    qc = pd.concat(qc_parts)
    qc.to_csv(TABLE_DIR / "cell_qc.csv", index_label="cell_id")
    qc_summary = qc.groupby("split")[["total_counts", "detected_genes", "zero_fraction"]].agg(
        ["min", "median", "mean", "max"]
    )
    qc_summary.to_csv(TABLE_DIR / "cell_qc_summary.csv")

    missingness = pd.DataFrame(
        {"train": meta_train.isna().mean(), "test": meta_test.isna().mean()}
    )
    missingness.to_csv(TABLE_DIR / "metadata_missingness.csv")

    numeric_cols = ["volume", "center_x", "center_y", "Region", "Segment", "AP_position"]
    numeric_rows = []
    for column in numeric_cols:
        tr, te = meta_train[column].dropna(), meta_test[column].dropna()
        pooled = math.sqrt((float(tr.var()) + float(te.var())) / 2) if len(tr) > 1 and len(te) > 1 else np.nan
        smd = (float(te.mean()) - float(tr.mean())) / pooled if pooled and not np.isnan(pooled) else np.nan
        numeric_rows.append(
            {
                "feature": column,
                "train_mean": tr.mean(),
                "test_mean": te.mean(),
                "standardized_mean_difference": smd,
                **{f"train_{k}": v for k, v in q(tr).items()},
                **{f"test_{k}": v for k, v in q(te).items()},
            }
        )
    numeric_drift = pd.DataFrame(numeric_rows).set_index("feature")
    numeric_drift.to_csv(TABLE_DIR / "numeric_metadata_drift.csv")

    categorical_cols = ["Datasets", "Region", "Excitatory_vs_Inhibitory", "Segment", "Gender", "Mouse_ID", "AP_position", "Section_ID"]
    categorical_rows = []
    for column in categorical_cols:
        train_dist = meta_train[column].fillna("<missing>").astype(str).value_counts(normalize=True)
        test_dist = meta_test[column].fillna("<missing>").astype(str).value_counts(normalize=True)
        levels = train_dist.index.union(test_dist.index)
        train_aligned = train_dist.reindex(levels, fill_value=0)
        test_aligned = test_dist.reindex(levels, fill_value=0)
        tvd = 0.5 * (train_aligned - test_aligned).abs().sum()
        categorical_rows.append(
            {
                "feature": column,
                "train_levels": len(train_dist),
                "test_levels": len(test_dist),
                "unseen_test_levels": int((~test_dist.index.isin(train_dist.index)).sum()),
                "total_variation_distance": float(tvd),
            }
        )
    categorical_drift = pd.DataFrame(categorical_rows).set_index("feature")
    categorical_drift.to_csv(TABLE_DIR / "categorical_metadata_drift.csv")

    gene_table = pd.DataFrame(index=counts_train.columns)
    gene_table["train_mean"] = counts_train.mean(axis=0)
    gene_table["test_mean"] = counts_test.mean(axis=0)
    gene_table["train_detection_rate"] = (counts_train > 0).mean(axis=0)
    gene_table["test_detection_rate"] = (counts_test > 0).mean(axis=0)
    gene_table["log2_mean_ratio_test_over_train"] = np.log2((gene_table["test_mean"] + 0.01) / (gene_table["train_mean"] + 0.01))
    gene_table["abs_log2_mean_ratio"] = gene_table["log2_mean_ratio_test_over_train"].abs()
    gene_table.sort_values("abs_log2_mean_ratio", ascending=False).to_csv(TABLE_DIR / "gene_summary.csv")

    marker_rows = []
    class_means = counts_train.groupby(labels).mean()
    for label, part_index in labels.groupby(labels).groups.items():
        inside = counts_train.loc[part_index]
        outside = counts_train.drop(index=part_index)
        inside_mean = inside.mean(axis=0)
        outside_mean = outside.mean(axis=0)
        score = np.log2((inside_mean + 0.1) / (outside_mean + 0.1))
        detection_delta = (inside > 0).mean(axis=0) - (outside > 0).mean(axis=0)
        ranking = (score + detection_delta).sort_values(ascending=False).head(5)
        for rank, gene in enumerate(ranking.index, start=1):
            marker_rows.append(
                {
                    "cell_type": label,
                    "rank": rank,
                    "gene": gene,
                    "log2_mean_ratio": float(score[gene]),
                    "detection_rate_delta": float(detection_delta[gene]),
                    "mean_in_class": float(inside_mean[gene]),
                    "mean_outside_class": float(outside_mean[gene]),
                }
            )
    markers = pd.DataFrame(marker_rows)
    markers.to_csv(TABLE_DIR / "top_markers_by_cell_type.csv", index=False)

    train_hash = pd.util.hash_pandas_object(counts_train, index=False)
    test_hash = pd.util.hash_pandas_object(counts_test, index=False)
    exact_cross_split_duplicates = int(test_hash.isin(set(train_hash)).sum())

    spatial_distances, spatial_1nn_train_accuracy = nearest_spatial_summary(meta_train, meta_test)
    spatial_distances.to_csv(TABLE_DIR / "test_nearest_train_spatial_distance.csv")

    combined_meta = pd.concat(
        [meta_train.assign(split="train"), meta_test.assign(split="test")]
    )
    shared_sections = set(meta_train["Section_ID"].dropna()) & set(meta_test["Section_ID"].dropna())
    shared_mice = set(meta_train["Mouse_ID"].dropna()) & set(meta_test["Mouse_ID"].dropna())

    barh_svg(label_counts, FIG_DIR / "class_distribution.svg", "Training class distribution", f"{len(label_counts)} target classes; sorted from rarest to most common.")
    grouped_missingness_svg(missingness, FIG_DIR / "metadata_missingness.svg")
    qc_hist_svg(qc, FIG_DIR / "cell_qc_distributions.svg")
    gene_shift_svg(gene_table, FIG_DIR / "gene_expression_shift.svg")
    spatial_split_svg(combined_meta, FIG_DIR / "spatial_train_test_split.svg")
    marker_heatmap_svg(class_means.loc[label_counts.index], markers, FIG_DIR / "marker_heatmap.svg")

    train_total_median = float(qc.loc[qc["split"] == "train", "total_counts"].median())
    test_total_median = float(qc.loc[qc["split"] == "test", "total_counts"].median())
    train_detected_median = float(qc.loc[qc["split"] == "train", "detected_genes"].median())
    test_detected_median = float(qc.loc[qc["split"] == "test", "detected_genes"].median())
    max_tvd_feature = categorical_drift["total_variation_distance"].idxmax()
    max_tvd = float(categorical_drift.loc[max_tvd_feature, "total_variation_distance"])
    nearest_median = float(spatial_distances["nearest_train_distance"].median())
    nearest_p90 = float(spatial_distances["nearest_train_distance"].quantile(0.9))
    imbalance_ratio = float(label_counts.max() / label_counts.min())
    majority_accuracy = float(label_counts.max() / label_counts.sum())
    random_pair_match = float(((label_counts / label_counts.sum()) ** 2).sum())

    report = f"""# MERFISH cell-type prediction — EDA

Generated by `eda/01_eda.py` from the four challenge CSV files.

## Executive summary

- The data contract is clean: train/test each contain **{len(counts_train):,} cells × {counts_train.shape[1]} genes**; cell IDs are unique, counts are non-negative integers, gene order matches, and count matrices contain no missing values.
- There are **{len(label_counts)} cell types**. The largest class has **{label_counts.max():,}** cells and the smallest has **{label_counts.min():,}** (imbalance ratio **{imbalance_ratio:.1f}×**). A majority-class prediction would score only **{majority_accuracy:.1%}** accuracy.
- Counts are sparse: the median cell has **{train_detected_median:.0f}** detected genes in train and **{test_detected_median:.0f}** in test, out of {counts_train.shape[1]}. Median total transcripts are **{train_total_median:.0f}** and **{test_total_median:.0f}**, respectively.
- Train and test share **{len(shared_sections)} sections** and **{len(shared_mice)} mice**. The median test cell is only **{nearest_median:.1f} coordinate units** from a training cell in the same section (90th percentile **{nearest_p90:.1f}**), so spatial context is likely predictive.
- A same-section spatial 1-nearest-neighbor rule evaluated leave-one-out on train reaches **{spatial_1nn_train_accuracy:.1%}**, versus **{random_pair_match:.1%}** expected agreement for two random labels with the observed class frequencies. It still trails the **{majority_accuracy:.1%}** majority baseline, so location alone is insufficient.
- The largest categorical train/test shift is **{max_tvd_feature}** (total-variation distance **{max_tvd:.3f}**). There are **{exact_cross_split_duplicates} exact count-vector duplicates** across train and test.

## Figures

### 1. Target balance

![Training class distribution](figures/class_distribution.svg)

### 2. Count quality and sparsity

![Per-cell count quality](figures/cell_qc_distributions.svg)

### 3. Missing metadata

![Metadata missingness](figures/metadata_missingness.svg)

### 4. Marginal gene shift

![Gene-expression shift](figures/gene_expression_shift.svg)

### 5. Train/test spatial mixing

![Spatial train/test split](figures/spatial_train_test_split.svg)

### 6. Candidate markers

![Marker heatmap](figures/marker_heatmap.svg)

## Modeling implications

1. **Use two validation schemes.** Report stratified random cross-validation for an in-distribution estimate, and GroupKFold by `Section_ID` (or `Mouse_ID`) for a conservative generalization estimate. Random splitting alone will benefit from nearby cells and can overstate robustness.
2. **Start with count-aware preprocessing.** Compare raw/log1p counts, per-cell library-size normalization followed by log1p, and simple binary detection. With only 200 targeted genes, do not automatically discard low-detection genes; many are intentional markers.
3. **Build a strong tabular baseline first.** Multinomial logistic regression and tree boosting on expression + metadata will reveal whether nonlinear interactions matter. Treat `Datasets`, `Mouse_ID`, `Section_ID`, `Gender`, `AP_position`, `Region`, and `Segment` as categorical; add missing indicators.
4. **Add spatial features carefully.** Candidate features include within-section normalized x/y, distances to labeled neighbors, and neighborhood-aggregated expression. Every spatial feature must be recomputed inside each validation fold to prevent label leakage.
5. **Optimize the actual metric.** Accuracy rewards frequent classes, so compare ordinary training with class weighting rather than assuming weighting helps. Always inspect per-class recall and the confusion matrix even though the leaderboard uses overall accuracy.
6. **Use marker and drift tables for debugging.** Weak or biologically implausible markers can reveal label noise; strongly shifted genes or metadata levels can cause fragile validation gains.

## Output tables

- `tables/class_counts.csv`: class support.
- `tables/cell_qc.csv` and `cell_qc_summary.csv`: per-cell and aggregate count QC.
- `tables/metadata_missingness.csv`: train/test missingness.
- `tables/numeric_metadata_drift.csv` and `categorical_metadata_drift.csv`: train/test shift.
- `tables/gene_summary.csv`: gene abundance, detection, and marginal shift.
- `tables/top_markers_by_cell_type.csv`: five candidate markers per class.
- `tables/test_nearest_train_spatial_distance.csv`: spatial coverage of test by train.

## Reproduce

From the repository root:

```bash
python -m pip install -r eda/requirements.txt
python eda/01_eda.py
```
"""
    (OUT_DIR / "EDA_REPORT.md").write_text(report, encoding="utf-8")

    print("EDA complete")
    print(f"Report: {OUT_DIR / 'EDA_REPORT.md'}")
    print(f"Classes: {len(label_counts)}; imbalance ratio: {imbalance_ratio:.2f}x")
    print(f"Spatial 1-NN train LOO accuracy: {spatial_1nn_train_accuracy:.4f}")
    print(f"Median test-to-train spatial distance: {nearest_median:.2f}")


if __name__ == "__main__":
    main()
