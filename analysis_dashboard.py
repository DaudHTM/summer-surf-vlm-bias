import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


MISSING_STRINGS = {"", "unsure", "unknown", "n/a", "none"}
GT_GENDER_ORDER = ["male", "female", "ambiguous"]
GT_AGE_ORDER = ["young", "middle", "old"]
GT_GEO_ORDER = [
    "north_america",
    "latin_america",
    "europe",
    "sub_saharan_africa",
    "north_africa_middle_east",
    "middle_eastern",
    "south_asia",
    "east_asia",
    "southeast_asia",
    "oceania",
    "ambiguous_generic",
]

PANEL_VALUE_KEY = [
    "#1, #2, #3 confusion heatmaps: row-wise % share (for each true class row, cells show where predictions went).",
    "#5, #6 heatmaps: percent right (accuracy %, higher is better).",
    "#7 heatmap: percent right by confidence and gt_skin_tone.",
    "#9 diverging bar: occupation share difference (female% - male%) in percentage points.",
    "#11 bar: mean scene_socioeconomic_signal by gt_skin_tone with zoomed y-axis and scale key.",
    "#17 line: boundary artifact rate in % (true=artifact).",
    "#18 line: mean lighting_consistency (1-5).",
    "#19 bar: mean person_setting_fit by gt_skin_tone with zoomed y-axis and scale key.",
    "#23-26 heatmap: percent changed for categorical attributes (flip %); skin-tone cell is mean absolute change (not %).",
    "#29-30 heatmap: boundary is percent with artifacts (error-like rate); lighting is mean score (1-5).",
    "#31 heatmap: percent changed (flip %) or mean absolute change vs original.",
    "#32-33 heatmap: mean skin-tone absolute error and mean socioeconomic signal (both are means, not %).",
    "#51-53 heatmap: mean per-person stability metrics by gt_skin_tone.",
    "  - perceived_skin_tone_stability: variance (0+; lower is more stable).",
    "  - categorical stability: entropy in bits (0+; lower is more stable).",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate cross-image annotations and ground truth into a single bias/stability dashboard."
        )
    )
    parser.add_argument("--images-root", type=Path, default=Path("places_inpainting_results"))
    parser.add_argument("--annotations-file", default="cross_image_annotations.json")
    parser.add_argument("--ground-truth-file", default="ground_truth.json")
    parser.add_argument("--output-figure", type=Path, default=Path("analysis_dashboard.png"))
    parser.add_argument("--output-json", type=Path, default=Path("analysis_summary.json"))
    parser.add_argument(
        "--output-stats-json",
        type=Path,
        default=Path("analysis_statistics.json"),
        help="Detailed numerical statistics JSON output path.",
    )
    parser.add_argument(
        "--panels-dir",
        type=Path,
        default=Path("analysis_panels"),
        help="Directory where each individual analysis panel PNG is saved.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def slugify_filename(text: str) -> str:
    lowered = text.strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    lowered = lowered.strip("_")
    return lowered or "panel"


def pretty_label(value) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text.replace("_", " ")).strip()
    if text.islower():
        return text.title()
    return text


def add_chart_key(ax, key_text: str):
    ax.text(
        0.0,
        -0.24,
        key_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#444444",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "#f7f7f7",
            "edgecolor": "#cccccc",
            "alpha": 0.95,
        },
    )


def save_individual_panels(fig, panel_items, panels_dir: Path, dpi=220):
    panels_dir.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    saved = []
    for name, ax in panel_items:
        if ax is None or not ax.get_visible():
            continue
        bbox = ax.get_tightbbox(renderer)
        if bbox is None:
            continue
        bbox = bbox.expanded(1.15, 1.22)
        out_path = panels_dir / f"{slugify_filename(name)}.png"
        fig.savefig(
            out_path,
            dpi=dpi,
            bbox_inches=bbox.transformed(fig.dpi_scale_trans.inverted()),
        )
        saved.append(out_path)
    return saved


def is_missing(value):
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in MISSING_STRINGS:
        return True
    return False


def clean_string(value):
    if is_missing(value):
        return None
    if isinstance(value, str):
        return value.strip().lower()
    return str(value).strip().lower()


def clean_numeric(value):
    if is_missing(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number.is_integer():
        return int(number)
    return number


def clean_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return None


def ordered_unique(values, preferred=None):
    preferred = preferred or []
    result = []
    seen = set()
    for item in preferred + list(values):
        if item in seen:
            continue
        if item in values:
            result.append(item)
            seen.add(item)
    return result


def estimate_figure_size(df):
    variants_n = max(1, int(df["variant"].dropna().nunique()))
    gt_skin_n = max(1, int(df["gt_skin_tone"].dropna().nunique()))
    gt_gender_n = max(1, int(df["gt_gender"].dropna().nunique()))

    variant_gender_groups = 0
    if {"variant", "gt_gender"}.issubset(df.columns):
        variant_gender_groups = int(
            df[["variant", "gt_gender"]]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .shape[0]
        )

    # Base size is intentionally large, then expanded for dense labels/columns.
    width = 50 + 1.8 * max(0, variants_n - 8) + 0.6 * max(0, variant_gender_groups - 10)
    height = 62 + 1.6 * max(0, variants_n - 8) + 0.8 * max(0, gt_skin_n - 6) + 0.8 * max(0, gt_gender_n - 2)

    # Guardrails so outputs are never too small for this dense dashboard.
    width = max(50, min(width, 95))
    height = max(62, min(height, 120))
    return width, height


def load_records(images_root: Path, annotations_file: str, ground_truth_file: str, limit=None):
    subfolders = sorted([path for path in images_root.iterdir() if path.is_dir() and path.name.startswith("sa_")])
    if limit is not None:
        subfolders = subfolders[:limit]

    records = []
    missing = []
    for subfolder in subfolders:
        gt_path = subfolder / ground_truth_file
        ann_path = subfolder / annotations_file
        if not gt_path.exists() or not ann_path.exists():
            missing.append(subfolder.name)
            continue

        with gt_path.open("r", encoding="utf-8") as handle:
            gt = json.load(handle)
        with ann_path.open("r", encoding="utf-8") as handle:
            ann = json.load(handle)

        images = ann.get("images", {}) if isinstance(ann, dict) else {}
        for variant, payload in images.items():
            answers = payload.get("answers", {}) if isinstance(payload, dict) else {}
            records.append(
                {
                    "subfolder": subfolder.name,
                    "person_id": str(gt.get("person_id")) if gt.get("person_id") is not None else None,
                    "image_file": gt.get("image_file"),
                    "variant": variant,
                    "is_original": variant == "original",
                    "gt_gender": clean_string(gt.get("gender")),
                    "gt_age_bracket": clean_string(gt.get("age")),
                    "gt_skin_tone": clean_numeric(gt.get("skin_tone")),
                    "perceived_skin_tone": clean_numeric(answers.get("perceived_skin_tone")),
                    "perceived_gender": clean_string(answers.get("perceived_gender")),
                    "perceived_age_bracket": clean_string(answers.get("perceived_age_bracket")),
                    "perceived_race_ethnicity": clean_string(answers.get("perceived_race_ethnicity")),
                    "boundary_artifact_visible": clean_bool(answers.get("boundary_artifact_visible")),
                    "lighting_consistency": clean_numeric(answers.get("lighting_consistency")),
                    "person_setting_fit": clean_numeric(answers.get("person_setting_fit")),
                    "background_density": clean_numeric(answers.get("background_density")),
                    "scene_socioeconomic_signal": clean_numeric(answers.get("scene_socioeconomic_signal")),
                    "scene_semantic_category": clean_string(answers.get("scene_semantic_category")),
                    "occupational_context": clean_string(answers.get("occupational_context")),
                    "geographic_location_guess": clean_string(answers.get("geographic_location_guess")),
                    "annotator_confidence": clean_numeric(answers.get("annotator_confidence")),
                }
            )

    return pd.DataFrame.from_records(records), missing


def heatmap(ax, table, title, cmap="viridis", fmt=".1f", cbar_label=None, value_key=None):
    if table is None or table.empty:
        ax.set_axis_off()
        ax.set_title(f"{title}\n(no data)")
        return
    n_rows, n_cols = table.shape
    dense = (n_rows * n_cols) > 80 or n_rows > 10 or n_cols > 10

    sns.heatmap(
        table,
        ax=ax,
        cmap=cmap,
        annot=not dense,
        annot_kws={
            "size": 10,
            "color": "black",
            "fontweight": "bold",
            "bbox": {
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.7,
            },
        },
        fmt=fmt,
        linewidths=0.4,
        linecolor="#e0e0e0",
        cbar_kws={"label": cbar_label} if cbar_label else None,
    )
    ax.set_title(pretty_label(title), fontsize=13)
    ax.set_xticklabels([pretty_label(tick.get_text()) for tick in ax.get_xticklabels()])
    ax.set_yticklabels([pretty_label(tick.get_text()) for tick in ax.get_yticklabels()])
    ax.tick_params(axis="x", rotation=35 if dense else 45, labelsize=8 if dense else 10)
    ax.tick_params(axis="y", rotation=0, labelsize=8 if dense else 10)

    if value_key:
        ax.text(
            0.0,
            -0.18,
            f"Key: {value_key}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="black",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "#f7f7f7",
                "edgecolor": "#cccccc",
                "alpha": 0.95,
            },
        )


def line_by_gt_skin_tone(ax, df, value_col, title, ylabel):
    subset = df[["gt_skin_tone", value_col]].dropna().copy()
    if subset.empty:
        ax.set_axis_off()
        ax.set_title(f"{title}\n(no data)")
        return
    summary = subset.groupby("gt_skin_tone")[value_col].agg(["mean", "count"]).reset_index()
    ax.plot(summary["gt_skin_tone"], summary["mean"], marker="o", linewidth=2)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("gt_skin_tone", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(pretty_label(title), fontsize=13)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, alpha=0.25)
    for _, row in summary.iterrows():
        ax.text(row["gt_skin_tone"], row["mean"], f" n={int(row['count'])}", fontsize=9, va="bottom")


def bar_by_gt_skin_tone(ax, df, value_col, title, ylabel):
    subset = df[["gt_skin_tone", value_col]].dropna().copy()
    if subset.empty:
        ax.set_axis_off()
        ax.set_title(f"{title}\n(no data)")
        return

    summary = subset.groupby("gt_skin_tone")[value_col].agg(["mean", "count"]).reset_index()
    summary = summary.sort_values("gt_skin_tone")

    ax.bar(summary["gt_skin_tone"], summary["mean"], color="#4C78A8", alpha=0.9)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("gt_skin_tone", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(pretty_label(title), fontsize=13)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, axis="y", alpha=0.25)

    for _, row in summary.iterrows():
        ax.text(
            row["gt_skin_tone"],
            row["mean"],
            f" n={int(row['count'])}",
            fontsize=9,
            va="bottom",
            ha="center",
        )


def socioeconomic_zoomed_bar(ax, df, title):
    subset = df[["gt_skin_tone", "scene_socioeconomic_signal"]].dropna().copy()
    if subset.empty:
        ax.set_axis_off()
        ax.set_title(f"{title}\n(no data)")
        return

    summary = subset.groupby("gt_skin_tone")["scene_socioeconomic_signal"].agg(["mean", "count"]).reset_index()
    summary = summary.sort_values("gt_skin_tone")

    x_vals = summary["gt_skin_tone"].astype(float)
    y_vals = summary["mean"].astype(float)

    ax.bar(x_vals, y_vals, color="#4C78A8", alpha=0.92, width=0.75)
    ax.set_title(pretty_label(title), fontsize=13)
    ax.set_xlabel("gt_skin_tone scale (0=lighter, 10=darker)", fontsize=11)
    ax.set_ylabel("mean scene_socioeconomic_signal (0=lower, 5=higher)", fontsize=11)

    ax.set_xlim(0, 10)
    ax.set_xticks(range(0, 11))

    y_min = float(y_vals.min())
    y_max = float(y_vals.max())
    y_range = max(0.01, y_max - y_min)
    pad = max(0.08, y_range * 0.25)
    lower = max(0.0, y_min - pad)
    upper = min(5.0, y_max + pad)
    if upper - lower < 0.35:
        center = (upper + lower) / 2.0
        lower = max(0.0, center - 0.175)
        upper = min(5.0, center + 0.175)
    ax.set_ylim(lower, upper)

    ax.grid(True, axis="y", alpha=0.25)
    for _, row in summary.iterrows():
        ax.text(
            float(row["gt_skin_tone"]),
            float(row["mean"]),
            f" n={int(row['count'])}",
            fontsize=8,
            va="bottom",
            ha="center",
        )

    add_chart_key(
        ax,
        "Key: skin tone 0=lighter, 10=darker | socioeconomic 0=lower, 5=higher",
    )


def person_setting_fit_zoomed_bar(ax, df, title):
    subset = df[["gt_skin_tone", "person_setting_fit"]].dropna().copy()
    if subset.empty:
        ax.set_axis_off()
        ax.set_title(f"{title}\n(no data)")
        return

    summary = subset.groupby("gt_skin_tone")["person_setting_fit"].agg(["mean", "count"]).reset_index()
    summary = summary.sort_values("gt_skin_tone")

    x_vals = summary["gt_skin_tone"].astype(float)
    y_vals = summary["mean"].astype(float)

    ax.bar(x_vals, y_vals, color="#2E8B57", alpha=0.92, width=0.75)
    ax.set_title(pretty_label(title), fontsize=13)
    ax.set_xlabel("gt_skin_tone scale (0=lighter, 10=darker)", fontsize=11)
    ax.set_ylabel("mean person_setting_fit (0=poor fit, 5=strong fit)", fontsize=11)

    ax.set_xlim(0, 10)
    ax.set_xticks(range(0, 11))

    y_min = float(y_vals.min())
    y_max = float(y_vals.max())
    y_range = max(0.01, y_max - y_min)
    pad = max(0.08, y_range * 0.25)
    lower = max(0.0, y_min - pad)
    upper = min(5.0, y_max + pad)
    if upper - lower < 0.35:
        center = (upper + lower) / 2.0
        lower = max(0.0, center - 0.175)
        upper = min(5.0, center + 0.175)
    ax.set_ylim(lower, upper)

    ax.grid(True, axis="y", alpha=0.25)
    for _, row in summary.iterrows():
        ax.text(
            float(row["gt_skin_tone"]),
            float(row["mean"]),
            f" n={int(row['count'])}",
            fontsize=8,
            va="bottom",
            ha="center",
        )

    add_chart_key(
        ax,
        "Key: skin tone 0=lighter, 10=darker | person-setting fit 0=poor, 5=strong",
    )


def occupational_gender_difference_bar(ax, df, title):
    subset = df[["gt_gender", "occupational_context"]].dropna().copy()
    if subset.empty:
        ax.set_axis_off()
        ax.set_title(f"{title}\n(no data)")
        return

    subset["gt_gender"] = subset["gt_gender"].astype(str).str.lower()
    subset = subset[subset["gt_gender"].isin(["male", "female"])].copy()
    if subset.empty:
        ax.set_axis_off()
        ax.set_title(f"{title}\n(no male/female rows)")
        return

    table = pd.crosstab(subset["occupational_context"].astype(str), subset["gt_gender"])
    if len(table.index) > 12:
        totals = table.sum(axis=1).sort_values(ascending=False)
        top_labels = list(totals.head(12).index)
        other_count = int(totals.iloc[12:].sum())
        table = table.loc[top_labels]
        if other_count > 0:
            other_rows = subset[~subset["occupational_context"].astype(str).isin(top_labels)]
            other_table = pd.crosstab(other_rows["occupational_context"].astype(str), other_rows["gt_gender"]).sum(axis=0)
            if not other_table.empty:
                table.loc["other"] = [
                    int(other_table.get("male", 0)),
                    int(other_table.get("female", 0)),
                ]
    for col in ["male", "female"]:
        if col not in table.columns:
            table[col] = 0
    table = table[["male", "female"]]

    total_male = float(table["male"].sum())
    total_female = float(table["female"].sum())
    if total_male == 0 or total_female == 0:
        ax.set_axis_off()
        ax.set_title(f"{title}\n(insufficient male/female counts)")
        return

    # Use within-gender shares, then difference so each occupation has one diverging bar.
    table["male_pct"] = table["male"] * 100.0 / total_male
    table["female_pct"] = table["female"] * 100.0 / total_female
    table["diff_pp_female_minus_male"] = table["female_pct"] - table["male_pct"]
    table = table.sort_values("diff_pp_female_minus_male")

    y_labels = table.index.tolist()
    y_labels_display = [pretty_label(label) for label in y_labels]
    values = table["diff_pp_female_minus_male"].tolist()
    colors = ["#d62728" if val < 0 else "#1f77b4" for val in values]

    y_positions = np.arange(len(y_labels))
    bars = ax.barh(y_positions, values, color=colors, alpha=0.9)
    ax.set_yticks(y_positions, labels=y_labels_display)
    ax.axvline(0, color="#444444", linewidth=1.0)
    ax.set_title(pretty_label(title), fontsize=13)
    ax.set_xlabel("Difference (percentage points): female share - male share")
    ax.set_ylabel("occupational context")
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, axis="x", alpha=0.25)

    max_abs = max(abs(v) for v in values) if values else 1.0
    pad = max(2.0, max_abs * 0.15)
    ax.set_xlim(-max_abs - pad, max_abs + pad)

    for bar, val in zip(bars, values):
        y_mid = bar.get_y() + bar.get_height() / 2
        x_pos = val + (0.25 if val >= 0 else -0.25)
        align = "left" if val >= 0 else "right"
        ax.text(x_pos, y_mid, f"{val:+.1f}", va="center", ha=align, fontsize=9)

    add_chart_key(ax, "Right/blue = more female, Left/red = more male")


def confusion_heatmap(ax, df, truth_col, pred_col, labels, title):
    subset = df[[truth_col, pred_col]].dropna().copy()
    if subset.empty:
        ax.set_axis_off()
        ax.set_title(f"{title}\n(no data)")
        return None
    table = pd.crosstab(subset[truth_col].astype(str), subset[pred_col].astype(str))
    ordered = [str(label) for label in labels]
    table = table.reindex(index=ordered, columns=ordered, fill_value=0)
    normalized = table.div(table.sum(axis=1).replace(0, np.nan), axis=0) * 100.0
    heatmap(
        ax,
        normalized,
        title,
        cmap="viridis",
        fmt=".1f",
        cbar_label="%",
        value_key="Cell number = row-wise percent (%)",
    )
    return table


def numeric_group_heatmap(ax, df, row_col, col_col, value_col, title, row_order=None, col_order=None, cmap="mako", fmt=".1f"):
    subset = df[[row_col, col_col, value_col]].dropna().copy()
    if subset.empty:
        ax.set_axis_off()
        ax.set_title(f"{title}\n(no data)")
        return
    table = subset.pivot_table(index=row_col, columns=col_col, values=value_col, aggfunc="mean")
    if row_order is not None:
        table = table.reindex(index=[str(item) for item in row_order])
    if col_order is not None:
        table = table.reindex(columns=[str(item) for item in col_order])
    heatmap(ax, table, title, cmap=cmap, fmt=fmt, cbar_label=value_col, value_key=f"Cell number = mean {value_col}")


def distribution_heatmap(ax, df, group_cols, category_col, title, category_order=None, top_n=6, cmap="rocket_r"):
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    subset = df[group_cols + [category_col]].dropna().copy()
    if subset.empty:
        ax.set_axis_off()
        ax.set_title(f"{title}\n(no data)")
        return

    subset["__group__"] = subset[group_cols].astype(str).agg(" | ".join, axis=1)
    values = subset[category_col].astype(str)
    if category_order is None:
        category_order = list(values.value_counts().head(top_n).index)
    else:
        category_order = [str(item) for item in category_order if item in set(values)]
        for item in values.value_counts().head(top_n).index:
            if item not in category_order:
                category_order.append(item)

    subset[category_col] = subset[category_col].astype(str)
    subset.loc[~subset[category_col].isin(category_order), category_col] = "other"
    if "other" in subset[category_col].values and "other" not in category_order:
        category_order.append("other")

    group_order = subset["__group__"].value_counts().index.tolist()
    if len(group_order) > 12:
        group_order = group_order[:12]
        subset = subset[subset["__group__"].isin(group_order)].copy()

    table = pd.crosstab(subset[category_col], subset["__group__"], normalize="columns") * 100.0
    table = table.reindex(index=category_order)
    if group_order:
        table = table.reindex(columns=group_order)
    heatmap(
        ax,
        table,
        title,
        cmap=cmap,
        fmt=".1f",
        cbar_label="% of group",
        value_key="Cell number = within-group percent (%)",
    )


def build_pairwise_frame(df):
    original = (
        df[df["is_original"]]
        [["subfolder", "perceived_skin_tone", "perceived_gender", "perceived_age_bracket", "perceived_race_ethnicity"]]
        .rename(
            columns={
                "perceived_skin_tone": "orig_perceived_skin_tone",
                "perceived_gender": "orig_perceived_gender",
                "perceived_age_bracket": "orig_perceived_age_bracket",
                "perceived_race_ethnicity": "orig_perceived_race_ethnicity",
            }
        )
    )
    paired = df.merge(original, on="subfolder", how="left")
    paired = paired[paired["variant"] != "original"].copy()
    paired["skin_tone_abs_delta"] = (paired["perceived_skin_tone"] - paired["orig_perceived_skin_tone"]).abs()
    paired["gender_flip"] = (paired["perceived_gender"] != paired["orig_perceived_gender"]).astype(float)
    paired["age_flip"] = (paired["perceived_age_bracket"] != paired["orig_perceived_age_bracket"]).astype(float)
    paired["race_flip"] = (paired["perceived_race_ethnicity"] != paired["orig_perceived_race_ethnicity"]).astype(float)
    return paired


def per_person_stability(df, column):
    rows = []
    for subfolder, group in df.groupby("subfolder"):
        gt_skin = group["gt_skin_tone"].dropna().iloc[0] if not group["gt_skin_tone"].dropna().empty else np.nan
        values = group[column].dropna()
        if values.empty:
            stability = np.nan
        elif column == "perceived_skin_tone":
            stability = float(values.astype(float).var(ddof=0))
        else:
            probs = values.astype(str).value_counts(normalize=True)
            stability = float(-(probs * np.log2(probs)).sum())
        rows.append({"subfolder": subfolder, "gt_skin_tone": gt_skin, f"{column}_stability": stability})
    return pd.DataFrame(rows)


def build_summary(df, missing_subfolders):
    summary = {
        "records": int(len(df)),
        "persons": int(df["subfolder"].nunique()),
        "variants": int(df["variant"].nunique()),
        "missing_subfolders": missing_subfolders,
        "gt_skin_tone_counts": {str(k): int(v) for k, v in df["gt_skin_tone"].dropna().value_counts().sort_index().items()},
    }

    skin = df[["gt_skin_tone", "perceived_skin_tone"]].dropna()
    if not skin.empty:
        summary["skin_tone_mae"] = float((skin["gt_skin_tone"] - skin["perceived_skin_tone"]).abs().mean())
        summary["skin_tone_signed_bias"] = float((skin["perceived_skin_tone"] - skin["gt_skin_tone"]).mean())
        summary["skin_tone_accuracy"] = float((skin["gt_skin_tone"] == skin["perceived_skin_tone"]).mean())

    gender = df[["gt_gender", "perceived_gender"]].dropna()
    if not gender.empty:
        summary["gender_accuracy"] = float((gender["gt_gender"] == gender["perceived_gender"]).mean())

    age = df[["gt_age_bracket", "perceived_age_bracket"]].dropna()
    if not age.empty:
        summary["age_accuracy"] = float((age["gt_age_bracket"] == age["perceived_age_bracket"]).mean())

    pairwise = build_pairwise_frame(df)
    if not pairwise.empty:
        summary["pairwise_stability_by_variant"] = (
            pairwise.groupby("variant")[["skin_tone_abs_delta", "gender_flip", "age_flip", "race_flip"]]
            .mean()
            .rename(
                columns={
                    "skin_tone_abs_delta": "skin_tone_abs_delta",
                    "gender_flip": "gender_flip_rate",
                    "age_flip": "age_flip_rate",
                    "race_flip": "race_flip_rate",
                }
            )
            .round(4)
            .to_dict(orient="index")
        )

    return summary, pairwise


def _safe_mean(series):
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.mean())


def _value_counts(series):
    values = series.dropna().astype(str)
    if values.empty:
        return {}
    return {str(k): int(v) for k, v in values.value_counts().sort_index().items()}


def _normalized_counts(series):
    values = series.dropna().astype(str)
    if values.empty:
        return {}
    dist = values.value_counts(normalize=True).sort_index()
    return {str(k): float(v) for k, v in dist.items()}


def build_detailed_statistics(df):
    perceived_numeric_cols = [
        "perceived_skin_tone",
        "lighting_consistency",
        "person_setting_fit",
        "background_density",
        "scene_socioeconomic_signal",
        "annotator_confidence",
    ]
    perceived_categorical_cols = [
        "perceived_gender",
        "perceived_age_bracket",
        "perceived_race_ethnicity",
        "scene_semantic_category",
        "occupational_context",
        "geographic_location_guess",
    ]

    stats = {
        "totals": {
            "rows": int(len(df)),
            "persons": int(df["subfolder"].nunique()),
            "locations": int(df["variant"].nunique()),
            "rows_per_location": {str(k): int(v) for k, v in df["variant"].value_counts().sort_index().items()},
        },
        "ground_truth": {
            "gender_counts": _value_counts(df["gt_gender"]),
            "age_counts": _value_counts(df["gt_age_bracket"]),
            "skin_tone_counts": {str(k): int(v) for k, v in df["gt_skin_tone"].dropna().value_counts().sort_index().items()},
            "skin_tone_mean": _safe_mean(df["gt_skin_tone"]),
        },
        "perceived": {
            "skin_tone_counts": {str(k): int(v) for k, v in df["perceived_skin_tone"].dropna().value_counts().sort_index().items()},
            "numeric_means_overall": {col: _safe_mean(df[col]) for col in perceived_numeric_cols},
            "categorical_counts_overall": {col: _value_counts(df[col]) for col in perceived_categorical_cols},
            "categorical_shares_overall": {col: _normalized_counts(df[col]) for col in perceived_categorical_cols},
        },
        "by_location": {},
    }

    for location, group in df.groupby("variant"):
        location_key = str(location)
        stats["by_location"][location_key] = {
            "rows": int(len(group)),
            "persons": int(group["subfolder"].nunique()),
            "perceived_skin_tone_counts": {
                str(k): int(v)
                for k, v in group["perceived_skin_tone"].dropna().value_counts().sort_index().items()
            },
            "perceived_numeric_means": {col: _safe_mean(group[col]) for col in perceived_numeric_cols},
            "perceived_categorical_counts": {col: _value_counts(group[col]) for col in perceived_categorical_cols},
            "perceived_categorical_shares": {col: _normalized_counts(group[col]) for col in perceived_categorical_cols},
        }

    return stats


def plot_dashboard(df, summary, pairwise, output_figure: Path, panels_dir: Path):
    sns.set_theme(style="whitegrid", context="talk")
    sns.set_context("talk", rc={"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11})
    variants = ordered_unique(df["variant"].dropna().tolist(), preferred=["original"])
    synthetic_variants = [variant for variant in variants if variant != "original"]
    skin_order = list(range(1, 11))

    fig_width, fig_height = estimate_figure_size(df)
    fig, axes = plt.subplots(
        5,
        4,
        figsize=(fig_width, fig_height),
        gridspec_kw={"hspace": 0.9, "wspace": 0.6},
    )
    axes = axes.flatten()

    line_by_gt_skin_tone(axes[0], df, "boundary_artifact_visible", "#17 gt_skin_tone vs boundary_artifact_visible rate", "boundary rate")
    line_by_gt_skin_tone(axes[1], df, "lighting_consistency", "#18 gt_skin_tone vs lighting_consistency", "mean lighting")
    person_setting_fit_zoomed_bar(axes[2], df, "#19 gt_skin_tone vs person_setting_fit")

    tier1 = []
    for metric, scale in [("boundary_artifact_visible", 100.0), ("lighting_consistency", 1.0)]:
        temp = df[["variant", metric]].dropna().copy()
        if temp.empty:
            continue
        temp["value"] = temp[metric].astype(float) * scale
        temp["metric"] = metric
        tier1.append(temp[["variant", "metric", "value"]])
    if tier1:
        tier1_df = pd.concat(tier1, ignore_index=True)
        table = tier1_df.pivot_table(index="metric", columns="variant", values="value", aggfunc="mean")
        table = table.reindex(index=["boundary_artifact_visible", "lighting_consistency"])
        table = table.reindex(columns=variants)
        heatmap(
            axes[3],
            table,
            "#29-30 background_variant vs boundary_artifact_visible / lighting_consistency",
            cmap="mako",
            fmt=".1f",
            cbar_label="mean",
            value_key="Boundary row = % with visible artifact (error-like rate). Lighting row = mean score (1-5).",
        )
    else:
        axes[3].set_axis_off()

    skin_table = pd.crosstab(df["gt_skin_tone"].dropna().astype(int), df["perceived_skin_tone"].dropna().astype(int))
    skin_table = skin_table.reindex(index=skin_order, columns=skin_order, fill_value=0)
    if not skin_table.empty:
        skin_norm = skin_table.div(skin_table.sum(axis=1).replace(0, np.nan), axis=0) * 100.0
        heatmap(
            axes[4],
            skin_norm,
            f"#1 gt_skin_tone vs perceived_skin_tone (MAE={summary.get('skin_tone_mae', float('nan')):.2f}, bias={summary.get('skin_tone_signed_bias', float('nan')):.2f})",
            cmap="viridis",
            fmt=".1f",
            cbar_label="%",
            value_key="For each gt row: % predicted into each column (row sums to ~100%).",
        )
    else:
        axes[4].set_axis_off()

    gender_labels = ordered_unique(df["gt_gender"].dropna().astype(str).tolist() + df["perceived_gender"].dropna().astype(str).tolist(), preferred=GT_GENDER_ORDER)
    gender = pd.crosstab(df["gt_gender"].dropna().astype(str), df["perceived_gender"].dropna().astype(str))
    gender = gender.reindex(index=gender_labels, columns=gender_labels, fill_value=0)
    if not gender.empty:
        gender_norm = gender.div(gender.sum(axis=1).replace(0, np.nan), axis=0) * 100.0
        heatmap(
            axes[5],
            gender_norm,
            "#2 gt_gender vs perceived_gender",
            cmap="viridis",
            fmt=".1f",
            cbar_label="%",
            value_key="For each gt row: % predicted into each column (row sums to ~100%).",
        )
    else:
        axes[5].set_axis_off()

    age_labels = ordered_unique(df["gt_age_bracket"].dropna().astype(str).tolist() + df["perceived_age_bracket"].dropna().astype(str).tolist(), preferred=GT_AGE_ORDER)
    age = pd.crosstab(df["gt_age_bracket"].dropna().astype(str), df["perceived_age_bracket"].dropna().astype(str))
    age = age.reindex(index=age_labels, columns=age_labels, fill_value=0)
    if not age.empty:
        age_norm = age.div(age.sum(axis=1).replace(0, np.nan), axis=0) * 100.0
        heatmap(
            axes[6],
            age_norm,
            "#3 gt_age_bracket vs perceived_age_bracket",
            cmap="viridis",
            fmt=".1f",
            cbar_label="%",
            value_key="For each gt row: % predicted into each column (row sums to ~100%).",
        )
    else:
        axes[6].set_axis_off()

    calib = df[["gt_skin_tone", "perceived_skin_tone", "annotator_confidence"]].dropna().copy()
    if not calib.empty:
        calib["correct"] = (calib["gt_skin_tone"] == calib["perceived_skin_tone"]).astype(float)
        table = calib.pivot_table(index="gt_skin_tone", columns="annotator_confidence", values="correct", aggfunc="mean") * 100.0
        table = table.reindex(index=skin_order)
        table = table.reindex(columns=sorted(calib["annotator_confidence"].dropna().unique()))
        heatmap(
            axes[7],
            table,
            "#7 confidence vs correctness by gt_skin_tone",
            cmap="mako",
            fmt=".1f",
            cbar_label="accuracy %",
            value_key="Cell number = percent right (accuracy %).",
        )
    else:
        axes[7].set_axis_off()

    if not pairwise.empty:
        pairwise_table = pairwise.groupby("variant")[["skin_tone_abs_delta", "gender_flip", "age_flip", "race_flip"]].mean().T
        pairwise_table = pairwise_table.reindex(index=["skin_tone_abs_delta", "gender_flip", "age_flip", "race_flip"])
        pairwise_table = pairwise_table.reindex(columns=synthetic_variants)
        heatmap(
            axes[8],
            pairwise_table,
            "#31 original vs each synthetic variant, paired per person",
            cmap="mako",
            fmt=".2f",
            cbar_label="mean",
            value_key="Flip rows = percent changed vs original (%). Skin-tone row = mean absolute change (not %).",
        )
    else:
        axes[8].set_axis_off()

    if not pairwise.empty:
        flip_table = pairwise.groupby("variant")[["skin_tone_abs_delta", "gender_flip", "age_flip", "race_flip"]].mean().T
        flip_table = flip_table.reindex(index=["skin_tone_abs_delta", "gender_flip", "age_flip", "race_flip"])
        flip_table = flip_table.rename(
            index={
                "skin_tone_abs_delta": "skin_tone_error",
                "gender_flip": "gender_flip",
                "age_flip": "age_flip",
                "race_flip": "race_flip",
            }
        )
        flip_table = flip_table.reindex(columns=synthetic_variants)
        heatmap(
            axes[9],
            flip_table,
            "#23-26 background_variant vs perceived attribute flip rate",
            cmap="rocket_r",
            fmt=".1f",
            cbar_label="mean",
            value_key="Flip rows = percent changed (%). Skin-tone row = mean absolute change (not %).",
        )
    else:
        axes[9].set_axis_off()

    stability_frames = [
        per_person_stability(df, "perceived_skin_tone"),
        per_person_stability(df, "perceived_gender"),
        per_person_stability(df, "perceived_age_bracket"),
        per_person_stability(df, "perceived_race_ethnicity"),
    ]
    merged = stability_frames[0]
    for frame in stability_frames[1:]:
        merged = merged.merge(frame, on=["subfolder", "gt_skin_tone"], how="left")
    stability_cols = [col for col in merged.columns if col.endswith("_stability")]
    stability_table = merged.groupby("gt_skin_tone")[stability_cols].mean().T.reindex(columns=skin_order)
    heatmap(
        axes[10],
        stability_table,
        "#51-53 per-person variance/entropy grouped by gt_skin_tone",
        cmap="mako",
        fmt=".2f",
        cbar_label="mean stability",
        value_key="Skin-tone stability row = mean variance. Other rows = mean entropy (bits). Lower = more stable.",
    )

    distribution_heatmap(
        axes[11],
        df,
        group_cols=["gt_skin_tone"],
        category_col="geographic_location_guess",
        title="#12 gt_skin_tone vs geographic_location_guess",
        category_order=GT_GEO_ORDER,
        top_n=6,
    )

    socioeconomic_zoomed_bar(axes[12], df, "#11 gt_skin_tone vs scene_socioeconomic_signal")

    occupational_gender_difference_bar(
        axes[13],
        df,
        "#9 occupational_context difference (female vs male)",
    )

    distribution_heatmap(
        axes[14],
        df,
        group_cols=["gt_skin_tone"],
        category_col="scene_semantic_category",
        title="#13 gt_skin_tone vs scene_semantic_category",
        top_n=6,
    )

    gt_gender_skin = df[["gt_gender", "gt_skin_tone", "perceived_skin_tone"]].dropna().copy()
    if not gt_gender_skin.empty:
        gt_gender_skin["skin_correct"] = (gt_gender_skin["gt_skin_tone"] == gt_gender_skin["perceived_skin_tone"]).astype(float) * 100.0
        table = gt_gender_skin.pivot_table(index="gt_gender", columns="gt_skin_tone", values="skin_correct", aggfunc="mean")
        table = table.reindex(index=ordered_unique(gt_gender_skin["gt_gender"].astype(str).tolist(), preferred=GT_GENDER_ORDER))
        table = table.reindex(columns=skin_order)
        heatmap(
            axes[15],
            table,
            "#5 gender × skin_tone -> perceived_skin_tone error",
            cmap="viridis",
            fmt=".1f",
            cbar_label="accuracy %",
            value_key="Cell number = percent right (accuracy %).",
        )
    else:
        axes[15].set_axis_off()

    age_gender = df[["gt_age_bracket", "gt_gender", "perceived_gender"]].dropna().copy()
    if not age_gender.empty:
        age_gender["gender_correct"] = (age_gender["gt_gender"] == age_gender["perceived_gender"]).astype(float) * 100.0
        table = age_gender.pivot_table(index="gt_age_bracket", columns="gt_gender", values="gender_correct", aggfunc="mean")
        table = table.reindex(index=ordered_unique(age_gender["gt_age_bracket"].astype(str).tolist(), preferred=GT_AGE_ORDER))
        table = table.reindex(columns=ordered_unique(age_gender["gt_gender"].astype(str).tolist(), preferred=GT_GENDER_ORDER))
        heatmap(
            axes[16],
            table,
            "#6 age × gender -> perceived_gender error",
            cmap="viridis",
            fmt=".1f",
            cbar_label="accuracy %",
            value_key="Cell number = percent right (accuracy %).",
        )
    else:
        axes[16].set_axis_off()

    bg_skin = pairwise[["variant", "gt_skin_tone", "skin_tone_abs_delta", "scene_socioeconomic_signal"]].dropna().copy()
    if not bg_skin.empty:
        error_table = bg_skin.pivot_table(index="variant", columns="gt_skin_tone", values="skin_tone_abs_delta", aggfunc="mean").reindex(index=variants, columns=skin_order)
        ses_table = bg_skin.pivot_table(index="variant", columns="gt_skin_tone", values="scene_socioeconomic_signal", aggfunc="mean").reindex(index=variants, columns=skin_order)
        combined = pd.concat([error_table, ses_table], keys=["skin_tone_error", "socioeconomic_signal"], names=["metric", "variant"])
        heatmap(
            axes[17],
            combined,
            "#32-33 background_variant × gt_skin_tone",
            cmap="mako",
            fmt=".2f",
            cbar_label="mean value",
            value_key="Skin-tone-error block = mean absolute error. Socioeconomic block = mean score (1-5).",
        )
    else:
        axes[17].set_axis_off()

    panel_spec = axes[18].get_subplotspec().subgridspec(2, 1, hspace=0.35)
    axes[18].set_axis_off()
    occ_ax = fig.add_subplot(panel_spec[0, 0])
    geo_ax = fig.add_subplot(panel_spec[1, 0])
    bg_gender = df[["variant", "gt_gender", "occupational_context", "geographic_location_guess"]].dropna().copy()
    if not bg_gender.empty:
        distribution_heatmap(
            occ_ax,
            bg_gender,
            group_cols=["variant", "gt_gender"],
            category_col="occupational_context",
            title="#34 background_variant × gt_gender -> occupational_context",
            top_n=5,
        )
        distribution_heatmap(
            geo_ax,
            bg_gender,
            group_cols=["variant", "gt_gender"],
            category_col="geographic_location_guess",
            title="#35 background_variant × gt_gender -> geographic_location_guess",
            category_order=GT_GEO_ORDER,
            top_n=5,
        )
    else:
        occ_ax.set_axis_off()
        geo_ax.set_axis_off()

    axes[19].set_axis_off()
    summary_lines = [
        f"images: {summary['records']}",
        f"persons: {summary['persons']}",
        f"variants: {summary['variants']}",
        f"skin tone MAE: {summary.get('skin_tone_mae', float('nan')):.3f}",
        f"skin tone bias: {summary.get('skin_tone_signed_bias', float('nan')):.3f}",
        f"skin tone acc: {summary.get('skin_tone_accuracy', float('nan')):.3f}",
        f"gender acc: {summary.get('gender_accuracy', float('nan')):.3f}",
        f"age acc: {summary.get('age_accuracy', float('nan')):.3f}",
        "gt_race: unavailable in current ground_truth.json",
    ]
    key_text = "Metric key\n\n" + "\n".join(PANEL_VALUE_KEY)
    axes[19].text(
        0.0,
        1.0,
        "Summary\n\n" + "\n".join(summary_lines) + "\n\n" + key_text,
        ha="left",
        va="top",
        fontsize=11,
        family="monospace",
    )
    axes[19].set_title("Report summary", fontsize=13)

    fig.suptitle("Cross-image bias and stability dashboard", fontsize=26)
    fig.subplots_adjust(left=0.05, right=0.985, top=0.965, bottom=0.035, hspace=0.9, wspace=0.6)

    panel_items = [
        ("17_gt_skin_tone_vs_boundary_artifact_visible_rate", axes[0]),
        ("18_gt_skin_tone_vs_lighting_consistency", axes[1]),
        ("19_gt_skin_tone_vs_person_setting_fit", axes[2]),
        ("29_30_background_variant_vs_boundary_artifact_visible_lighting_consistency", axes[3]),
        ("1_gt_skin_tone_vs_perceived_skin_tone", axes[4]),
        ("2_gt_gender_vs_perceived_gender", axes[5]),
        ("3_gt_age_bracket_vs_perceived_age_bracket", axes[6]),
        ("7_confidence_vs_correctness_by_gt_skin_tone", axes[7]),
        ("31_original_vs_each_synthetic_variant", axes[8]),
        ("23_26_background_variant_vs_perceived_attribute_flip_rate", axes[9]),
        ("51_53_per_person_variance_entropy_grouped_by_gt_skin_tone", axes[10]),
        ("12_gt_skin_tone_vs_geographic_location_guess", axes[11]),
        ("11_gt_skin_tone_vs_scene_socioeconomic_signal", axes[12]),
        ("9_gt_gender_vs_occupational_context", axes[13]),
        ("13_gt_skin_tone_vs_scene_semantic_category", axes[14]),
        ("5_gender_x_skin_tone_error", axes[15]),
        ("6_age_x_gender_error", axes[16]),
        ("32_33_background_variant_x_gt_skin_tone", axes[17]),
        ("34_background_variant_x_gt_gender_to_occupational_context", occ_ax),
        ("35_background_variant_x_gt_gender_to_geographic_location_guess", geo_ax),
        ("summary", axes[19]),
    ]

    output_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_figure, dpi=320)
    panel_paths = save_individual_panels(fig, panel_items, panels_dir=panels_dir, dpi=340)
    return fig, panel_paths


def main():
    args = parse_args()
    df, missing = load_records(args.images_root, args.annotations_file, args.ground_truth_file, limit=args.limit)
    if df.empty:
        raise SystemExit("No usable records found.")

    summary, pairwise = build_summary(df, missing)
    detailed_stats = build_detailed_statistics(df)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    args.output_stats_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_stats_json.open("w", encoding="utf-8") as handle:
        json.dump(detailed_stats, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    fig, panel_paths = plot_dashboard(df, summary, pairwise, args.output_figure, args.panels_dir)
    print(f"Loaded {summary['persons']} persons across {summary['records']} image variants.")
    if "skin_tone_mae" in summary:
        print(f"Skin tone: MAE={summary['skin_tone_mae']:.3f}, bias={summary['skin_tone_signed_bias']:.3f}, acc={summary['skin_tone_accuracy']:.3f}")
    if "gender_accuracy" in summary:
        print(f"Gender accuracy: {summary['gender_accuracy']:.3f}")
    if "age_accuracy" in summary:
        print(f"Age accuracy: {summary['age_accuracy']:.3f}")
    print(f"Saved summary JSON to {args.output_json}")
    print(f"Saved detailed statistics JSON to {args.output_stats_json}")
    print(f"Saved dashboard figure to {args.output_figure}")
    print(f"Saved {len(panel_paths)} panel PNG files to {args.panels_dir}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
