import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MISSING_STRINGS = {"", "unsure", "unknown", "n/a", "none"}
DEFAULT_LOCATION_FILE = "location_attribute_annotations.json"
DEFAULT_GROUND_TRUTH_FILE = "ground_truth.json"
DEFAULT_CROSS_IMAGE_FILE = "cross_image_annotations.json"
DEFAULT_TAXONOMY_FILE = "location_attribute_labels.json"
DEFAULT_SIGNIFICANCE_CSV = "location_attribute_demographic_significance.csv"
ORIGINAL_VARIANT_KEY = "original"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze location attribute annotations by human attributes and create "
            "one grouped-bar chart per location x demographic slice."
        )
    )
    parser.add_argument("--images-root", type=Path, default=Path("places_inpainting_results"))
    parser.add_argument("--location-annotations-file", default=DEFAULT_LOCATION_FILE)
    parser.add_argument("--ground-truth-file", default=DEFAULT_GROUND_TRUTH_FILE)
    parser.add_argument("--cross-image-file", default=DEFAULT_CROSS_IMAGE_FILE)
    parser.add_argument("--taxonomy-file", type=Path, default=Path(DEFAULT_TAXONOMY_FILE))
    parser.add_argument(
        "--race-source",
        choices=["original_perceived", "skip"],
        default="original_perceived",
        help=(
            "Race is not present in ground_truth.json. Use the original-image "
            "perceived_race_ethnicity from cross_image_annotations.json, or skip race plots."
        ),
    )
    parser.add_argument("--output-figure", type=Path, default=Path("location_attribute_demographic_analysis.png"))
    parser.add_argument("--output-json", type=Path, default=Path("location_attribute_demographic_statistics.json"))
    parser.add_argument("--output-significance-csv", type=Path, default=Path(DEFAULT_SIGNIFICANCE_CSV))
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold. Only rows with p_value < alpha are kept in condensed JSON.",
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=30,
        help="Minimum sample size required for both compared groups in condensed JSON.",
    )
    parser.add_argument("--panels-dir", type=Path, default=Path("location_attribute_demographic_panels"))
    parser.add_argument(
        "--split-parts",
        type=int,
        default=4,
        help="Split each location/demographic chart into this many smaller graphs when possible.",
    )
    parser.add_argument(
        "--top-n-labels",
        type=int,
        default=0,
        help="Optional cap on flattened location labels per subplot; 0 means include the full taxonomy.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def is_missing(value):
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in MISSING_STRINGS:
        return True
    return False


def clean_string(value):
    if is_missing(value):
        return None
    return str(value).strip().lower()


def slugify_filename(text: str) -> str:
    lowered = text.strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    lowered = lowered.strip("_")
    return lowered or "panel"


def flatten_answers(answers):
    labels = []
    if not isinstance(answers, dict):
        return labels
    for attribute, value in answers.items():
        if isinstance(value, list):
            for item in value:
                if is_missing(item):
                    continue
                labels.append(f"{attribute}: {str(item).strip()}")
        elif not is_missing(value):
            labels.append(f"{attribute}: {str(value).strip()}")
    return labels


def load_race_proxy(cross_image_path: Path):
    if not cross_image_path.exists():
        return None
    with cross_image_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    images = data.get("images", {}) if isinstance(data, dict) else {}
    original = images.get(ORIGINAL_VARIANT_KEY)
    if not isinstance(original, dict):
        return None
    answers = original.get("answers", {})
    if not isinstance(answers, dict):
        return None
    return clean_string(answers.get("perceived_race_ethnicity"))


def load_location_label_universe(taxonomy_file: Path):
    if not taxonomy_file.exists():
        return {}

    with taxonomy_file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list):
        return {}

    universes = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        location = clean_string(item.get("location") or item.get("caption"))
        labels_by_attribute = item.get("labels_by_attribute")
        if not location or not isinstance(labels_by_attribute, dict):
            continue

        flattened = []
        for attribute in item.get("attribute_categories", []):
            label_values = labels_by_attribute.get(attribute, [])
            if not isinstance(label_values, list):
                continue
            for label in label_values:
                if is_missing(label):
                    continue
                flattened.append(f"{attribute}: {str(label).strip()}")
        universes[location] = flattened

    return universes


def load_records(images_root: Path, location_annotations_file: str, ground_truth_file: str, cross_image_file: str, race_source: str, limit=None):
    subfolders = sorted([path for path in images_root.iterdir() if path.is_dir() and path.name.startswith("sa_")])
    if limit is not None:
        subfolders = subfolders[:limit]

    records = []
    missing = []
    for subfolder in subfolders:
        location_path = subfolder / location_annotations_file
        gt_path = subfolder / ground_truth_file
        if not location_path.exists() or not gt_path.exists():
            missing.append(subfolder.name)
            continue

        with gt_path.open("r", encoding="utf-8") as handle:
            gt = json.load(handle)
        with location_path.open("r", encoding="utf-8") as handle:
            location_data = json.load(handle)

        race_proxy = None
        if race_source == "original_perceived":
            race_proxy = load_race_proxy(subfolder / cross_image_file)

        images = location_data.get("images", {}) if isinstance(location_data, dict) else {}
        for variant_name, payload in images.items():
            if variant_name == ORIGINAL_VARIANT_KEY:
                continue
            if not isinstance(payload, dict):
                continue
            answers = payload.get("answers", {})
            location_name = clean_string(payload.get("location")) or clean_string(variant_name)
            labels = flatten_answers(answers)
            records.append(
                {
                    "subfolder": subfolder.name,
                    "location": location_name,
                    "variant": clean_string(variant_name),
                    "gt_gender": clean_string(gt.get("gender")),
                    "gt_age": clean_string(gt.get("age")),
                    "race_proxy": race_proxy,
                    "labels": labels,
                }
            )

    return records, missing


def build_location_stats(records):
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    group_sizes = defaultdict(lambda: defaultdict(Counter))

    for row in records:
        location = row["location"]
        demographic_map = {
            "gender": row.get("gt_gender"),
            "race": row.get("race_proxy"),
            "age": row.get("gt_age"),
        }
        for demo_name, demo_value in demographic_map.items():
            if is_missing(demo_value):
                continue
            group_sizes[location][demo_name][demo_value] += 1
            for label in row.get("labels", []):
                grouped[location][demo_name][demo_value][label] += 1

    return grouped, group_sizes


def build_serializable_stats(grouped, group_sizes, location_label_universe):
    payload = {"locations": {}}
    for location in sorted(grouped.keys()):
        payload["locations"][location] = {}
        location_labels = list(location_label_universe.get(location, []))
        for demo_name in ("gender", "race", "age"):
            label_percentages = {}
            label_counts = {}
            for group, counter in grouped[location][demo_name].items():
                denom = group_sizes[location][demo_name].get(group, 0)
                if location_labels:
                    label_counts[str(group)] = {label: int(counter.get(label, 0)) for label in location_labels}
                else:
                    label_counts[str(group)] = {label: int(count) for label, count in counter.most_common()}
                if denom <= 0:
                    label_percentages[str(group)] = {label: 0.0 for label in label_counts[str(group)]}
                    continue
                label_percentages[str(group)] = {
                    label: float(count) * 100.0 / float(denom)
                    for label, count in label_counts[str(group)].items()
                }
            payload["locations"][location][demo_name] = {
                "label_universe": location_labels,
                "group_sizes": {str(k): int(v) for k, v in group_sizes[location][demo_name].items()},
                "label_counts": label_counts,
                "label_percentages": label_percentages,
            }
    return payload


def sort_significance_rows(rows):
    return sorted(
        rows,
        key=lambda row: (
            row["significance_score"] is None,
            -(row["significance_score"] if row["significance_score"] is not None else -1.0),
            -row["abs_difference"],
        ),
    )


def build_condensed_significance_payload(sorted_rows, race_source, ground_truth_file, taxonomy_file, missing, records, locations, alpha, min_group_size, output_significance_csv):
    filtered_rows = []
    for row in sorted_rows:
        p_value = row.get("p_value")
        n_a = int(row.get("n_a", 0))
        n_b = int(row.get("n_b", 0))
        if p_value is None:
            continue
        if p_value >= alpha:
            continue
        if n_a < min_group_size or n_b < min_group_size:
            continue
        filtered_rows.append(
            {
                "location": row["location"],
                "demographic": row["demographic"],
                "label": row["label"],
                "group_a": row["group_a"],
                "group_b": row["group_b"],
                "count_a": row["count_a"],
                "n_a": n_a,
                "pct_a": row["pct_a"],
                "count_b": row["count_b"],
                "n_b": n_b,
                "pct_b": row["pct_b"],
                "difference_pp": row["difference_pp"],
                "abs_difference": row["abs_difference"],
                "p_value": p_value,
                "significance_score": row["significance_score"],
                "used_top_two_groups": row["used_top_two_groups"],
            }
        )

    return {
        "meta": {
            "race_source": race_source,
            "ground_truth_source": ground_truth_file,
            "taxonomy_file": str(taxonomy_file),
            "missing_subfolders": missing,
            "records": records,
            "locations": locations,
            "statistical_test": {
                "name": "Two-proportion z-test (two-sided)",
                "null_hypothesis": "p(group_a) = p(group_b) for each label",
                "alternative_hypothesis": "p(group_a) != p(group_b) for each label",
                "p_value_formula": "2 * (1 - Phi(|z|))",
                "significance_score": "-log10(p_value)",
                "alpha": alpha,
                "minimum_group_size": min_group_size,
                "sorting": "descending significance_score, then descending abs_difference",
            },
            "output_significance_csv": str(output_significance_csv),
            "n_rows_before_filter": len(sorted_rows),
            "n_rows_after_filter": len(filtered_rows),
        },
        "results": filtered_rows,
    }


def save_individual_panels(fig, panel_items, panels_dir: Path, dpi=260):
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
        bbox = bbox.expanded(1.10, 1.18)
        out_path = panels_dir / f"{slugify_filename(name)}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches=bbox.transformed(fig.dpi_scale_trans.inverted()))
        saved.append(out_path)
    return saved


def estimate_figure_size(location_label_universe, n_rows):
    max_labels = 1
    if location_label_universe:
        max_labels = max(len(labels) for labels in location_label_universe.values() if labels) or 1

    fig_width = max(30, min(90, 18 + 0.55 * max_labels))
    fig_height = max(7 * n_rows, 20)
    return fig_width, fig_height


def select_plot_labels(location_labels, location_demo_counts, top_n_labels):
    if location_labels:
        labels = list(location_labels)
        if top_n_labels and top_n_labels > 0:
            labels = labels[:top_n_labels]
        return labels

    total_counter = Counter()
    for counter in location_demo_counts.values():
        total_counter.update(counter)
    if top_n_labels and top_n_labels > 0:
        return [label for label, _ in total_counter.most_common(top_n_labels)]
    return [label for label, _ in total_counter.most_common()]


def split_labels(labels, split_parts):
    if not labels:
        return []
    if split_parts <= 1 or len(labels) <= 1:
        return [labels]

    chunk_size = max(1, math.ceil(len(labels) / float(split_parts)))
    return [labels[index:index + chunk_size] for index in range(0, len(labels), chunk_size)]


def select_two_groups(location_group_sizes):
    if not location_group_sizes:
        return None
    ranked = sorted(location_group_sizes.items(), key=lambda item: (-item[1], str(item[0])))
    if len(ranked) < 2:
        return None
    return ranked[0][0], ranked[1][0], len(ranked) > 2


def two_proportion_p_value(count_a, n_a, count_b, n_b):
    if n_a <= 0 or n_b <= 0:
        return None
    pooled_n = n_a + n_b
    if pooled_n <= 0:
        return None

    pooled_p = float(count_a + count_b) / float(pooled_n)
    variance = pooled_p * (1.0 - pooled_p) * ((1.0 / float(n_a)) + (1.0 / float(n_b)))
    if variance <= 0.0:
        return 1.0

    z_value = ((float(count_a) / float(n_a)) - (float(count_b) / float(n_b))) / math.sqrt(variance)
    p_value = math.erfc(abs(z_value) / math.sqrt(2.0))
    return max(0.0, min(1.0, p_value))


def build_significance_rows(location, demo_name, location_demo_counts, location_group_sizes, labels):
    group_choice = select_two_groups(location_group_sizes)
    if group_choice is None:
        return []

    group_a, group_b, used_top_two = group_choice
    n_a = int(location_group_sizes.get(group_a, 0))
    n_b = int(location_group_sizes.get(group_b, 0))
    counter_a = location_demo_counts.get(group_a, Counter())
    counter_b = location_demo_counts.get(group_b, Counter())

    rows = []
    for label in labels:
        count_a = int(counter_a.get(label, 0))
        count_b = int(counter_b.get(label, 0))
        pct_a = 100.0 * float(count_a) / float(n_a) if n_a > 0 else 0.0
        pct_b = 100.0 * float(count_b) / float(n_b) if n_b > 0 else 0.0
        diff_pp = pct_a - pct_b
        p_value = two_proportion_p_value(count_a, n_a, count_b, n_b)
        if p_value is None:
            significance_score = None
        else:
            significance_score = -math.log10(max(p_value, 1e-300))

        rows.append(
            {
                "location": location,
                "demographic": demo_name,
                "group_a": str(group_a),
                "group_b": str(group_b),
                "label": label,
                "count_a": count_a,
                "n_a": n_a,
                "pct_a": pct_a,
                "count_b": count_b,
                "n_b": n_b,
                "pct_b": pct_b,
                "difference_pp": diff_pp,
                "abs_difference": abs(diff_pp),
                "p_value": p_value,
                "significance_score": significance_score,
                "significant_0_05": bool(p_value is not None and p_value < 0.05),
                "used_top_two_groups": used_top_two,
            }
        )
    return rows


def plot_location_demo(axis, location, demo_name, location_demo_counts, location_group_sizes, labels, part_label=None):
    group_choice = select_two_groups(location_group_sizes)
    if group_choice is None:
        axis.set_axis_off()
        axis.set_title(f"{location}\n{demo_name} (need at least two groups)")
        return

    group_a, group_b, used_top_two = group_choice
    n_a = location_group_sizes[group_a]
    n_b = location_group_sizes[group_b]

    if not labels:
        axis.set_axis_off()
        axis.set_title(f"{location}\n{demo_name} (no labels)")
        return

    x_spacing = 1.35 if len(labels) > 12 else 1.15
    x_positions = np.arange(len(labels), dtype=float) * x_spacing
    counter_a = location_demo_counts.get(group_a, Counter())
    counter_b = location_demo_counts.get(group_b, Counter())
    values = []
    colors = []
    for label in labels:
        pct_a = 100.0 * float(counter_a.get(label, 0)) / float(n_a) if n_a > 0 else 0.0
        pct_b = 100.0 * float(counter_b.get(label, 0)) / float(n_b) if n_b > 0 else 0.0
        diff = pct_a - pct_b
        values.append(diff)
        if diff > 0:
            colors.append("#1f77b4")
        elif diff < 0:
            colors.append("#d62728")
        else:
            colors.append("#9e9e9e")

    bar_width = min(0.78, 0.55 + 0.12 / max(len(labels), 1))
    axis.bar(x_positions, values, width=bar_width, color=colors)

    title = f"{location}\n{demo_name}"
    if part_label:
        title = f"{title}\n{part_label}"
    title = f"{title}\nDifference: {group_a} (n={n_a}) - {group_b} (n={n_b})"
    if used_top_two:
        title = f"{title} [top-2 groups by sample size]"
    axis.set_title(title, fontsize=12)
    axis.set_ylabel("Percentage-point difference", fontsize=10)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(labels, rotation=65, ha="right", fontsize=7)
    axis.tick_params(axis="y", labelsize=9)
    if len(x_positions) > 0:
        axis.set_xlim(x_positions[0] - 0.8, x_positions[-1] + 0.8)
    axis.axhline(0.0, color="#444444", linewidth=1.0, alpha=0.9)
    axis.grid(True, axis="y", alpha=0.25)
    axis.text(
        0.0,
        -0.23,
        "Key: positive = higher in group A; negative = higher in group B."
        " Significance for each label is written to CSV.",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "#f7f7f7", "edgecolor": "#cccccc", "alpha": 0.95},
    )


def save_location_split_figures(grouped, group_sizes, location_label_universe, panels_dir: Path, top_n_labels: int, split_parts: int):
    locations = sorted(grouped.keys())
    demo_columns = ["gender", "race", "age"]

    saved_paths = []
    location_summaries = []
    significance_rows = []

    for location in locations:
        location_folder = panels_dir / slugify_filename(location)
        location_folder.mkdir(parents=True, exist_ok=True)

        overview_rows = []
        max_parts = 1
        for demo_name in demo_columns:
            labels = select_plot_labels(location_label_universe.get(location, []), grouped[location][demo_name], top_n_labels)
            significance_rows.extend(
                build_significance_rows(
                    location,
                    demo_name,
                    grouped[location][demo_name],
                    group_sizes[location][demo_name],
                    labels,
                )
            )
            label_chunks = split_labels(labels, split_parts)
            if not label_chunks:
                label_chunks = [[]]
            overview_rows.append((demo_name, label_chunks))
            max_parts = max(max_parts, len(label_chunks))

        overview_figure, overview_axes = plt.subplots(
            len(demo_columns),
            max_parts,
            figsize=(max(7 * max_parts, 14), 5.8 * len(demo_columns)),
        )
        overview_axes = np.atleast_2d(overview_axes)
        if overview_axes.shape[0] != len(demo_columns):
            overview_axes = overview_axes.reshape(len(demo_columns), max_parts)

        for row_idx, (demo_name, label_chunks) in enumerate(overview_rows):
            for col_idx in range(max_parts):
                axis = overview_axes[row_idx, col_idx]
                if col_idx >= len(label_chunks):
                    axis.set_axis_off()
                    continue
                labels = label_chunks[col_idx]
                part_label = f"part {col_idx + 1} / {len(label_chunks)}"
                plot_location_demo(
                    axis,
                    location,
                    demo_name,
                    grouped[location][demo_name],
                    group_sizes[location][demo_name],
                    labels,
                    part_label=part_label,
                )

                single_figure, single_axis = plt.subplots(figsize=(max(12, 0.7 * max(len(labels), 1) + 4), 8))
                plot_location_demo(
                    single_axis,
                    location,
                    demo_name,
                    grouped[location][demo_name],
                    group_sizes[location][demo_name],
                    labels,
                    part_label=part_label,
                )
                single_figure.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.34)
                single_path = location_folder / f"{slugify_filename(demo_name)}_part_{col_idx + 1}.png"
                single_figure.savefig(single_path, dpi=280)
                plt.close(single_figure)
                saved_paths.append(single_path)

        overview_figure.suptitle(
            f"{location}\nGender and age come from ground_truth.json; race defaults to original-image perceived_race_ethnicity.",
            fontsize=18,
        )
        overview_figure.subplots_adjust(left=0.04, right=0.995, top=0.93, bottom=0.05, hspace=1.0, wspace=0.45)
        overview_path = location_folder / "overview.png"
        overview_figure.savefig(overview_path, dpi=260)
        plt.close(overview_figure)
        saved_paths.append(overview_path)
        location_summaries.append({
            "location": location,
            "folder": str(location_folder),
            "overview": str(overview_path),
            "n_files": 1 + sum(len(chunks) for _, chunks in overview_rows),
        })

    return saved_paths, location_summaries, significance_rows


def write_significance_csv(rows, output_csv: Path):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sort_significance_rows(rows)

    fieldnames = [
        "location",
        "demographic",
        "group_a",
        "group_b",
        "label",
        "count_a",
        "n_a",
        "pct_a",
        "count_b",
        "n_b",
        "pct_b",
        "difference_pp",
        "abs_difference",
        "p_value",
        "significance_score",
        "significant_0_05",
        "used_top_two_groups",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted_rows:
            writer.writerow(row)

    return sorted_rows


def main():
    args = parse_args()
    location_label_universe = load_location_label_universe(args.taxonomy_file)
    records, missing = load_records(
        args.images_root,
        args.location_annotations_file,
        args.ground_truth_file,
        args.cross_image_file,
        args.race_source,
        limit=args.limit,
    )
    if not records:
        raise SystemExit("No usable location attribute records found.")

    grouped, group_sizes = build_location_stats(records)
    args.panels_dir.mkdir(parents=True, exist_ok=True)
    panel_paths, location_summaries, significance_rows = save_location_split_figures(
        grouped,
        group_sizes,
        location_label_universe,
        args.panels_dir,
        args.top_n_labels,
        args.split_parts,
    )
    sorted_significance_rows = write_significance_csv(significance_rows, args.output_significance_csv)
    stats_payload = build_condensed_significance_payload(
        sorted_significance_rows,
        race_source=args.race_source,
        ground_truth_file=args.ground_truth_file,
        taxonomy_file=args.taxonomy_file,
        missing=missing,
        records=len(records),
        locations=sorted(grouped.keys()),
        alpha=args.alpha,
        min_group_size=args.min_group_size,
        output_significance_csv=args.output_significance_csv,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(stats_payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    overview_payload = {
        "message": "Per-location split graphs are saved under panels_dir, one folder per location.",
        "race_source": args.race_source,
        "ground_truth_source": args.ground_truth_file,
        "split_parts": args.split_parts,
        "output_significance_csv": str(args.output_significance_csv),
        "locations": location_summaries,
    }
    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    with args.output_figure.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(overview_payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"Loaded {len(records)} location-annotation rows across {len(grouped)} locations.")
    print(f"Race source: {args.race_source}")
    print(f"Saved statistics JSON to {args.output_json}")
    print(f"Saved per-location split graphs under {args.panels_dir}")
    print(f"Saved location overview manifest to {args.output_figure.with_suffix('.json')}")
    print(f"Saved {len(panel_paths)} PNG files across location folders")
    print(f"Saved significance CSV to {args.output_significance_csv} ({len(sorted_significance_rows)} rows)")

    if args.show:
        print("Use the per-location overview PNG files in panels_dir for interactive viewing.")


if __name__ == "__main__":
    main()
