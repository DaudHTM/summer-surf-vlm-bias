import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


DEMOGRAPHICS = ("race", "gender", "age")
SLUG_RE = re.compile(r"[^a-z0-9]+")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create 3 demographic difference graphs (race, gender, age) from "
            "location_attribute_demographic_statistics.json using exactly 5 labels per graph."
        )
    )
    parser.add_argument(
        "--stats-file",
        type=Path,
        default=Path("location_attribute_demographic_statistics.json"),
        help="Path to location_attribute_demographic_statistics.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("location_attribute_demographic_panels/selected_differences"),
        help="Directory where the 3 graphs are saved.",
    )
    parser.add_argument(
        "--race-labels",
        nargs="+",
        default=None,
        help="Exactly 5 labels for the race graph.",
    )
    parser.add_argument(
        "--gender-labels",
        nargs="+",
        default=None,
        help="Exactly 5 labels for the gender graph.",
    )
    parser.add_argument(
        "--age-labels",
        nargs="+",
        default=None,
        help="Exactly 5 labels for the age graph.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for labels when they are not provided via CLI.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures after saving.",
    )
    return parser.parse_args()


def slugify(text: str) -> str:
    lowered = text.strip().lower()
    lowered = SLUG_RE.sub("_", lowered).strip("_")
    return lowered or "plot"


def load_rows(stats_file: Path):
    with stats_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("results", [])
    if not isinstance(rows, list):
        raise ValueError("Invalid stats file: 'results' must be a list.")
    return rows


def aggregate_by_label(rows, demographic: str):
    grouped = defaultdict(list)
    for row in rows:
        if row.get("demographic") != demographic:
            continue
        label = row.get("label")
        if not isinstance(label, str) or not label.strip():
            continue
        grouped[label.strip()].append(row)

    if not grouped:
        return {}

    aggregated = {}
    for label, label_rows in grouped.items():
        total_weight = 0.0
        weighted_diff_sum = 0.0
        weighted_pct_a_sum = 0.0
        weighted_pct_b_sum = 0.0
        significance_max = None

        group_a = str(label_rows[0].get("group_a", "group_a"))
        group_b = str(label_rows[0].get("group_b", "group_b"))

        for item in label_rows:
            n_a = float(item.get("n_a", 0) or 0)
            n_b = float(item.get("n_b", 0) or 0)
            weight = max(1.0, n_a + n_b)

            diff = float(item.get("difference_pp", 0.0) or 0.0)
            pct_a = float(item.get("pct_a", 0.0) or 0.0)
            pct_b = float(item.get("pct_b", 0.0) or 0.0)

            weighted_diff_sum += diff * weight
            weighted_pct_a_sum += pct_a * weight
            weighted_pct_b_sum += pct_b * weight
            total_weight += weight

            score = item.get("significance_score")
            if isinstance(score, (int, float)):
                significance_max = score if significance_max is None else max(significance_max, score)

        aggregated[label] = {
            "group_a": group_a,
            "group_b": group_b,
            "difference_pp": weighted_diff_sum / total_weight,
            "pct_a": weighted_pct_a_sum / total_weight,
            "pct_b": weighted_pct_b_sum / total_weight,
            "significance_max": significance_max if significance_max is not None else 0.0,
            "n_rows": len(label_rows),
        }

    return aggregated


def label_ranking(aggregated):
    return sorted(
        aggregated.items(),
        key=lambda item: (
            -(item[1].get("significance_max", 0.0) or 0.0),
            -abs(item[1].get("difference_pp", 0.0) or 0.0),
            item[0].lower(),
        ),
    )


def parse_label_input(raw: str, ranked_pairs):
    by_index = {str(i + 1): label for i, (label, _) in enumerate(ranked_pairs)}
    chosen = []
    for token in [part.strip() for part in raw.split(",") if part.strip()]:
        if token in by_index:
            chosen.append(by_index[token])
        else:
            chosen.append(token)
    deduped = []
    seen = set()
    for label in chosen:
        key = label.lower()
        if key in seen:
            continue
        deduped.append(label)
        seen.add(key)
    return deduped


def resolve_five_labels(demographic: str, aggregated, provided_labels, interactive: bool):
    ranked_pairs = label_ranking(aggregated)
    available_labels = {label.lower(): label for label, _ in ranked_pairs}

    if provided_labels:
        normalized = []
        for label in provided_labels:
            key = label.strip().lower()
            if key not in available_labels:
                raise ValueError(
                    f"Unknown {demographic} label '{label}'. Use only labels present in the statistics file."
                )
            normalized.append(available_labels[key])
        if len(normalized) != 5:
            raise ValueError(f"{demographic}: provide exactly 5 labels (got {len(normalized)}).")
        if len({x.lower() for x in normalized}) != 5:
            raise ValueError(f"{demographic}: labels must be unique.")
        return normalized

    if interactive:
        print(f"\n[{demographic}] choose exactly 5 labels.")
        print("Type comma-separated indices (e.g., 1,4,7,10,13) or exact label names.")
        preview = ranked_pairs[:40]
        for i, (label, info) in enumerate(preview, start=1):
            print(
                f"{i:>2}. {label} | diff={info['difference_pp']:+.2f} pp | "
                f"max_significance={info['significance_max']:.2f}"
            )
        while True:
            raw = input("Enter 5 labels: ").strip()
            picks = parse_label_input(raw, preview)
            if len(picks) != 5:
                print(f"Need exactly 5 labels, got {len(picks)}. Try again.")
                continue
            invalid = [label for label in picks if label.lower() not in available_labels]
            if invalid:
                print(f"Unknown labels: {invalid}. Try again.")
                continue
            return [available_labels[label.lower()] for label in picks]

    # Fallback for non-interactive use: top 5 labels by ranking.
    return [label for label, _ in ranked_pairs[:5]]


def plot_demographic_difference(demographic: str, aggregated, selected_labels, output_path: Path):
    selected_payload = [aggregated[label] for label in selected_labels]
    differences = [item["difference_pp"] for item in selected_payload]
    group_a = selected_payload[0]["group_a"]
    group_b = selected_payload[0]["group_b"]

    colors = ["#1f77b4" if value >= 0 else "#d62728" for value in differences]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(selected_labels, differences, color=colors)
    ax.axvline(0, color="#444444", linewidth=1.0)

    ax.set_title(
        f"{demographic.title()} difference by label ({group_a} vs {group_b})",
        fontsize=13,
        pad=10,
    )
    ax.set_xlabel(f"Difference (percentage points): % {group_a} - % {group_b}")
    ax.set_ylabel("Label")

    max_abs = max(abs(value) for value in differences) if differences else 1.0
    pad = max(3.0, max_abs * 0.2)
    ax.set_xlim(-max_abs - pad, max_abs + pad)

    for bar, value in zip(bars, differences):
        y_mid = bar.get_y() + bar.get_height() / 2
        x_pos = value + (0.4 if value >= 0 else -0.4)
        align = "left" if value >= 0 else "right"
        ax.text(x_pos, y_mid, f"{value:+.2f}", va="center", ha=align, fontsize=9)

    subtitle = "Blue: group_a higher | Red: group_b higher"
    ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color="#444444")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    return fig


def main():
    args = parse_args()
    rows = load_rows(args.stats_file)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    provided = {
        "race": args.race_labels,
        "gender": args.gender_labels,
        "age": args.age_labels,
    }

    saved_paths = []
    figures = []

    for demographic in DEMOGRAPHICS:
        aggregated = aggregate_by_label(rows, demographic)
        if not aggregated:
            raise ValueError(f"No rows found for demographic '{demographic}'.")

        selected_labels = resolve_five_labels(
            demographic=demographic,
            aggregated=aggregated,
            provided_labels=provided[demographic],
            interactive=args.interactive,
        )

        output_path = args.output_dir / f"{slugify(demographic)}_difference_5_labels.png"
        fig = plot_demographic_difference(demographic, aggregated, selected_labels, output_path)
        figures.append(fig)
        saved_paths.append(output_path)

        print(f"Saved {demographic} graph -> {output_path}")
        print(f"Selected labels ({demographic}): {selected_labels}")

    if args.show:
        plt.show()
    else:
        for fig in figures:
            plt.close(fig)

    print("\nCreated 3 demographic difference graphs.")


if __name__ == "__main__":
    main()
