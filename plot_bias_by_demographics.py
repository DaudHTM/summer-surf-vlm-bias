import argparse
import json
import re
from collections import Counter
from pathlib import Path


ID_PATTERN = re.compile(r"(sa_\d+)")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create 3 grouped bar charts comparing generated labels against "
            "demographic groups (age, gender, skin tone)."
        )
    )
    parser.add_argument(
        "--selected-file",
        type=Path,
        default=Path("selected_images.txt"),
        help="Path to selected_images.txt.",
    )
    parser.add_argument(
        "--annotations-file",
        type=Path,
        default=Path("religious_sites_annotations.json"),
        help="Path to per-image generated label annotations JSON.",
    )
    parser.add_argument(
        "--taxonomy-file",
        type=Path,
        default=Path("location_attribute_labels.json"),
        help=(
            "Optional taxonomy JSON; when present, include all taxonomy labels "
            "(even if rare/zero) in the charts."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("characteristic_graphs"),
        help="Directory where output charts are saved.",
    )
    parser.add_argument(
        "--top-n-labels",
        type=int,
        default=0,
        help=(
            "How many labels to include per chart (sorted by overall frequency). "
            "Use 0 to include all labels."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display charts after saving.",
    )
    return parser.parse_args()


def extract_image_id(text: str):
    match = ID_PATTERN.search(text)
    return match.group(1) if match else None


def parse_combo_desc(combo_desc: str):
    result = {}
    for raw_item in combo_desc.split(","):
        item = raw_item.strip()
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            result[key] = value
    return result


def load_demographics(selected_file: Path):
    demographics_by_id = {}
    with open(selected_file, "r", encoding="utf-8") as handle:
        for line in handle:
            row = line.strip()
            if not row:
                continue

            parts = row.split("|", 3)
            if len(parts) < 4:
                continue

            image_name = parts[0].strip()
            combo_desc = parts[3].strip()
            image_id = extract_image_id(image_name)
            if not image_id:
                continue

            combo = parse_combo_desc(combo_desc)
            gender = combo.get("gender")
            age = combo.get("age")
            skin_tone = combo.get("skin_tone")
            if not (gender and age and skin_tone):
                continue

            record = {
                "gender": gender,
                "age": age,
                "skin_tone": skin_tone,
            }

            if image_id in demographics_by_id and demographics_by_id[image_id] != record:
                # Keep first seen value to avoid unstable merges from noisy duplicates.
                continue

            demographics_by_id[image_id] = record

    return demographics_by_id


def load_generated_labels(annotations_file: Path):
    with open(annotations_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    images = data.get("images", {})
    labels_by_id = {}
    for image_name, attrs in images.items():
        image_id = extract_image_id(image_name)
        if not image_id:
            continue

        labels = set()
        if isinstance(attrs, dict):
            for attribute, value in attrs.items():
                if not isinstance(value, list):
                    continue
                for label in value:
                    if not isinstance(label, str):
                        continue
                    label_text = label.strip()
                    if not label_text:
                        continue
                    labels.add(f"{attribute}: {label_text}")

        labels_by_id[image_id] = labels

    return labels_by_id


def load_taxonomy_labels(taxonomy_file: Path):
    if not taxonomy_file.exists():
        return None

    with open(taxonomy_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    labels_by_attribute = data.get("labels_by_attribute")
    if not isinstance(labels_by_attribute, dict):
        return None

    taxonomy_labels = []
    seen = set()
    for attribute, labels in labels_by_attribute.items():
        if not isinstance(labels, list):
            continue
        for label in labels:
            if not isinstance(label, str):
                continue
            label_text = label.strip()
            if not label_text:
                continue
            key = f"{attribute}: {label_text}"
            if key in seen:
                continue
            seen.add(key)
            taxonomy_labels.append(key)

    return taxonomy_labels if taxonomy_labels else None


def pick_two_values(values_counter: Counter, preferred_a: str, preferred_b: str):
    values_lower = {k.lower(): k for k in values_counter}
    if preferred_a.lower() in values_lower and preferred_b.lower() in values_lower:
        return values_lower[preferred_a.lower()], values_lower[preferred_b.lower()]

    top_two = [name for name, _ in values_counter.most_common(2)]
    if len(top_two) < 2:
        return None
    return top_two[0], top_two[1]


def compute_label_rates(
    matched_rows,
    group_key: str,
    group_a: str,
    group_b: str,
    top_n_labels: int,
    taxonomy_labels,
):
    # Build a deterministic label universe from matched images.
    label_counts = Counter()
    for row in matched_rows:
        label_counts.update(row["labels"])

    if taxonomy_labels:
        labels = list(taxonomy_labels)
    elif top_n_labels and top_n_labels > 0:
        labels = [name for name, _ in label_counts.most_common(top_n_labels)]
    else:
        labels = [name for name, _ in label_counts.most_common()]

    subset_a = [row for row in matched_rows if row[group_key].lower() == group_a.lower()]
    subset_b = [row for row in matched_rows if row[group_key].lower() == group_b.lower()]

    denom_a = len(subset_a)
    denom_b = len(subset_b)
    if denom_a == 0 or denom_b == 0:
        return None

    rates_a = []
    rates_b = []
    for label in labels:
        count_a = sum(1 for row in subset_a if label in row["labels"])
        count_b = sum(1 for row in subset_b if label in row["labels"])
        rates_a.append(100.0 * count_a / denom_a)
        rates_b.append(100.0 * count_b / denom_b)

    return {
        "labels": labels,
        "rates_a": rates_a,
        "rates_b": rates_b,
        "denom_a": denom_a,
        "denom_b": denom_b,
    }


def plot_grouped_chart(metric_name: str, group_a: str, group_b: str, payload, output_path: Path):
    import matplotlib.pyplot as plt
    import numpy as np

    labels = payload["labels"]
    x_positions = np.arange(len(labels))
    bar_width = 0.42

    fig_width = max(12, len(labels) * 0.7)
    figure, axis = plt.subplots(figsize=(fig_width, 6))

    bars_a = axis.bar(
        x_positions - bar_width / 2,
        payload["rates_a"],
        width=bar_width,
        color="#4C78A8",
        label=f"{group_a} (n={payload['denom_a']})",
    )
    bars_b = axis.bar(
        x_positions + bar_width / 2,
        payload["rates_b"],
        width=bar_width,
        color="#F58518",
        label=f"{group_b} (n={payload['denom_b']})",
    )

    axis.set_title(f"{metric_name} vs generated labels")
    axis.set_ylabel("Label presence rate (%)")
    axis.set_ylim(0, 100)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(labels, rotation=70, ha="right")
    axis.legend()

    for bars in (bars_a, bars_b):
        for bar in bars:
            y = bar.get_height()
            if y <= 0:
                continue
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                y + 0.8,
                f"{y:.0f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    return figure


def main():
    args = parse_args()

    if not args.selected_file.exists():
        raise SystemExit(f"selected file not found: {args.selected_file}")
    if not args.annotations_file.exists():
        raise SystemExit(f"annotations file not found: {args.annotations_file}")

    demographics_by_id = load_demographics(args.selected_file)
    labels_by_id = load_generated_labels(args.annotations_file)
    taxonomy_labels = load_taxonomy_labels(args.taxonomy_file)

    matched_ids = sorted(set(demographics_by_id) & set(labels_by_id))
    if not matched_ids:
        raise SystemExit(
            "No overlapping image IDs between demographics and annotations. "
            "Check selected/annotation inputs."
        )

    matched_rows = []
    for image_id in matched_ids:
        row = {
            **demographics_by_id[image_id],
            "labels": labels_by_id[image_id],
        }
        matched_rows.append(row)

    gender_counter = Counter(row["gender"] for row in matched_rows)
    age_counter = Counter(row["age"] for row in matched_rows)
    skin_counter = Counter(row["skin_tone"] for row in matched_rows)

    gender_pair = pick_two_values(gender_counter, "male", "female")
    age_pair = pick_two_values(age_counter, "young", "old")
    skin_pair = pick_two_values(skin_counter, "dark", "light")

    if not gender_pair:
        raise SystemExit("Not enough gender values in matched data to build a 2-bar chart.")
    if not age_pair:
        raise SystemExit("Not enough age values in matched data to build a 2-bar chart.")
    if not skin_pair:
        raise SystemExit("Not enough skin_tone values in matched data to build a 2-bar chart.")

    payload_gender = compute_label_rates(
        matched_rows,
        group_key="gender",
        group_a=gender_pair[0],
        group_b=gender_pair[1],
        top_n_labels=args.top_n_labels,
        taxonomy_labels=taxonomy_labels,
    )
    payload_age = compute_label_rates(
        matched_rows,
        group_key="age",
        group_a=age_pair[0],
        group_b=age_pair[1],
        top_n_labels=args.top_n_labels,
        taxonomy_labels=taxonomy_labels,
    )
    payload_skin = compute_label_rates(
        matched_rows,
        group_key="skin_tone",
        group_a=skin_pair[0],
        group_b=skin_pair[1],
        top_n_labels=args.top_n_labels,
        taxonomy_labels=taxonomy_labels,
    )

    if payload_gender is None or payload_age is None or payload_skin is None:
        raise SystemExit("At least one chart could not be built due to empty subgroup(s).")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    gender_path = args.output_dir / "bias_gender.png"
    age_path = args.output_dir / "bias_age.png"
    skin_path = args.output_dir / "bias_skin_tone.png"

    figures = []
    figures.append(
        plot_grouped_chart("Gender", gender_pair[0], gender_pair[1], payload_gender, gender_path)
    )
    figures.append(
        plot_grouped_chart("Age", age_pair[0], age_pair[1], payload_age, age_path)
    )
    figures.append(
        plot_grouped_chart("Skin Tone (Race Proxy)", skin_pair[0], skin_pair[1], payload_skin, skin_path)
    )

    print(f"Matched images: {len(matched_rows)}")
    print(f"Saved: {gender_path}")
    print(f"Saved: {age_path}")
    print(f"Saved: {skin_path}")

    if args.show:
        import matplotlib.pyplot as plt

        plt.show()
    else:
        import matplotlib.pyplot as plt

        for fig in figures:
            plt.close(fig)


if __name__ == "__main__":
    main()
