"""
FACET dataset sampler.

For each attribute (gender, hair_color, age, occupation, skin_tone), pick the
SPECIFIC labels you care about. Leave a list empty to ignore that attribute
entirely (no filtering, no splitting on it).

The script takes the cartesian product of all NON-EMPTY attribute selections
and balances the total evenly across those resulting groups.

Only (filename, person_id) pairs that have a corresponding mask in
coco_masks.json (matched via facet_person_id) are eligible for selection.

Examples:
    SELECTED = {
        "gender": ["male"],
        "skin_tone": ["dark", "white"],
        "hair_color": [],
        "age": [],
        "occupation": [],
    }
    -> 2 groups: (male, dark) and (male, white), each ~TOTAL_IMAGES/2
       hair_color/age/occupation are ignored entirely (any value allowed).

    SELECTED = {
        "gender": ["male", "female"],
        "skin_tone": ["dark", "light"],
        "hair_color": [],
        "age": [],
        "occupation": [],
    }
    -> 4 groups: (male,dark), (male,light), (female,dark), (female,light),
       each ~TOTAL_IMAGES/4.

FACET annotations.csv reference columns:
    gender_presentation_masc, gender_presentation_fem
    hair_color_black, hair_color_grey, hair_color_blonde, hair_color_brown, hair_color_red, hair_color_colored
    age_presentation_young, age_presentation_middle, age_presentation_older
    class1 (primary occupation/category label, e.g. "guitarist", "doctor")
    skin_tone_1 .. skin_tone_10 (Monk Skin Tone scale; each is a sum across 3
        annotators, 0-3, so a person can have multiple non-zero values)

Output file (selected_images.txt) format, one row per selected image:
    filename|person_id|segmentation_rle_json|combo_desc
where segmentation_rle_json is the JSON-encoded COCO RLE segmentation dict
for that person's mask (counts kept as the original string). A pipe '|' is
used as the field separator since the RLE JSON itself contains commas.
"""

import csv
import os
from pathlib import Path
import json
import random
from itertools import product
from collections import defaultdict

# ----------------------- CONFIG -----------------------
FACET_ROOT = Path(os.getenv("FACET_ROOT", "data/facet"))
INPUT_CSV = FACET_ROOT / "annotations" / "annotations.csv"
MASKS_JSON = FACET_ROOT / "annotations" / "coco_masks.json"
OUTPUT_FILE = "selected_images.txt"
TOTAL_IMAGES = 1000

# Keep only masks whose COCO annotation area is within this range.
# Set either bound to None to disable that side of the filter.
MASK_AREA_MIN = 150000
MASK_AREA_MAX = None

# Keep only full-body person masks from COCO annotations.
# FACET uses category_id 53 for whole-person masks; other nearby IDs can be
# partial/clothing regions.
FULL_BODY_CATEGORY_IDS = {53}

# For each attribute, list ONLY the labels you want to include.
# Leave a list empty ([]) to ignore that attribute entirely.
SELECTED = {
    "gender": ["male" , "female"],
    "hair_color": [],
    "age": ["young", "old"],
    "occupation": [],
    "skin_tone": ["dark", "light"],
}

RANDOM_SEED = 42

# --- Skin tone bucketing (Monk Skin Tone scale, 1-10) ---
SKIN_TONE_BUCKETS = {
    "light": [1, 2, 3],
    "medium": [4, 5, 6, 7],
    "dark": [8, 9, 10],
}

# --- Attribute definitions ---
# For one-hot attributes: {label: column_name}
# For "occupation", values are read directly from the 'class1' column.
# For "skin_tone", labels come from SKIN_TONE_BUCKETS above.
ATTRIBUTE_DEFINITIONS = {
    "gender": {
        "male": "gender_presentation_masc",
        "female": "gender_presentation_fem",
    },
    "hair_color": {
        "black": "hair_color_black",
        "white": "hair_color_grey",
        "blonde": "hair_color_blonde",
        "brown": "hair_color_brown",
        "red": "hair_color_red",
        "colored": "hair_color_colored",
    },
    "age": {
        "young": "age_presentation_young",
        "middle": "age_presentation_middle",
        "old": "age_presentation_older",
    },
    "occupation": {
        # label -> exact value expected in the 'class1' column
        "guitarist": "guitarist",
        "doctor": "doctor",
        # add more occupations here as needed; label can match SELECTED values directly
    },
    "skin_tone": SKIN_TONE_BUCKETS,  # special-cased, see get_matches()
}
# --------------------------------------------------------


def is_true(value):
    """FACET binary fields may be '1', '0', 'True', 'False', etc."""
    return str(value).strip().lower() in ("1", "true", "yes")


def get_matches(row, category, label_map):
    """Return list of ALL labels (from label_map) this row matches for a category."""
    if category == "occupation":
        class1 = row.get("class1", "").strip().lower()
        return [label for label, value in label_map.items()
                if class1 == value.strip().lower()]

    elif category == "skin_tone":
        matched = []
        for bucket_label, mst_values in label_map.items():
            for mst in mst_values:
                col = f"skin_tone_{mst}"
                if col in row:
                    try:
                        val = float(row[col])
                    except (ValueError, TypeError):
                        val = 0
                    if val > 0:
                        matched.append(bucket_label)
                        break
        return matched

    else:
        return [label for label, col in label_map.items()
                if col in row and is_true(row[col])]


def build_person_mask_lookup(masks_json_path):
    """
    Build person_id -> (segmentation, area) lookup.

    Keep only full-body person masks (category_id in FULL_BODY_CATEGORY_IDS).
    If multiple full-body masks exist, keep the largest by area.
    """
    with open(masks_json_path, encoding="utf-8") as f:
        coco_masks = json.load(f)

    person_segments = {}
    for ann in coco_masks.get("annotations", []):
        seg = ann.get("segmentation")
        if not seg:
            continue

        person_id = str(ann["facet_person_id"])
        cid = ann.get("category_id")
        if cid not in FULL_BODY_CATEGORY_IDS:
            continue

        area = ann.get("area", 0)
        if person_id not in person_segments:
            person_segments[person_id] = (cid, seg, area)
            continue

        prev_cid, prev_seg, prev_area = person_segments[person_id]
        if area > prev_area:
            person_segments[person_id] = (cid, seg, area)

    return {pid: {"segmentation": seg, "area": area}
            for pid, (cid, seg, area) in person_segments.items()}


def load_images(csv_path, active_attrs, ids_with_masks):
    """
    Group unique (filename, person_id) pairs by their combination of active
    attribute labels (only attributes with a non-empty SELECTED list).

    A row qualifies for a combo if, for every active attribute, it matches
    at least one of the SELECTED labels for that attribute, AND its
    person_id has a corresponding mask in coco_masks.json.
    """
    groups = defaultdict(set)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get("filename")
            person_id = row.get("person_id", "")
            if not filename:
                continue

            # Skip rows whose person_id has no mask available
            if person_id not in ids_with_masks:
                continue

            matches_per_category = {}
            skip_row = False
            for category, selected_labels in active_attrs.items():
                label_map = {
                    label: col for label, col in ATTRIBUTE_DEFINITIONS[category].items()
                    if label in selected_labels
                } if category != "skin_tone" else {
                    label: vals for label, vals in ATTRIBUTE_DEFINITIONS[category].items()
                    if label in selected_labels
                }

                matched = get_matches(row, category, label_map)
                # restrict to only the selected labels for this category
                matched = [m for m in matched if m in selected_labels]

                if not matched:
                    skip_row = True
                    break
                matches_per_category[category] = matched

            if skip_row:
                continue

            ordered_matches = [matches_per_category[c] for c in active_attrs]
            for combo in product(*ordered_matches):
                groups[combo].add((filename, person_id))

    # Collapse to: combo -> {filename: [person_id, ...]}
    collapsed = {}
    for combo, pairs in groups.items():
        per_file = defaultdict(list)
        for filename, person_id in pairs:
            per_file[filename].append(person_id)
        collapsed[combo] = per_file

    return collapsed


def even_split(total, n_groups):
    base = total // n_groups
    remainder = total % n_groups
    return [base + (1 if i < remainder else 0) for i in range(n_groups)]


def main():
    random.seed(RANDOM_SEED)

    # Only keep attributes where the user actually selected at least one label
    active_attrs = {cat: labels for cat, labels in SELECTED.items() if labels}

    if not active_attrs:
        print("Error: no attributes selected. Pick at least one label in SELECTED.")
        return

    invalid_cats = [c for c in active_attrs if c not in ATTRIBUTE_DEFINITIONS]
    if invalid_cats:
        print(f"Error: unknown attribute(s) {invalid_cats}. "
              f"Valid options: {list(ATTRIBUTE_DEFINITIONS.keys())}")
        return

    # Validate selected labels exist for each active attribute
    for cat, labels in active_attrs.items():
        valid_labels = set(ATTRIBUTE_DEFINITIONS[cat].keys())
        bad = [l for l in labels if l not in valid_labels]
        if bad:
            print(f"Error: invalid label(s) {bad} for attribute '{cat}'. "
                  f"Valid options: {sorted(valid_labels)}")
            return

    print("Loading mask index...")
    person_masks = build_person_mask_lookup(MASKS_JSON)
    print(
        f"Found {len(person_masks)} person_id(s) with full-body masks "
        f"(category_id in {sorted(FULL_BODY_CATEGORY_IDS)})."
    )

    def area_ok(mask_area):
        if MASK_AREA_MIN is not None and mask_area < MASK_AREA_MIN:
            return False
        if MASK_AREA_MAX is not None and mask_area > MASK_AREA_MAX:
            return False
        return True

    eligible_person_masks = {
        pid: data for pid, data in person_masks.items() if area_ok(data["area"])
    }
    ids_with_masks = set(eligible_person_masks.keys())

    print(
        "After mask area filter "
        f"(min={MASK_AREA_MIN}, max={MASK_AREA_MAX}), "
        f"{len(ids_with_masks)} person_id(s) remain eligible."
    )

    groups = load_images(INPUT_CSV, active_attrs, ids_with_masks)

    # All combinations of the SELECTED labels across active attributes
    label_lists = [active_attrs[c] for c in active_attrs]
    combos = list(product(*label_lists))
    targets = even_split(TOTAL_IMAGES, len(combos))

    person_id_to_seg = {
        pid: data["segmentation"] for pid, data in eligible_person_masks.items()
    }

    selected = []
    for combo, target in zip(combos, targets):
        per_file = groups.get(combo, {})
        filenames = list(per_file.keys())
        random.shuffle(filenames)
        chosen = filenames[:target]

        combo_desc = ", ".join(f"{attr}={val}" for attr, val in zip(active_attrs.keys(), combo))
        if len(chosen) < target:
            print(f"Warning: only {len(chosen)}/{target} images available for ({combo_desc})")

        for name in chosen:
            person_id = per_file[name][0]  # only one person per image
            seg = person_id_to_seg.get(person_id)
            seg_json = json.dumps(seg) if seg is not None else "{}"
            selected.append((name, person_id, seg_json, combo_desc))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for name, person_id, seg_json, combo_desc in selected:
            f.write(f"{name}|{person_id}|{seg_json}|{combo_desc}\n")

    print(f"Active attributes: {active_attrs}")
    print(f"Number of groups: {len(combos)}")
    print(f"Selected {len(selected)} images (target was {TOTAL_IMAGES}).")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
