import argparse
import os
import ast
import json
import re
from pathlib import Path

import requests

DEFAULT_LOCATIONS = [
    "shopping mall",
    "public park",
    "restaurant",
    "sports stadium",
    "religious site",
    "cultural site",
    "office building",
    "public transport",
    "a street with people, cars, buses, bicycles, etc.",
]
DEFAULT_MODEL = "qwen3-vl:8b-instruct"
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MAX_RETRIES = 2
DEFAULT_OUTPUT_FILE = Path("location_attribute_labels.json")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate attribute categories and labels for predefined locations using Ollama."
        )
    )
    parser.add_argument(
        "--n-attrs",
        type=int,
        required=True,
        help="Number of attribute categories to generate.",
    )
    parser.add_argument(
        "--n-labels",
        type=int,
        required=True,
        help="Number of labels/examples to generate per attribute category.",
    )
    return parser.parse_args()


def check_ollama(ollama_url: str) -> bool:
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception as exc:
        print(f"Error connecting to Ollama at {ollama_url}: {exc}")
        return False


def chat_ollama(ollama_url: str, model: str, messages: list[dict]) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        # Nudge Ollama-compatible models to actually emit JSON when possible.
        # Harmless no-op for models/backends that ignore unknown fields.
        "format": "json",
    }
    response = requests.post(f"{ollama_url}/api/chat", json=payload, timeout=180)
    response.raise_for_status()
    data = response.json()
    return data.get("message", {}).get("content", "").strip()


def _is_malformed_item(item: str) -> bool:
    """
    Catches JSON-residue artifacts that leak through when a model's JSON gets
    mangled and a fallback parser splits it line-by-line, e.g. items like
    '"label one": "Roman Mosaic"' or bare '{' / '}' braces.
    """
    s = item.strip()
    if not s or s in ("{", "}", "[", "]"):
        return True
    if re.match(r'^"?label[\s_]*\w*"?\s*:', s, re.IGNORECASE):
        return True
    if re.match(r'^(label|item|attribute|category|key)\s+\w{1,12}$', s, re.IGNORECASE):
        return True
    if re.match(r'^(label|item|attribute|category|key)[\s_-]?\d+$', s, re.IGNORECASE):
        return True
    if '":' in s or "':" in s:
        return True
    return False


def _normalize_list(values: list, max_items: int, reject_malformed: bool = False) -> list[str]:
    seen = set()
    normalized = []
    for value in values:
        item = str(value).strip()
        if not item:
            continue
        if reject_malformed and _is_malformed_item(item):
            continue
        if item not in seen:
            seen.add(item)
            normalized.append(item)
        if len(normalized) >= max_items:
            break
    return normalized


def parse_python_list(text: str, max_items: int) -> list[str]:
    # First try direct literal parsing.
    try:
        data = ast.literal_eval(text)
        if isinstance(data, list):
            return _normalize_list(data, max_items, reject_malformed=True)
    except Exception:
        pass

    # Then try extracting one list block from free-form text.
    candidates = re.findall(r"\[[\s\S]*?\]", text)
    for candidate in candidates:
        try:
            data = ast.literal_eval(candidate)
            if isinstance(data, list):
                return _normalize_list(data, max_items, reject_malformed=True)
        except Exception:
            continue

    # Last-resort fallback: pull out quoted strings if any exist (much safer
    # than raw line/comma splitting, which mangles broken JSON into garbage
    # like '"label one": "Roman Mosaic"').
    quoted = re.findall(r'"([^"\n]{1,80})"', text)
    quoted = [q for q in quoted if not _is_malformed_item(q)]
    if quoted:
        return _normalize_list(quoted, max_items, reject_malformed=True)

    # Absolute last resort: split by lines/commas.
    rough = []
    for line in text.splitlines():
        cleaned = line.strip(" -\t\n\r")
        if cleaned:
            rough.extend([p.strip() for p in cleaned.split(",") if p.strip()])
    return _normalize_list(rough, max_items, reject_malformed=True)


def parse_json_list(raw: str, key: str, max_items: int) -> list[str]:
    """
    Parse {"<key>": [...]} out of model output, tolerating markdown fences and
    stray prose. Falls back to the legacy Python-list parser if JSON parsing
    fails outright, so older/smaller models that ignore the JSON instruction
    still produce usable output.
    """
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE).strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and key in data:
            value = data[key]
            if isinstance(value, list):
                return _normalize_list(value, max_items, reject_malformed=True)
            if isinstance(value, dict):
                # Model emitted {"label one": "Roman Mosaic", ...} instead of
                # a list — recover the values rather than discarding them.
                return _normalize_list(list(value.values()), max_items, reject_malformed=True)
        if isinstance(data, list):
            return _normalize_list(data, max_items, reject_malformed=True)
    except Exception:
        pass

    # Try to find a {...} block containing the key.
    for candidate in re.findall(r"\{[\s\S]*?\}", cleaned):
        try:
            data = json.loads(candidate)
            if isinstance(data, dict) and key in data:
                value = data[key]
                if isinstance(value, list):
                    return _normalize_list(value, max_items, reject_malformed=True)
                if isinstance(value, dict):
                    return _normalize_list(
                        list(value.values()), max_items, reject_malformed=True
                    )
        except Exception:
            continue

    # Fall back to the old bracket/line-based parser.
    return parse_python_list(cleaned, max_items)


def _generate_with_retry(
    build_messages_fn,
    key: str,
    expected_n: int,
    ollama_url: str,
    model: str,
    max_retries: int,
):
    """
    Calls chat_ollama, parses JSON, and validates the item count. Retries with
    a corrective follow-up message if the count is wrong or parsing fails.
    """
    messages = build_messages_fn()
    last_items: list[str] = []

    for attempt in range(max_retries + 1):
        raw = chat_ollama(ollama_url, model, messages)
        items = parse_json_list(raw, key, expected_n)
        last_items = items

        if len(items) == expected_n:
            return items

        if attempt < max_retries:
            messages = messages + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        f"That returned {len(items)} items but I need exactly "
                        f"{expected_n}. Return ONLY valid JSON of the exact shape "
                        f'{{"{key}": [...]}} with exactly {expected_n} entries, '
                        "no prose, no markdown fences."
                    ),
                },
            ]

    print(
        f"Warning: expected {expected_n} items for key '{key}', "
        f"got {len(last_items)} after {max_retries + 1} attempt(s). Using what was returned."
    )
    return last_items


def get_attribute_categories(
    caption: str,
    n_attrs: int,
    ollama_url: str,
    model: str,
    max_retries: int = 2,
) -> list[str]:
    def build_messages():
        system_prompt = (
            "You are building a visual attribute taxonomy for a bias-auditing dataset. "
            "Each attribute category you propose will later be filled with labels, and "
            "each label will be tested against real images by asking a vision-language "
            "model a yes/no question: 'Is <label> present in this image?' Therefore "
            "every category MUST describe something that is directly visible in a "
            "single still photograph — no inferred facts (e.g. compass orientation, "
            "GPS coordinates, city/country name, historical facts) and nothing that "
            "requires context outside the frame."
        )

        user_prompt = f"""Scene description: "{caption}"

Propose exactly {n_attrs} attribute categories that could visually distinguish different
images matching this scene description.

Rules:
- Each category must be a short noun phrase (2-5 words), e.g. "building material",
  "color palette", "crowd density", "weather condition".
- Each category must be answerable purely from pixels in one photo — no location names,
  no orientation/compass facts, no dates, no text requiring OCR of tiny details.
- Categories must be mutually distinct (no overlapping meaning).
- Do not include categories about count/number unless the count is easily eyeballed
  (e.g. "crowd density" is fine, "exact number of buildings" is not).
- Do NOT limit yourself to architecture, material, and color categories. Include a mix
  of category types, and specifically include at least 2-3 categories about concrete
  objects, symbols, attire, or items that people or the scene might contain, if relevant
  to the scene description (e.g. for a religious site: "religious symbol present",
  "religious attire", "ritual object present"; for an office: "device present",
  "signage type"; for a park: "recreational equipment").

Return ONLY valid JSON, no prose, no markdown fences, in this exact shape:
{{"categories": ["category one", "category two", ...]}}"""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    return _generate_with_retry(
        build_messages, "categories", n_attrs, ollama_url, model, max_retries
    )


def get_labels_for_category(
    caption: str,
    attr_c: str,
    n_labels: int,
    ollama_url: str,
    model: str,
    max_retries: int = 2,
) -> list[str]:
    def build_messages():
        system_prompt = (
            "You are generating candidate labels for one attribute category in a "
            "visual bias-auditing dataset. Each label will be tested against real "
            "photos with the yes/no question: 'Is <label> present in this image?' A "
            "label is only useful if that question is unambiguous and answerable by "
            "looking at the image alone."
        )

        user_prompt = f"""Scene description: "{caption}"
Attribute category: "{attr_c}"

Generate exactly {n_labels} labels that are mutually exclusive values within this one
category (i.e. a single image should plausibly match only one, or very few, of them).

Rules:
- Each label must be a short phrase, 1-4 words, same grammatical form as the others
  (e.g. all adjectives, or all noun phrases — do not mix).
- Each label must describe ONLY the category above — do not smuggle in facts from other
  categories (e.g. under "building material" don't write "small brick chapel", just
  "brick").
- No proper nouns (no specific place names, people, or brand names).
- No full sentences or scene descriptions — a label is a single visual attribute value,
  not a caption.
- Labels must be visually distinguishable from each other by a VLM looking at a photo.
- If the category is about an object, symbol, or attire (e.g. "religious symbol
  present"), labels should name the specific concrete object/item itself (e.g. "cross",
  "crescent", "star of David", "headscarf", "prayer beads"), not abstract descriptions.

Return ONLY valid JSON, no prose, no markdown fences, in this exact shape:
{{"labels": ["label one", "label two", ...]}}"""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    return _generate_with_retry(
        build_messages, "labels", n_labels, ollama_url, model, max_retries
    )


def main():
    args = parse_args()

    if args.n_attrs <= 0:
        raise SystemExit("--n-attrs must be > 0")
    if args.n_labels <= 0:
        raise SystemExit("--n-labels must be > 0")

    locations = DEFAULT_LOCATIONS.copy()

    if not locations:
        raise SystemExit("Provide at least one non-empty location.")

    if not check_ollama(DEFAULT_OLLAMA_URL):
        raise SystemExit(
            f"Ollama not running at {DEFAULT_OLLAMA_URL}. Start Ollama first: ollama serve"
        )

    per_location_results = []
    for idx, location in enumerate(locations, start=1):
        caption = location
        print(f"[{idx}/{len(locations)}] Processing location: {location}")

        categories = get_attribute_categories(
            caption=caption,
            n_attrs=args.n_attrs,
            ollama_url=DEFAULT_OLLAMA_URL,
            model=DEFAULT_MODEL,
            max_retries=DEFAULT_MAX_RETRIES,
        )

        if not categories:
            print(
                f"Warning: Could not parse any attribute categories for '{location}'. Skipping."
            )
            continue

        labels_by_attribute = {}
        for attr in categories:
            labels = get_labels_for_category(
                caption=caption,
                attr_c=attr,
                n_labels=args.n_labels,
                ollama_url=DEFAULT_OLLAMA_URL,
                model=DEFAULT_MODEL,
                max_retries=DEFAULT_MAX_RETRIES,
            )
            labels_by_attribute[attr] = labels

        per_location_results.append(
            {
                "location": location,
                "caption": caption,
                "n_attrs_requested": args.n_attrs,
                "n_labels_requested": args.n_labels,
                "attribute_categories": categories,
                "labels_by_attribute": labels_by_attribute,
            }
        )

    if not per_location_results:
        raise SystemExit("No results generated for any location.")

    if len(per_location_results) == 1:
        result = per_location_results[0]
    else:
        result = {
            "n_locations": len(per_location_results),
            "locations_requested": locations,
            "model": DEFAULT_MODEL,
            "ollama_url": DEFAULT_OLLAMA_URL,
            "max_retries": DEFAULT_MAX_RETRIES,
            "n_attrs_requested": args.n_attrs,
            "n_labels_requested": args.n_labels,
            "results": per_location_results,
        }

    DEFAULT_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DEFAULT_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("Generated attribute taxonomy:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved to {DEFAULT_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
