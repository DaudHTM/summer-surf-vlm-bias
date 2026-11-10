import argparse
import base64
import io
import json
import os
from pathlib import Path
import time

import requests
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SKIP_IMAGE_STEMS = {"mask", "comparison", "masked_person_input"}
VISION_MODEL_HINTS = ("llava", "bakllava", "vl", "vision", "gemma3", "qwen2-vl", "qwen-vl")
DEFAULT_TAXONOMY_FILE = Path("location_attribute_labels.json")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Annotate each image in each result subfolder using the per-location "
            "attribute taxonomy from location_attribute_labels.json, then write one "
            "JSON per subfolder."
        )
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=Path("places_inpainting_results"),
        help="Root directory containing per-image result subfolders.",
    )
    parser.add_argument(
        "--attributes-file",
        type=Path,
        default=DEFAULT_TAXONOMY_FILE,
        help="Location taxonomy JSON file containing per-location attribute labels.",
    )
    parser.add_argument(
        "--location",
        type=str,
        default=None,
        help=(
            "Optional location name to use for every image. If omitted, the script "
            "tries to infer the location from the image filename or folder name."
        ),
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        help="Ollama API URL.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-vl:8b-instruct",
        help=(
            "Vision-capable Ollama model name. Defaults to qwen3-vl:8b-instruct."
        ),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("location_attribute_annotations.json"),
        help="Output JSON filename written in each subfolder.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retries per image if JSON parsing or label matching fails.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--vlm-long-side",
        type=int,
        default=1024,
        help="Resize each image so the longer side equals this value before VLM inference.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of subfolders processed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-subfolder output JSON files.",
    )
    parser.add_argument(
        "--claims-dir",
        type=Path,
        default=None,
        help=(
            "Directory to store folder claim lock files. Defaults to "
            "<images-root>/.location_attribute_claims."
        ),
    )
    parser.add_argument(
        "--claim-stale-seconds",
        type=int,
        default=6 * 60 * 60,
        help="Treat a folder claim as stale after this many seconds.",
    )
    return parser.parse_args()


def encode_image_b64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def encode_resized_image_b64(image_path: Path, target_long_side: int = 1024) -> str:
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        width, height = rgb.size
        long_side = max(width, height)
        if long_side <= 0:
            raise ValueError(f"Invalid image dimensions for {image_path}")

        scale = float(target_long_side) / float(long_side)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))

        resized = rgb.resize((new_width, new_height), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        resized.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def parse_json_object(raw: str) -> dict:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Expected top-level JSON object")
    return data


def extract_json_candidate(raw: str) -> str:
    text = raw.strip()
    if not text:
        return text

    try:
        parse_json_object(text)
        return text
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1].strip()
        try:
            parse_json_object(candidate)
            return candidate
        except Exception:
            return candidate

    return text


def normalize_key(value: str) -> str:
    return " ".join(value.lower().replace("_", " ").replace("-", " ").split())


def load_location_taxonomies(attributes_file: Path) -> tuple[dict, dict[str, dict]]:
    if not attributes_file.exists():
        raise SystemExit(f"Attributes file not found: {attributes_file}")

    with open(attributes_file, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    entries = taxonomy.get("results", [])
    if not entries:
        raise SystemExit(
            "Attributes file must contain a 'results' array with one taxonomy per location."
        )

    taxonomy_by_key = {}
    for entry in entries:
        location_name = entry.get("location") or entry.get("caption")
        attribute_categories = entry.get("attribute_categories")
        labels_by_attribute = entry.get("labels_by_attribute")
        if not location_name or not attribute_categories or not labels_by_attribute:
            continue

        key = normalize_key(location_name)
        taxonomy_by_key[key] = {
            "location": location_name,
            "attribute_categories": attribute_categories,
            "labels_by_attribute": labels_by_attribute,
            "source_entry": entry,
        }

    if not taxonomy_by_key:
        raise SystemExit("No valid location taxonomies found in the attributes file.")

    return taxonomy, taxonomy_by_key


def save_taxonomy(attributes_file: Path, taxonomy: dict):
    attributes_file.parent.mkdir(parents=True, exist_ok=True)
    temp_path = attributes_file.with_suffix(attributes_file.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, ensure_ascii=False, indent=2)
    temp_path.replace(attributes_file)


def append_label_if_missing(labels: list[str], candidate: str) -> tuple[str, bool]:
    normalized_candidate = candidate.strip()
    normalized_key = normalized_candidate.lower()
    for existing in labels:
        if existing.strip().lower() == normalized_key:
            return existing, False

    labels.append(normalized_candidate)
    return normalized_candidate, True


def build_location_prompt(location: str, attribute_categories: list[str], labels_by_attribute: dict[str, list[str]]) -> str:
    field_lines = []
    for category in attribute_categories:
        labels = labels_by_attribute.get(category, [])
        label_text = ", ".join(labels) if labels else "(no existing labels)"
        field_lines.append(
            f'- "{category}": choose one or more existing labels from [{label_text}] or null if the attribute is not applicable. '
            "If none of the listed labels fit but the attribute is clearly visible, invent one or more short new labels."
        )

    fields_text = "\n".join(field_lines)
    return (
        f'You are annotating one image for the location "{location}".\n\n'
        "For each attribute category below, return exactly one JSON value.\n"
        "Rules:\n"
        "- First try to match one of the existing answer choices exactly.\n"
        "- If multiple existing labels fit, return a JSON array of the exact label strings.\n"
        "- If only one existing label fits, return that label as a single string or as a one-item array.\n"
        "- Use JSON null if the attribute is not applicable or not visible.\n"
        "- Only if none of the listed labels fit, invent one or more short new labels and use them.\n"
        "- Prefer existing labels over inventing new ones.\n"
        "- If a label choice is literally the word none, treat it as a normal label choice; JSON null is only for not applicable.\n\n"
        "Return ONLY valid JSON with exactly these keys and no extra text.\n\n"
        f"{fields_text}"
    )


def call_ollama_chat(
    ollama_url: str,
    model: str,
    prompt: str,
    image_b64: str,
    timeout: int,
) -> tuple[str, str, str]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_b64],
            }
        ],
        "stream": False,
        "format": "json",
        "options": {
            "num_ctx": 4096,
            "num_predict": 2048,
            "temperature": 0,
        },
    }
    response = requests.post(f"{ollama_url}/api/chat", json=payload, timeout=timeout)
    if not response.ok:
        raise requests.HTTPError(
            f"{response.status_code} Client Error for url: {response.url}: {response.text}",
            response=response,
        )
    response.raise_for_status()
    response_text = response.text
    data = response.json()
    content = data.get("message", {}).get("content", "").strip()
    thinking = data.get("message", {}).get("thinking", "")
    if thinking is None:
        thinking = ""
    return content, str(thinking).strip(), response_text


def is_vision_model_name(model_name: str) -> bool:
    lowered = model_name.lower()
    return any(hint in lowered for hint in VISION_MODEL_HINTS)


def detect_model(ollama_url: str, override_model: str | None) -> str:
    if override_model:
        if not is_vision_model_name(override_model):
            raise SystemExit(
                f"Model override '{override_model}' does not look vision-capable. "
                "Use a VLM such as llava, bakllava, qwen2-vl, or llama3.2-vision."
            )
        return override_model

    ps_resp = requests.get(f"{ollama_url}/api/ps", timeout=30)
    ps_resp.raise_for_status()
    ps_data = ps_resp.json()
    running = ps_data.get("models", [])
    for model_info in running:
        name = model_info.get("name")
        if name and is_vision_model_name(name):
            return name

    tags_resp = requests.get(f"{ollama_url}/api/tags", timeout=30)
    tags_resp.raise_for_status()
    tags_data = tags_resp.json()
    installed = tags_data.get("models", [])
    for model_info in installed:
        name = model_info.get("name")
        if name and is_vision_model_name(name):
            return name

    available = [m.get("name") for m in running if m.get("name")]
    if not available:
        available = [m.get("name") for m in installed if m.get("name")]

    raise SystemExit(
        "No vision-capable Ollama model found. Available models: "
        + (", ".join(available) if available else "none")
        + ". Start a vision model like llava, bakllava, qwen2-vl, or llama3.2-vision."
    )


def collect_result_dirs(images_root: Path) -> list[Path]:
    return sorted(p for p in images_root.iterdir() if p.is_dir() and p.name != ".claims")


def collect_target_images(result_dir: Path) -> list[Path]:
    images = []
    for path in sorted(result_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.stem in SKIP_IMAGE_STEMS:
            continue
        images.append(path)
    return images


def resolve_claims_dir(images_root: Path, claims_dir_arg: Path | None) -> Path:
    claims_dir = claims_dir_arg or (images_root / ".location_attribute_claims")
    claims_dir.mkdir(parents=True, exist_ok=True)
    return claims_dir


def claim_file_for_folder(claims_dir: Path, folder_name: str) -> Path:
    return claims_dir / f"{folder_name}.lock"


def claim_is_stale(claim_file: Path, stale_seconds: int) -> bool:
    try:
        content = claim_file.read_text(encoding="utf-8").strip()
    except Exception:
        return True

    if not content:
        return True

    parts = [part.strip() for part in content.split("|") if part.strip()]
    if not parts:
        return True

    state = parts[0].lower()
    if state != "processing":
        return True

    pid = None
    timestamp = None
    for part in parts[1:]:
        if part.startswith("pid="):
            try:
                pid = int(part.split("=", 1)[1])
            except Exception:
                pid = None
        elif part.startswith("ts="):
            try:
                timestamp = float(part.split("=", 1)[1])
            except Exception:
                timestamp = None

    if timestamp is not None and (time.time() - timestamp) > stale_seconds:
        return True

    if pid is not None:
        try:
            os.kill(pid, 0)
        except OSError:
            return True

    return False


def claim_folder(claims_dir: Path, folder_name: str, stale_seconds: int) -> bool:
    claim_file = claim_file_for_folder(claims_dir, folder_name)

    if claim_file.exists():
        if claim_is_stale(claim_file, stale_seconds):
            try:
                claim_file.unlink()
            except FileNotFoundError:
                pass
        else:
            return False

    payload = f"processing|pid={os.getpid()}|ts={time.time()}\n"
    try:
        fd = os.open(str(claim_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False

    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(payload)
    return True


def release_folder_claim(claims_dir: Path, folder_name: str):
    claim_file = claim_file_for_folder(claims_dir, folder_name)
    try:
        claim_file.unlink()
    except FileNotFoundError:
        pass


def infer_location_key(image_path: Path, known_keys: set[str], explicit_location: str | None = None) -> str | None:
    if explicit_location:
        candidate = normalize_key(explicit_location)
        return candidate if candidate in known_keys else None

    candidates = [normalize_key(image_path.stem), normalize_key(image_path.parent.name)]

    if image_path.stem.endswith("s"):
        candidates.append(normalize_key(image_path.stem[:-1]))

    for candidate in candidates:
        if candidate in known_keys:
            return candidate
    return None


def load_or_init_payload(
    output_path: Path,
    result_dir: Path,
    model: str,
    ollama_url: str,
    attributes_file: Path,
    overwrite: bool,
) -> dict:
    if overwrite or not output_path.exists():
        payload = {}
    else:
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            payload = {}

    payload["subfolder"] = result_dir.name
    payload["vlm_model"] = model
    payload["ollama_url"] = ollama_url
    payload["attributes_file"] = str(attributes_file)
    if not isinstance(payload.get("images"), dict):
        payload["images"] = {}
    return payload


def load_existing_results(output_file: Path) -> dict:
    if not output_file.exists():
        return {}
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        images = data.get("images", {})
        return images if isinstance(images, dict) else {}
    except Exception:
        return {}


def write_payload(output_path: Path, payload: dict):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def coerce_attribute_answers(
    value,
    labels: list[str],
) -> tuple[list[object] | None, bool]:
    label_lookup = {label.strip().lower(): label for label in labels}

    if value is None:
        return None, False

    if isinstance(value, str):
        normalized_values = [part.strip() for part in value.split(",") if part.strip()]
        if not normalized_values:
            return None, False
    elif isinstance(value, list):
        normalized_values = []
        for item in value:
            if isinstance(item, str):
                parts = [part.strip() for part in item.split(",") if part.strip()]
                normalized_values.extend(parts)
            elif item is not None:
                text = str(item).strip()
                if text:
                    normalized_values.append(text)
    else:
        normalized = str(value).strip()
        if not normalized:
            return None, False
        normalized_values = [normalized]

    if not normalized_values:
        return None, False

    answers = []
    added_any = False
    seen = set()
    for item in normalized_values:
        lowered = item.lower()
        if lowered == "null" or lowered == "none":
            continue
        if lowered in label_lookup:
            canonical = label_lookup[lowered]
        else:
            canonical, added = append_label_if_missing(labels, item)
            added_any = added_any or added
        if canonical.lower() not in seen:
            seen.add(canonical.lower())
            answers.append(canonical)

    if not answers:
        return None, added_any

    return answers, added_any


def annotate_single_image(
    image_path: Path,
    location_entry: dict,
    model: str,
    ollama_url: str,
    max_retries: int,
    timeout: int,
    vlm_long_side: int,
) -> tuple[dict, dict, bool]:
    prompt = build_location_prompt(
        location_entry["location"],
        location_entry["attribute_categories"],
        location_entry["labels_by_attribute"],
    )

    last_error = None
    taxonomy_updated = False

    for attempt in range(max_retries + 1):
        try:
            image_b64 = encode_resized_image_b64(image_path, vlm_long_side)
            raw, thinking, response_text = call_ollama_chat(ollama_url, model, prompt, image_b64, timeout)
            candidate_text = extract_json_candidate(raw or thinking)
            if not candidate_text:
                print(f"    full Ollama response for {image_path.name}: {response_text}")
                raise ValueError("Model returned empty assistant content")

            data = parse_json_object(candidate_text)

            missing_keys = [key for key in location_entry["attribute_categories"] if key not in data]
            if missing_keys and attempt < max_retries:
                raise ValueError(f"Missing keys: {', '.join(missing_keys[:3])}")

            answers = {}
            new_labels = {}
            for category in location_entry["attribute_categories"]:
                labels = location_entry["labels_by_attribute"].setdefault(category, [])
                value, added = coerce_attribute_answers(
                    data.get(category),
                    labels,
                )
                answers[category] = value
                if added:
                    new_labels[category] = value

            taxonomy_updated = bool(new_labels)
            return answers, new_labels, taxonomy_updated
        except Exception as exc:
            last_error = str(exc)

    print(f"    Failed after {max_retries + 1} attempt(s) for {image_path.name}: {last_error}")
    return (
        {category: None for category in location_entry["attribute_categories"]},
        {},
        False,
    )


def annotate_result_directory(
    result_dir: Path,
    taxonomy: dict,
    taxonomy_by_key: dict[str, dict],
    args: argparse.Namespace,
) -> tuple[dict[str, dict], list[str], bool]:
    per_image_results = {} if args.overwrite else load_existing_results(result_dir / args.output_file.name)
    already_done = set(per_image_results.keys())
    skipped_files = []
    known_keys = set(taxonomy_by_key.keys())
    taxonomy_updated = False

    image_paths = collect_target_images(result_dir)
    for image_path in image_paths:
        location_key = infer_location_key(image_path, known_keys, args.location)
        if location_key is None:
            skipped_files.append(image_path.name)
            continue

        key = image_path.stem
        if key in already_done:
            continue

        location_entry = taxonomy_by_key[location_key]
        print(f"  Annotating {result_dir.name}/{image_path.name} as {location_entry['location']} ...")
        answers, new_labels, image_taxonomy_updated = annotate_single_image(
            image_path,
            location_entry,
            args.model,
            args.ollama_url,
            args.max_retries,
            args.timeout,
            args.vlm_long_side,
        )
        per_image_results[key] = {
            "file_name": image_path.name,
            "location": location_entry["location"],
            "answers": answers,
            "new_labels_added": new_labels,
        }
        print(
            "    output for "
            f"{image_path.name}: {json.dumps(per_image_results[key], ensure_ascii=False)}"
        )
        taxonomy_updated = taxonomy_updated or image_taxonomy_updated

    return per_image_results, skipped_files, taxonomy_updated


def main():
    args = parse_args()

    if not args.images_root.exists() or not args.images_root.is_dir():
        raise SystemExit(f"Images root not found: {args.images_root}")

    taxonomy, taxonomy_by_key = load_location_taxonomies(args.attributes_file)
    model = detect_model(args.ollama_url, args.model)
    claims_dir = resolve_claims_dir(args.images_root, args.claims_dir)
    print(f"Using Ollama model: {model}")
    print(f"Using folder claims directory: {claims_dir}")

    if args.location:
        explicit_key = normalize_key(args.location)
        if explicit_key not in taxonomy_by_key:
            raise SystemExit(
                f"Location '{args.location}' not found in {args.attributes_file}."
            )

    result_dirs = collect_result_dirs(args.images_root)
    if args.limit:
        result_dirs = result_dirs[: args.limit]

    if not result_dirs:
        raise SystemExit(f"No result subfolders found under {args.images_root}")

    for result_dir in result_dirs:
        if not claim_folder(claims_dir, result_dir.name, args.claim_stale_seconds):
            print(f"Skipping folder {result_dir.name} (already claimed by another worker).")
            continue

        output_path = result_dir / args.output_file.name

        try:
            print(f"Processing folder {result_dir.name} ...")
            per_image_results, skipped_files, taxonomy_updated = annotate_result_directory(
                result_dir,
                taxonomy,
                taxonomy_by_key,
                args,
            )

            payload = load_or_init_payload(
                output_path,
                result_dir,
                model,
                args.ollama_url,
                args.attributes_file,
                args.overwrite,
            )
            payload["images"] = per_image_results
            write_payload(output_path, payload)

            if taxonomy_updated:
                save_taxonomy(args.attributes_file, taxonomy)

            if skipped_files:
                print(
                    "  Skipped files without a matching location taxonomy: "
                    + ", ".join(sorted(skipped_files))
                )

            print(f"  Wrote {output_path}")
        finally:
            release_folder_claim(claims_dir, result_dir.name)


if __name__ == "__main__":
    main()
