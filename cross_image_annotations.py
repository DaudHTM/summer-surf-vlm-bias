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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Annotate each original/inpainted image in each places_inpainting_results "
            "subfolder using the currently running Ollama VLM, then write one JSON per subfolder."
        )
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=Path("places_inpainting_results"),
        help="Root directory containing per-image result subfolders.",
    )
    parser.add_argument(
        "--questions-file",
        type=Path,
        default=Path("cross_image_questions.json"),
        help="Question schema JSON file.",
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
        default=None,
        help=(
            "Optional model override. If omitted, use the first currently running "
            "model from Ollama /api/ps, falling back to the first installed model."
        ),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("cross_image_annotations.json"),
        help="Output JSON filename written in each subfolder.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retries per image if JSON parsing/validation fails.",
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
            "<images-root>/.cross_image_claims."
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


def load_questions(questions_file: Path) -> list[dict]:
    if not questions_file.exists():
        raise SystemExit(f"Questions file not found: {questions_file}")

    with open(questions_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_questions = data.get("cross_location_questions")
    if not isinstance(raw_questions, list) or not raw_questions:
        raise SystemExit("Questions file must contain a non-empty 'cross_location_questions' list.")

    normalized_questions = []
    seen_keys = set()
    for item in raw_questions:
        key = str(item.get("key", "")).strip()
        prompt = str(item.get("prompt", "")).strip()
        response_type = str(item.get("response_type", "")).strip().lower()
        allowed_values = item.get("allowed_values")

        if not key or not prompt or not response_type:
            raise SystemExit("Each question must include key, prompt, and response_type.")
        if key in seen_keys:
            raise SystemExit(f"Duplicate question key found: {key}")
        seen_keys.add(key)

        if response_type not in {"integer", "boolean", "categorical", "short_text"}:
            raise SystemExit(f"Unsupported response_type '{response_type}' for key '{key}'")

        if response_type in {"integer", "boolean", "categorical"}:
            if not isinstance(allowed_values, list) or not allowed_values:
                raise SystemExit(
                    f"Question '{key}' with response_type '{response_type}' requires non-empty allowed_values."
                )

        normalized_questions.append(
            {
                "key": key,
                "prompt": prompt,
                "response_type": response_type,
                "allowed_values": allowed_values,
            }
        )

    return normalized_questions


def coerce_answer(question: dict, value):
    response_type = question["response_type"]
    allowed_values = question.get("allowed_values")

    if response_type == "short_text":
        if value is None:
            return ""
        return str(value).strip()

    if isinstance(value, str) and value.strip().lower() == "unsure":
        if isinstance(allowed_values, list):
            for allowed in allowed_values:
                if isinstance(allowed, str) and allowed.lower() == "unsure":
                    return "unsure"
        raise ValueError(f"'unsure' is not allowed for {question['key']}")

    if response_type == "integer":
        if isinstance(value, bool):
            raise ValueError(f"Invalid integer answer for {question['key']}")
        parsed = int(value)
        allowed_ints = [v for v in (allowed_values or []) if isinstance(v, int)]
        if parsed not in allowed_ints:
            raise ValueError(f"Value {parsed} not allowed for {question['key']}")
        return parsed

    if response_type == "boolean":
        if isinstance(value, bool):
            parsed = value
        elif isinstance(value, str):
            normalized = value.strip().lower()
            if normalized == "true":
                parsed = True
            elif normalized == "false":
                parsed = False
            else:
                raise ValueError(f"Invalid boolean answer for {question['key']}")
        else:
            raise ValueError(f"Invalid boolean answer for {question['key']}")

        allowed_bools = [v for v in (allowed_values or []) if isinstance(v, bool)]
        if parsed not in allowed_bools:
            raise ValueError(f"Value {parsed} not allowed for {question['key']}")
        return parsed

    if response_type == "categorical":
        normalized = str(value).strip().lower()
        allowed_strs = [v for v in (allowed_values or []) if isinstance(v, str)]
        allowed_lookup = {v.lower(): v for v in allowed_strs}
        if normalized not in allowed_lookup:
            raise ValueError(f"Value {normalized!r} not allowed for {question['key']}")
        return allowed_lookup[normalized]

    raise ValueError(f"Unsupported response_type: {response_type}")


def fallback_answer_value(question: dict, value):
    response_type = question["response_type"]

    if response_type == "integer":
        if isinstance(value, bool):
            return value
        try:
            return int(value)
        except Exception:
            return value

    if response_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized == "true":
                return True
            if normalized == "false":
                return False
        return value

    if response_type in {"categorical", "short_text"}:
        return value if isinstance(value, str) else str(value)

    return value


def build_questions_prompt(questions: list[dict]) -> str:
    lines = []
    for q in questions:
        allowed_values = q.get("allowed_values")
        if q["response_type"] == "short_text":
            allowed_text = "any short text answer"
        else:
            allowed_text = ", ".join(str(v) for v in allowed_values)
        lines.append(f'- "{q["key"]}": {q["prompt"]} Allowed values: {allowed_text}.')

    return (
        "Look carefully at this image and answer the following questions about the visible person and scene. "
        "Use only visually supported evidence.\n\n"
        "Return ONLY valid JSON with exactly these keys:\n"
        + "\n".join(lines)
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
        "think": False,
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
            "num_predict": 15000,
            "temperature": 0,
        },
    }
    response = requests.post(f"{ollama_url}/api/chat", json=payload, timeout=timeout)
    response.raise_for_status()
    response_text = response.text
    data = response.json()
    content = data.get("message", {}).get("content", "").strip()
    thinking = data.get("message", {}).get("thinking", "")
    if thinking is None:
        thinking = ""
    return content, str(thinking).strip(), response_text


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


def is_vision_model_name(model_name: str) -> bool:
    lowered = model_name.lower()
    return any(hint in lowered for hint in VISION_MODEL_HINTS)


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
    claims_dir = claims_dir_arg or (images_root / ".cross_image_claims")
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


def load_or_init_payload(
    output_path: Path,
    result_dir: Path,
    model: str,
    ollama_url: str,
    questions: list[dict],
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
    payload["questions"] = questions
    if not isinstance(payload.get("images"), dict):
        payload["images"] = {}
    return payload


def annotate_single_image(
    image_path: Path,
    questions: list[dict],
    model: str,
    ollama_url: str,
    max_retries: int,
    timeout: int,
    vlm_long_side: int,
) -> dict:
    prompt = build_questions_prompt(questions)

    last_error = None
    last_data = None
    for attempt in range(max_retries + 1):
        try:
            image_b64 = encode_resized_image_b64(image_path, vlm_long_side)
            raw, thinking, response_text = call_ollama_chat(ollama_url, model, prompt, image_b64, timeout)
            print(f"    assistant content for {image_path.name}: {raw!r}")
            if thinking:
                print(f"    assistant thinking for {image_path.name}: {thinking!r}")

            candidate_text = extract_json_candidate(raw or thinking)
            if not candidate_text:
                print(f"    full Ollama response for {image_path.name}: {response_text}")
                raise ValueError("Model returned empty assistant content and empty thinking")

            data = parse_json_object(candidate_text)
            last_data = data
            answers = {}
            for q in questions:
                if q["key"] not in data:
                    raise ValueError(f"Missing key '{q['key']}' in model response")
                try:
                    answers[q["key"]] = coerce_answer(q, data[q["key"]])
                except Exception:
                    if attempt == max_retries:
                        answers[q["key"]] = fallback_answer_value(q, data[q["key"]])
                    else:
                        raise
            return answers
        except Exception as exc:
            last_error = str(exc)

    if isinstance(last_data, dict):
        fallback_answers = {}
        for q in questions:
            if q["key"] in last_data:
                fallback_answers[q["key"]] = fallback_answer_value(q, last_data[q["key"]])
            else:
                fallback_answers[q["key"]] = "unsure" if "unsure" in (q.get("allowed_values") or []) else ""
        return {
            "error": f"Failed after {max_retries + 1} attempts: {last_error}",
            **fallback_answers,
        }

    return {
        "error": f"Failed after {max_retries + 1} attempts: {last_error}",
        **{q["key"]: "unsure" if "unsure" in (q.get("allowed_values") or []) else "" for q in questions},
    }


def write_payload(output_path: Path, payload: dict):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()

    if not args.images_root.exists() or not args.images_root.is_dir():
        raise SystemExit(f"Images root not found: {args.images_root}")

    questions = load_questions(args.questions_file)
    model = detect_model(args.ollama_url, args.model)
    claims_dir = resolve_claims_dir(args.images_root, args.claims_dir)
    print(f"Using Ollama model: {model}")
    print(f"Using folder claims directory: {claims_dir}")

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
        images = collect_target_images(result_dir)

        try:
            print(f"Processing folder {result_dir.name} ...")
            payload = load_or_init_payload(
                output_path,
                result_dir,
                model,
                args.ollama_url,
                questions,
                args.overwrite,
            )

            processed_count = 0
            for image_path in images:
                image_key = image_path.stem
                existing = payload["images"].get(image_key)
                if isinstance(existing, dict) and isinstance(existing.get("answers"), dict):
                    continue

                print(f"  Annotating {result_dir.name}/{image_path.name} ...")
                answers = annotate_single_image(
                    image_path,
                    questions,
                    model,
                    args.ollama_url,
                    args.max_retries,
                    args.timeout,
                    args.vlm_long_side,
                )
                payload["images"][image_key] = {
                    "file_name": image_path.name,
                    "answers": answers,
                }
                processed_count += 1
                write_payload(output_path, payload)

            if processed_count == 0:
                print(f"  Nothing new to annotate (resume state is up-to-date).")
            else:
                print(f"  Wrote/updated {output_path} ({processed_count} new image(s)).")
        finally:
            release_folder_claim(claims_dir, result_dir.name)


if __name__ == "__main__":
    main()
