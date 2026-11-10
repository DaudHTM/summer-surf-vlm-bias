import argparse
import os
import base64
import json
from pathlib import Path
import requests


def parse_args():
    parser = argparse.ArgumentParser(
        description="Caption all images in subfolders under inpainting_results using Ollama."
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path("inpainting_results"),
        help="Root folder containing per-image subfolders.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llava",
        help="Ollama model name (e.g., llava, qwen-vl).",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        help="Ollama API endpoint URL.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("inpainting_results_captions.json"),
        help="Output JSON file for captions.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe this image in rich detail in English. Include the main subjects, their appearance, actions, setting, mood, lighting, colors, and any notable objects or context. Keep the caption concise but highly specific:",
        help="Caption prompt text sent to the model.",
    )
    parser.add_argument(
        "--skip-mask-images",
        action="store_true",
        help="Skip captioning images whose filename contains 'mask'.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-generate captions even when existing captions are present.",
    )
    return parser.parse_args()


def check_ollama(ollama_url: str) -> bool:
    """Check if Ollama is running."""
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception as e:
        print(f"Error connecting to Ollama at {ollama_url}: {e}")
        return False


def caption_image(image_path: Path, model: str, prompt: str, ollama_url: str) -> str:
    """Caption a single image using Ollama."""
    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_data],
        "stream": False,
    }

    response = requests.post(
        f"{ollama_url}/api/generate",
        json=payload,
        timeout=300,
    )
    response.raise_for_status()

    result = response.json()
    caption = result.get("response", "").strip()
    if prompt in caption:
        caption = caption.split(prompt, 1)[-1].strip()
    return caption


if __name__ == "__main__":
    args = parse_args()
    root_dir = args.root_dir

    if not root_dir.exists() or not root_dir.is_dir():
        raise SystemExit(f"Root directory not found: {root_dir}")

    if not check_ollama(args.ollama_url):
        raise SystemExit(
            f"Ollama not running at {args.ollama_url}. "
            f"Start Ollama first: ollama serve"
        )

    output_data = {}
    if args.output_file.exists() and not args.force:
        try:
            with open(args.output_file, "r", encoding="utf-8") as f:
                output_data = json.load(f)
        except Exception:
            output_data = {}

    for subdir in sorted(root_dir.iterdir()):
        if not subdir.is_dir():
            continue

        folder_name = subdir.name
        print(f"Processing folder: {folder_name}")
        folder_data = output_data.get(folder_name, {})

        for image_path in sorted(subdir.iterdir()):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
                continue
            if args.skip_mask_images and "mask" in image_path.name.lower():
                continue
            if image_path.name in folder_data and not args.force:
                print(f"  Skipping already captioned image: {image_path.name}")
                continue

            try:
                caption = caption_image(image_path, args.model, args.prompt, args.ollama_url)
                print(f"  {image_path.name}: {caption}")
                folder_data[image_path.name] = caption
                output_data[folder_name] = folder_data

                args.output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(args.output_file, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)

                print(f"  Saved progress to {args.output_file}")
            except Exception as e:
                print(f"  ERROR captioning {image_path.name}: {e}")
                continue

    print(f"Finished. Final captions saved to {args.output_file}")
