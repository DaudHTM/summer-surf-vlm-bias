import json
import os
import time
import numpy as np
from pathlib import Path
from pycocotools import mask as mask_utils
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import torch
from diffusers import AutoPipelineForInpainting

selected_images_file = "selected_images.txt"
FACET_ROOT = Path(os.getenv("FACET_ROOT", "data/facet"))
IMAGE_DIRS = [str(FACET_ROOT / f"imgs_{i}") for i in range(1, 4)]
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR") or None
# places = ["shopping mall" , "park", "restaurant" , "sports stadium", "religious site", "cultural site", "office", "public transport" , "a street with people, cars, busses, bicycles, etc."]
places = [
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
targets = [(place.replace(" ", "_"), place) for place in places]
RESULTS_DIR = Path("places_inpainting_results")
RESULTS_DIR.mkdir(exist_ok=True)
CLAIMS_DIR = RESULTS_DIR / ".claims"
CLAIMS_DIR.mkdir(exist_ok=True)
TRACKING_FILE = RESULTS_DIR / "generated_images_log.txt"
HF_TOKEN_ENV_VAR = "HF_TOKEN"
GPU_INDEX = int(os.getenv("GPU_INDEX", "0"))
CLAIM_STALE_SECONDS = 6 * 60 * 60

# FLUX Fill tends to work better with moderate guidance and fewer steps.
FLUX_GUIDANCE_SCALE = 31
FLUX_NUM_INFERENCE_STEPS = 45
FLUX_MAX_SEQUENCE_LENGTH = 256

def get_pipeline():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires GPU execution.")

    if not 0 <= GPU_INDEX < torch.cuda.device_count():
        raise RuntimeError(
            f"Requested GPU index {GPU_INDEX}, but only {torch.cuda.device_count()} GPU(s) are visible."
        )

    cuda_device = f"cuda:{GPU_INDEX}"

    model_id = "black-forest-labs/FLUX.1-Fill-dev"
    hf_token = os.getenv(HF_TOKEN_ENV_VAR)
    if not hf_token:
        print(
            f"Warning: {HF_TOKEN_ENV_VAR} is not set. "
            "If model download fails, set your Hugging Face token in this env var."
        )

    dtype_attempts = [
        (torch.bfloat16, "bfloat16"),
        (torch.float16, "float16"),
        (torch.float32, "float32"),
    ]

    last_exc = None

    try:
        from diffusers import FluxFillPipeline
    except Exception:
        FluxFillPipeline = None

    for dtype, dtype_name in dtype_attempts:
        try:
            if FluxFillPipeline is not None:
                pipe = FluxFillPipeline.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    token=hf_token,
                    cache_dir=MODEL_CACHE_DIR,
                    local_files_only=os.getenv("HF_HUB_OFFLINE", "0") == "1",
                ).to(cuda_device)
            else:
                pipe = AutoPipelineForInpainting.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    token=hf_token,
                    cache_dir=MODEL_CACHE_DIR,
                    local_files_only=os.getenv("HF_HUB_OFFLINE", "0") == "1",
                ).to(cuda_device)
            pipe.enable_attention_slicing()
            print(
                f"Using FLUX.1 Fill on CUDA device {GPU_INDEX}: {torch.cuda.get_device_name(GPU_INDEX)} "
                f"with {dtype_name}"
            )
            print(f"Using local Hugging Face cache at: {MODEL_CACHE_DIR}")
            return pipe, cuda_device
        except Exception as exc:
            last_exc = exc
            print(f"FLUX.1 Fill init with {dtype_name} failed; trying next dtype.")
            print(f"  Exception: {exc}")

    if FluxFillPipeline is None:
        raise RuntimeError(
            "FluxFillPipeline is unavailable in this diffusers build. "
            "Upgrade diffusers (and transformers) and retry. "
            f"Last initialization error for {model_id}: {last_exc}"
        ) from last_exc

    raise RuntimeError(f"CUDA initialization failed for {model_id}: {last_exc}") from last_exc

pipe, device = get_pipeline()


def load_processed_images():
    """Load set of already-processed image IDs from tracking file."""
    processed = set()
    if TRACKING_FILE.exists():
        with open(TRACKING_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    image_id = line.split("|")[0].strip()
                    processed.add(image_id)

    for claim_file in CLAIMS_DIR.glob("*.lock"):
        try:
            contents = claim_file.read_text(encoding="utf-8").strip().splitlines()
        except Exception:
            continue

        if not contents:
            continue

        status = contents[0].split("|", 1)[0].strip().lower()
        if status == "done":
            processed.add(claim_file.stem)
    return processed


processed_images = load_processed_images()


def _claim_file_for(image_id: str) -> Path:
    return CLAIMS_DIR / f"{Path(image_id).name}.lock"


def _claim_is_stale(claim_file: Path) -> bool:
    try:
        content = claim_file.read_text(encoding="utf-8").strip().splitlines()[0]
    except Exception:
        return True

    parts = [part.strip() for part in content.split("|") if part.strip()]
    if not parts:
        return True

    state = parts[0].lower()
    if state == "done":
        return False

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

    if timestamp is not None and (time.time() - timestamp) > CLAIM_STALE_SECONDS:
        return True

    if pid is not None:
        try:
            os.kill(pid, 0)
        except OSError:
            return True

    return False


def claim_image(image_id: str) -> bool:
    if image_id in processed_images:
        return False

    claim_file = _claim_file_for(image_id)
    if claim_file.exists() and not _claim_is_stale(claim_file):
        return False

    if claim_file.exists() and _claim_is_stale(claim_file):
        try:
            claim_file.unlink()
        except FileNotFoundError:
            pass

    payload = f"processing|pid={os.getpid()}|ts={time.time()}\n"
    try:
        fd = os.open(str(claim_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False

    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(payload)
    return True


def mark_image_done(image_id: str):
    claim_file = _claim_file_for(image_id)
    payload = f"done|pid={os.getpid()}|ts={time.time()}\n"
    with open(claim_file, "w", encoding="utf-8") as f:
        f.write(payload)
    processed_images.add(image_id)



def get_target_size(image, max_dim=1024):
    """Scale down so the longer side is at most max_dim, rounded to a multiple of 16."""
    w, h = image.size
    scale = min(1.0, max_dim / max(w, h))
    new_w = round_to_multiple(int(w * scale))
    new_h = round_to_multiple(int(h * scale))
    return new_w, new_h

def round_to_multiple(value, multiple=16):
    return max(multiple, round(value / multiple) * multiple)

def find_image_path(filename):
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename
    for base in IMAGE_DIRS:
        for root, _, files in os.walk(base):
            if filename in files:
                return os.path.join(root, filename)
    return None


def build_inpainting_mask(person_mask, blur_radius=3):
    if person_mask.mode != "L":
        person_mask = person_mask.convert("L")
    mask_arr = (np.array(person_mask) > 127).astype(np.uint8) * 255
    mask_arr = 255 - mask_arr
    mask = Image.fromarray(mask_arr)
    # Soften the hard binary edge so the model blends the boundary smoothly
    mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return Image.merge("RGB", (mask, mask, mask))


def generate_target_image(source_image, mask_image, target_prompt):

    width, height = get_target_size(source_image, max_dim=1024)

    if (width, height) != source_image.size:
        source_image = source_image.resize((width, height), Image.LANCZOS)
        mask_image = mask_image.resize((width, height), Image.LANCZOS)

    prompt = (
        "A realistic candid photograph of a person. "
        f"Place the person naturally in a {target_prompt} setting. "
        "Keep the person unchanged in identity, appearance, pose, and clothing. "
        "Generate a realistic, coherent background around the person. "
    )
    extra_kwargs = {
        "guidance_scale": FLUX_GUIDANCE_SCALE,
        "num_inference_steps": FLUX_NUM_INFERENCE_STEPS,
        "max_sequence_length": FLUX_MAX_SEQUENCE_LENGTH,
        "height": height,
        "width": width,
    }
    if device == "cuda":
        with torch.autocast("cuda"):
            result = pipe(prompt=prompt, image=source_image, mask_image=mask_image, **extra_kwargs)
    else:
        result = pipe(prompt=prompt, image=source_image, mask_image=mask_image, **extra_kwargs)
    return result.images[0]


with open(selected_images_file, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split("|", 3)  # filename, person_id, seg_json, combo_desc
        image_id = parts[0].strip()
        person_id = parts[1].strip()
        seg_json = parts[2].strip()
        attributes = parts[3].strip() if len(parts) > 3 else ""

        if image_id in processed_images:
            print(f"Image ID: {image_id} (already processed, skipping)")
            continue

        print(f"Image ID: {image_id}")
        print(f"Person ID: {person_id}")
        print(f"Attributes: {attributes}")

        if not claim_image(image_id):
            print(f"Image ID: {image_id} (already claimed or processed elsewhere, skipping)")
            continue

        success = False

        try:
            rle = json.loads(seg_json)
            if not rle:
                print(f"  No mask found for facet_person_id {person_id}")
                print()
                continue

            if isinstance(rle["counts"], str):
                rle["counts"] = rle["counts"].encode("utf-8")

            binary_mask = mask_utils.decode(rle)
            person_mask = Image.fromarray((binary_mask * 255).astype(np.uint8))

            print(f"  Found mask, size: {person_mask.size}")

            image_path = find_image_path(image_id)
            source_image = None
            if image_path:
                try:
                    source_image = Image.open(image_path).convert("RGB")
                except Exception as exc:
                    print(f"  Warning: failed to load image '{image_path}': {exc}")

            if source_image is None:
                print(f"  Warning: image file not found for '{image_id}'")
                continue

            mask_image = build_inpainting_mask(person_mask)
            generated_images = []
            for target_name, target_prompt in targets:
                print(f"  Generating for target: {target_name}")
                generated_images.append((target_name, generate_target_image(source_image, mask_image, target_prompt)))

            image_folder = RESULTS_DIR / Path(image_id).stem
            image_folder.mkdir(parents=True, exist_ok=True)

            original_path = image_folder / "original.png"
            source_image.save(original_path)
            print(f"  Saved original image to {original_path}")

            mask_path = image_folder / "mask.png"
            person_mask.save(mask_path)
            print(f"  Saved mask to {mask_path}")

            for target_name, generated in generated_images:
                out_path = image_folder / f"{target_name}.png"
                generated.save(out_path)
                print(f"  Saved inpainting for {target_name} to {out_path}")

            num_cols = 2 + len(generated_images)
            fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 6))

            axes[0].imshow(source_image)
            axes[0].set_title("Original image")
            axes[0].axis("off")

            axes[1].imshow(person_mask, cmap="gray")
            axes[1].set_title("Mask")
            axes[1].axis("off")

            for i, (target_name, generated) in enumerate(generated_images, start=2):
                axes[i].imshow(generated)
                axes[i].set_title(target_name.replace("_", " ").title())
                axes[i].axis("off")

            fig.suptitle(f"{image_id} | person_id={person_id} | {attributes}", fontsize=16)
            plt.tight_layout()

            comparison_path = image_folder / "comparison.png"
            fig.savefig(comparison_path, bbox_inches="tight")
            print(f"  Saved comparison figure to {comparison_path}")

            with open(TRACKING_FILE, "a", encoding="utf-8") as track_f:
                targets_str = ", ".join(target_name for target_name, _ in generated_images)
                log_line = f"{image_id} | attributes: {attributes} | targets: {targets_str}\n"
                track_f.write(log_line)
                print(f"  Logged generation metadata to {TRACKING_FILE}")

            mark_image_done(image_id)
            success = True

            #plt.show()
        finally:
            if not success:
                claim_file = _claim_file_for(image_id)
                try:
                    claim_file.unlink()
                except FileNotFoundError:
                    pass
