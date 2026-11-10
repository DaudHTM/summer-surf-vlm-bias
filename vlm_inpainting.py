import json
import os
import inspect
import numpy as np
from pathlib import Path
from pycocotools import mask as mask_utils
from PIL import Image
import matplotlib.pyplot as plt
import torch
from diffusers import DiffusionPipeline

selected_images_file = "selected_images.txt"
FACET_ROOT = Path(os.getenv("FACET_ROOT", "data/facet"))
IMAGE_DIRS = [str(FACET_ROOT / f"imgs_{i}") for i in range(1, 4)]
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR") or None
MODEL_ID = "black-forest-labs/FLUX.2-Klein-9B"
# Zero-based line index in selected_images.txt to start processing from.
START_POSITION = int(os.getenv("START_POSITION", "0"))
# places = ["shopping mall" , "park", "restaurant" , "sports stadium", "religious site", "cultural site", "office", "public transport" , "a street with people, cars, busses, bicycles, etc."]
targets = [
    (
        "single_pet",
        "one single pet naturally present in the scene",
    ),
    (
        "single_sports_item",
        "one single sports item naturally present in the scene",
    ),
    (
        "single_flower",
        "one single flower naturally present in the scene",
    ),
    (
        "single_fruit",
        "one single fruit naturally present in the scene",
    ),
]
RESULTS_DIR = Path("item_VLM_inpainting_results")
RESULTS_DIR.mkdir(exist_ok=True)
TRACKING_FILE = RESULTS_DIR / "generated_images_log.txt"
HF_TOKEN_ENV_VAR = "HF_TOKEN"
GPU_INDEX = int(os.getenv("GPU_INDEX", "0"))

# FLUX Fill tends to work better with moderate guidance and fewer steps.
FLUX_GUIDANCE_SCALE = 1.0
FLUX_NUM_INFERENCE_STEPS = 4
FLUX_MAX_SEQUENCE_LENGTH = 256
FLUX_STRENGTH = 0.35

def get_pipeline():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires GPU execution.")

    if not 0 <= GPU_INDEX < torch.cuda.device_count():
        raise RuntimeError(
            f"Requested GPU index {GPU_INDEX}, but only {torch.cuda.device_count()} GPU(s) are visible."
        )

    cuda_device = f"cuda:{GPU_INDEX}"

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

    for dtype, dtype_name in dtype_attempts:
        try:
            pipe = DiffusionPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                token=hf_token,
                cache_dir=MODEL_CACHE_DIR,
            ).to(cuda_device)
            pipe.enable_attention_slicing()
            print(
                f"Using {MODEL_ID} on CUDA device {GPU_INDEX}: {torch.cuda.get_device_name(GPU_INDEX)} "
                f"with {dtype_name}"
            )
            print(f"Using local Hugging Face cache at: {MODEL_CACHE_DIR}")
            return pipe, cuda_device
        except Exception as exc:
            last_exc = exc
            print(f"Pipeline init with {dtype_name} failed; trying next dtype.")
            print(f"  Exception: {exc}")

    raise RuntimeError(f"CUDA initialization failed for {MODEL_ID}: {last_exc}") from last_exc

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
    return processed


processed_images = load_processed_images()



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


def call_pipe_with_supported_kwargs(**kwargs):
    """Pass only kwargs supported by the currently loaded pipeline call signature."""
    accepted = set(inspect.signature(pipe.__call__).parameters.keys())
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    return pipe(**filtered_kwargs)


def build_person_only_image(source_image, person_mask):
    """Keep only the masked person and set background to black."""
    if person_mask.mode != "L":
        person_mask = person_mask.convert("L")

    if person_mask.size != source_image.size:
        person_mask = person_mask.resize(source_image.size, Image.NEAREST)

    src_arr = np.array(source_image)
    mask_arr = (np.array(person_mask) > 127).astype(np.uint8)
    cutout_arr = src_arr * mask_arr[:, :, None]
    return Image.fromarray(cutout_arr)


def generate_target_image(masked_person_image, target_prompt):

    width, height = get_target_size(masked_person_image, max_dim=1024)

    if (width, height) != masked_person_image.size:
        masked_person_image = masked_person_image.resize((width, height), Image.LANCZOS)

    prompt = (
        "Edit this image as a realistic candid photograph. "
        "The input image contains only the person on a blank background. "
        "Keep the person unchanged in identity, appearance, pose, and clothing. "
        "Generate a realistic, coherent background around the person. "
        f"Incorporate {target_prompt} naturally into the generated background scene. "
        "Keep exactly one instance of that object category. "
    )
    call_kwargs = {
        "prompt": prompt,
        "image": masked_person_image,
        "guidance_scale": FLUX_GUIDANCE_SCALE,
        "num_inference_steps": FLUX_NUM_INFERENCE_STEPS,
        "max_sequence_length": FLUX_MAX_SEQUENCE_LENGTH,
        "strength": FLUX_STRENGTH,
        "height": height,
        "width": width,
    }
    if device.startswith("cuda"):
        with torch.autocast("cuda"):
            result = call_pipe_with_supported_kwargs(**call_kwargs)
    else:
        result = call_pipe_with_supported_kwargs(**call_kwargs)
    return result.images[0]


with open(selected_images_file, encoding="utf-8") as f:
    for line_idx, line in enumerate(f):
        if line_idx < START_POSITION:
            continue

        line = line.strip()
        if not line:
            continue
        parts = line.split("|", 3)  # filename, person_id, seg_json, attributes
        image_id = parts[0].strip()
        person_id = parts[1].strip() if len(parts) > 1 else ""
        seg_json = parts[2].strip() if len(parts) > 2 else ""
        attributes = parts[3].strip() if len(parts) > 3 else ""

        if image_id in processed_images:
            print(f"Image ID: {image_id} (already processed, skipping)")
            continue

        print(f"Image ID: {image_id}")
        print(f"Person ID: {person_id}")
        print(f"Attributes: {attributes}")

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

        if not seg_json:
            print(f"  Warning: missing segmentation for '{image_id}', skipping")
            continue

        try:
            rle = json.loads(seg_json)
        except Exception as exc:
            print(f"  Warning: invalid segmentation JSON for '{image_id}': {exc}")
            continue

        if not rle:
            print(f"  Warning: empty segmentation for person_id {person_id}, skipping")
            continue

        if isinstance(rle.get("counts"), str):
            rle["counts"] = rle["counts"].encode("utf-8")

        try:
            binary_mask = mask_utils.decode(rle)
        except Exception as exc:
            print(f"  Warning: failed to decode mask for '{image_id}': {exc}")
            continue

        person_mask = Image.fromarray((binary_mask * 255).astype(np.uint8))
        masked_person_image = build_person_only_image(source_image, person_mask)

        generated_images = []
        for target_name, target_prompt in targets:
            print(f"  Generating for target: {target_name}")
            generated_images.append((target_name, generate_target_image(masked_person_image, target_prompt)))

        image_folder = RESULTS_DIR / Path(image_id).stem
        image_folder.mkdir(parents=True, exist_ok=True)

        original_path = image_folder / "original.png"
        source_image.save(original_path)
        print(f"  Saved original image to {original_path}")

        masked_input_path = image_folder / "masked_person_input.png"
        masked_person_image.save(masked_input_path)
        print(f"  Saved masked person input to {masked_input_path}")

        for target_name, generated in generated_images:
            out_path = image_folder / f"{target_name}.png"
            generated.save(out_path)
            print(f"  Saved inpainting for {target_name} to {out_path}")

        num_cols = 1 + len(generated_images)
        fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 6))

        axes[0].imshow(masked_person_image)
        axes[0].set_title("Masked Person Input")
        axes[0].axis("off")

        for i, (target_name, generated) in enumerate(generated_images, start=1):
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

        #plt.show()
