# GitHub preparation

This repository is a sanitized source-only copy of the research export. The original export was left intact.

## Included

- Fourteen top-level research Python scripts.
- General evaluation questions and the location attribute taxonomy.
- Setup documentation, dependency inventories, environment-variable examples, ignore rules, and a dependency-free repository checker.

## Excluded from this copy

- Generated images, comparison figures, analysis panels, logs, and caches.
- Per-image annotations, segmentation selections, and generated statistical JSON/CSV outputs.
- Empty religious-site output files.
- `old_stuff/`, containing earlier scripts and experiments.
- `install_flux.py`, a machine-specific model-cache helper; generation scripts can download/cache models directly.

The original export remains the source for historical results. Exclusion from this copy does not delete those files.

## Source changes

- Replaced literal Hugging Face tokens with `HF_TOKEN` environment lookups, including an SDXL call that previously ignored the environment token.
- Replaced private dataset/cache paths with `FACET_ROOT` and `MODEL_CACHE_DIR` settings.
- Made GPU selection configurable with a default of 0 and rejected negative device indices.
- Changed the experimental scripts' starting offset from 100 to configurable `START_POSITION`, defaulting to 0.
- Made the Ollama URL configurable and normalized its default to localhost port 11434.
- Allowed the FLUX Fill script to download weights unless `HF_HUB_OFFLINE=1`.
- Fixed the missing comma between religious-site and cultural-site prompts.

These changes mean this source is not byte-for-byte identical to the version associated with the historical outputs. No statistical conclusions were recomputed.

## Before publication

Revoke credentials exposed in the original export and use fresh credentials through the environment. This copy cannot invalidate them. Publish only this clean repository, not the original export.

Full reproduction still requires dataset access, model weights, CUDA resources, Ollama, and per-person reference labels. The original ground-truth conversion script and dependency lockfile were not provided. No license is added without the owner's choice.
