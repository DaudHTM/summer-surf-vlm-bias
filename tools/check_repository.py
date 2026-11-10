"""Check publishable source without importing models or exposing matched secrets."""

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "GitHub token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    "API secret": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "AWS access key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "literal credential assignment": re.compile(
        r'''(?i)\b(?:hf_token|api_key|api_secret|access_token|password)\s*=\s*["'][^"'\n]{8,}["']'''
    ),
}


def main():
    errors = []
    checked = 0
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if any(part in {".git", ".venv", "venv", "__pycache__", ".cache", "data", "models"} for part in relative.parts):
            continue
        if not path.is_file() or path.suffix not in {".py", ".json", ".md", ".txt", ".yml", ".yaml", ".example"}:
            continue
        text = path.read_text(encoding="utf-8-sig")
        checked += 1
        for label, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                errors.append(f"{relative}:{line}: possible {label} (value withheld)")
        try:
            if path.suffix == ".py":
                compile(text, str(relative), "exec")
            elif path.suffix == ".json":
                json.loads(text)
        except (SyntaxError, ValueError) as exc:
            errors.append(f"{relative}: invalid syntax ({type(exc).__name__})")

    tree = ast.parse((ROOT / "inpainting.py").read_text(encoding="utf-8"))
    places = next(ast.literal_eval(node.value) for node in tree.body
                  if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "places" for target in node.targets))
    if len(places) != 9 or "religious site" not in places or "cultural site" not in places:
        errors.append("inpainting.py: expected nine separate location prompts")
    if errors:
        print("\n".join(errors))
        raise SystemExit(1)
    print(f"PASS: {checked} text files checked; Python/JSON syntax, credential patterns, and nine locations verified.")


if __name__ == "__main__":
    main()
