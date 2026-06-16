from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_MINILM_ROOT = REPO_ROOT / "all-MiniLM-L6-v2"
HF_MINILM_CACHE_ROOT = Path.home() / ".cache" / "huggingface" / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2"

# Optional MITRE ATT&CK JSON locations for attribution explanations.
# The release branch intentionally does not include reproduction/experiment assets.
ATTACK_JSON_CANDIDATES: list[Path] = []
