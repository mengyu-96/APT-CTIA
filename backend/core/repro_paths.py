from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REPRO_ROOT = REPO_ROOT / "APT归因复现"
APT_MMF_PROJECT_ROOT = REPRO_ROOT / "基于多模态与多级特征融合的高级持续性威胁行为者归因方法"
APT_MMF_ROOT = APT_MMF_PROJECT_ROOT / "APT-MMF"
TRAIL_ROOT = REPRO_ROOT / "Trail"
TRAIL_SRC = TRAIL_ROOT / "Trail-main" / "src"
APT_ATT_ROOT = REPRO_ROOT / "APT-ATT_project"
MLDSJ_ROOT = REPRO_ROOT / "MLDSJ"
LOCAL_MINILM_ROOT = REPO_ROOT / "all-MiniLM-L6-v2"
HF_MINILM_CACHE_ROOT = Path.home() / ".cache" / "huggingface" / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2"

ATTACK_JSON_CANDIDATES = [
    APT_MMF_PROJECT_ROOT / "external_knowledge" / "attack" / "enterprise-attack.json",
    APT_MMF_PROJECT_ROOT / "external_knowledge" / "cache" / "enterprise-attack.json",
    APT_MMF_ROOT / "external_knowledge" / "attack" / "enterprise-attack.json",
    APT_MMF_ROOT / "external_knowledge" / "cache" / "enterprise-attack.json",
]
