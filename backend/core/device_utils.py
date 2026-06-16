from __future__ import annotations

from typing import Dict

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def cuda_available() -> bool:
    return bool(torch is not None and torch.cuda.is_available())


def preferred_sentence_transformer_device() -> str:
    return "cuda" if cuda_available() else "cpu"


def preferred_xgboost_params() -> Dict[str, object]:
    if not cuda_available():
        return {}
    return {
        "tree_method": "hist",
        "device": "cuda",
    }


def ctgan_cuda_enabled() -> bool:
    return cuda_available()
