from __future__ import annotations

import os
from typing import Any


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


DEMO_MODE = _env_bool("DEMO_MODE", False)
ENABLE_PREPROCESSING = _env_bool("ENABLE_PREPROCESSING", True)
ENABLE_INFERENCE = _env_bool("ENABLE_INFERENCE", True)
ENABLE_TRAINING = _env_bool("ENABLE_TRAINING", not DEMO_MODE)
SHOW_TRAINING_UI = _env_bool("SHOW_TRAINING_UI", ENABLE_TRAINING)
ALLOW_TRAINING_REPORT_SOURCE = _env_bool("ALLOW_TRAINING_REPORT_SOURCE", ENABLE_TRAINING)


def build_runtime_config() -> dict[str, Any]:
    return {
        "demo_mode": DEMO_MODE,
        "preprocessing_enabled": ENABLE_PREPROCESSING,
        "inference_enabled": ENABLE_INFERENCE,
        "training_enabled": ENABLE_TRAINING,
        "training_ui_enabled": SHOW_TRAINING_UI and ENABLE_TRAINING,
        "training_report_source_enabled": ALLOW_TRAINING_REPORT_SOURCE and ENABLE_TRAINING,
    }
