"""Read reproducible dataset split preferences saved by the dataset UI."""

from __future__ import annotations

import json
from pathlib import Path


def load_split_preferences(dataset_path: Path, *, seed: int = 42) -> dict[str, float | int | str]:
    defaults: dict[str, float | int | str] = {
        "train_ratio": 0.7,
        "val_ratio": 0.1,
        "test_ratio": 0.2,
        "seed": int(seed),
        "source": "training-default",
    }
    split_path = Path(dataset_path) / "splits.json"
    if not split_path.exists():
        return defaults
    try:
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        config = payload.get("config") or {}
        train_ratio = float(config.get("train_ratio"))
        val_ratio = float(config.get("val_ratio"))
        test_ratio = float(config.get("test_ratio"))
        split_seed = int(config.get("seed", seed))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return defaults

    ratios = (train_ratio, val_ratio, test_ratio)
    if any(value <= 0 or value >= 1 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        return defaults
    return {
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "seed": split_seed,
        "source": "splits.json",
    }
