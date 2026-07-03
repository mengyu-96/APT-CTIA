from __future__ import annotations

import csv
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def parse_iso_date(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text or text.upper() in {"UNKNOWN", "NONE", "NAN"}:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m-%d-%Y", "%m/%d/%Y", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def load_report_dates(raw_index_path: str | Path | None) -> Dict[str, str]:
    if raw_index_path is None:
        return {}
    path = Path(raw_index_path)
    if not path.exists():
        return {}
    rows: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as fp:
        for row in csv.DictReader(fp):
            report_id = str(row.get("report_id", "")).strip()
            if not report_id:
                continue
            rows[report_id] = str(row.get("published_date", "")).strip()
    return rows


def _ratio_counts(n_items: int, train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    if n_items <= 0:
        return 0, 0, 0
    total = train_ratio + val_ratio + test_ratio
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    test_count = max(1, int(round(n_items * test_ratio)))
    val_count = max(1, int(round(n_items * val_ratio)))
    train_count = n_items - test_count - val_count

    if train_count < 1:
        deficit = 1 - train_count
        if val_count >= test_count and val_count - deficit >= 1:
            val_count -= deficit
        else:
            test_count = max(1, test_count - deficit)
        train_count = n_items - test_count - val_count

    while train_count + val_count + test_count > n_items:
        if val_count > 1:
            val_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            break
        train_count = n_items - test_count - val_count

    return train_count, val_count, test_count


def _simple_random_split(
    n_items: int,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_count, val_count, test_count = _ratio_counts(n_items, train_ratio, val_ratio, test_ratio)
    rng = np.random.default_rng(seed)
    indices = np.arange(n_items)
    rng.shuffle(indices)
    train_end = train_count
    val_end = train_end + val_count
    return indices[:train_end], indices[train_end:val_end], indices[val_end:val_end + test_count]


def split_indices_stratified(
    labels: Sequence[int],
    seed: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    y = np.asarray(labels, dtype=np.int64)
    indices = np.arange(len(y))
    train_count, val_count, test_count = _ratio_counts(len(y), train_ratio, val_ratio, test_ratio)
    metadata: Dict[str, Any] = {
        "split_mode_requested": "stratified",
        "split_mode_resolved": "stratified",
        "split_train_size": 0,
        "split_val_size": 0,
        "split_test_size": 0,
    }
    if len(y) == 0:
        return indices, indices, indices, metadata

    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_count, random_state=seed)
        train_val_idx, test_idx = next(sss.split(indices, y))
        train_val_labels = y[train_val_idx]
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_count, random_state=seed)
        train_rel, val_rel = next(sss2.split(train_val_idx, train_val_labels))
        train_idx = train_val_idx[train_rel]
        val_idx = train_val_idx[val_rel]
    except ValueError:
        train_idx, val_idx, test_idx = _simple_random_split(len(y), seed, train_ratio, val_ratio, test_ratio)
        metadata["split_fallback_reason"] = "stratified_split_unavailable"

    metadata.update(
        split_train_size=int(len(train_idx)),
        split_val_size=int(len(val_idx)),
        split_test_size=int(len(test_idx)),
    )
    return train_idx, val_idx, test_idx, metadata


def split_indices_time_based(
    report_ids: Sequence[str],
    labels: Sequence[int],
    report_dates: Mapping[str, str],
    seed: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    train_count, val_count, test_count = _ratio_counts(len(report_ids), train_ratio, val_ratio, test_ratio)
    dated: list[tuple[int, date]] = []
    undated: list[int] = []
    for idx, report_id in enumerate(report_ids):
        parsed = parse_iso_date(report_dates.get(str(report_id), ""))
        if parsed is None:
            undated.append(idx)
        else:
            dated.append((idx, parsed))

    metadata: Dict[str, Any] = {
        "split_mode_requested": "time_based",
        "split_mode_resolved": "time_based",
        "split_train_size": 0,
        "split_val_size": 0,
        "split_test_size": 0,
        "split_dated_reports": int(len(dated)),
        "split_undated_reports": int(len(undated)),
    }

    if len(dated) < val_count + test_count + 1:
        train_idx, val_idx, test_idx, fallback_meta = split_indices_stratified(
            labels=labels,
            seed=seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        fallback_meta["split_mode_requested"] = "time_based"
        fallback_meta["split_fallback_reason"] = "insufficient_dated_reports"
        fallback_meta["split_dated_reports"] = int(len(dated))
        fallback_meta["split_undated_reports"] = int(len(undated))
        return train_idx, val_idx, test_idx, fallback_meta

    dated.sort(key=lambda item: (item[1], item[0]))
    ordered_dated = [idx for idx, _ in dated]
    val_start = max(0, len(ordered_dated) - (val_count + test_count))
    test_start = max(val_start, len(ordered_dated) - test_count)

    val_idx = np.asarray(ordered_dated[val_start:test_start], dtype=np.int64)
    test_idx = np.asarray(ordered_dated[test_start:], dtype=np.int64)
    train_idx = np.asarray(undated + ordered_dated[:val_start], dtype=np.int64)

    metadata.update(
        split_train_size=int(len(train_idx)),
        split_val_size=int(len(val_idx)),
        split_test_size=int(len(test_idx)),
        split_train_date_end=str(dated[val_start - 1][1]) if val_start > 0 else "",
        split_val_date_start=str(dated[val_start][1]) if len(val_idx) else "",
        split_test_date_start=str(dated[test_start][1]) if len(test_idx) else "",
    )
    return train_idx, val_idx, test_idx, metadata


def build_report_level_split(
    report_ids: Sequence[str],
    labels: Sequence[int],
    seed: int,
    split_mode: str = "stratified",
    raw_index_path: str | Path | None = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    normalized_mode = str(split_mode or "stratified").strip().lower()
    if normalized_mode == "time_based":
        return split_indices_time_based(
            report_ids=report_ids,
            labels=labels,
            report_dates=load_report_dates(raw_index_path),
            seed=seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
    return split_indices_stratified(
        labels=labels,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
