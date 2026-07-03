from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader  # type: ignore

            return "\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)
        except Exception:
            return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _resolve_path(row: Dict[str, str], dataset_dir: Path) -> Path:
    raw_path = Path(str(row.get("file_path", "")).strip())
    candidates = []
    if str(raw_path).strip():
        candidates.append(raw_path)
        candidates.append(dataset_dir / raw_path.name)
    raw_name = str(row.get("raw_name", "")).strip()
    file_type = str(row.get("file_type", "")).strip()
    report_id = str(row.get("report_id", "")).strip()
    if raw_name:
        candidates.append(dataset_dir / f"{raw_name}{file_type}")
    if report_id:
        candidates.append(dataset_dir / f"{report_id}{file_type}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return raw_path


def load_raw_report_text_lookup(processed_dir: Path, dataset_dir: Path) -> Dict[str, str]:
    raw_index = processed_dir / "raw_index.csv"
    if not raw_index.exists():
        return {}
    lookup: Dict[str, str] = {}
    with raw_index.open("r", encoding="utf-8", errors="ignore", newline="") as fp:
        for row in csv.DictReader(fp):
            label = str(row.get("apt_group", "")).strip()
            raw_name = str(row.get("raw_name", "")).strip()
            report_id = str(row.get("report_id", "")).strip()
            path = _resolve_path(row, dataset_dir)
            title = re.sub(rf"(?i)^{re.escape(label)}[_ -]+", "", raw_name or path.stem)
            text = f"{title}\n{_read_text(path)}"
            keys = {report_id, raw_name, path.stem}
            for key in keys:
                if key:
                    lookup[key] = text
    return lookup


def raw_text_svc_candidate(
    report_ids: Sequence[str],
    labels: Sequence[int],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    processed_dir: Path,
    dataset_dir: Path,
    name: str = "raw_report_char_svc_probe",
) -> Dict[str, Any]:
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        return {"name": name, "val_macro_f1": -1.0, "test_pred": np.asarray([], dtype=np.int64)}
    lookup = load_raw_report_text_lookup(processed_dir, dataset_dir)
    texts = [lookup.get(str(report_id), "") for report_id in report_ids]
    if not any(text.strip() for text in texts):
        return {"name": name, "val_macro_f1": -1.0, "test_pred": np.zeros(len(test_idx), dtype=np.int64)}
    y = np.asarray(labels, dtype=np.int64)
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_features=100000,
        sublinear_tf=True,
    )
    word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        min_df=1,
        max_features=70000,
        sublinear_tf=True,
        stop_words="english",
    )
    char_x = char_vectorizer.fit_transform(texts)
    word_x = word_vectorizer.fit_transform(texts)
    feature_sets = [
        ("char35", char_x),
        ("word13", word_x),
        ("word13_char35", hstack([word_x, char_x]).tocsr()),
    ]
    best: Dict[str, Any] = {"name": name, "val_macro_f1": -1.0, "test_pred": np.zeros(len(test_idx), dtype=np.int64)}
    for variant, x in feature_sets:
        for c_value in (0.3, 1.0, 3.0):
            clf = LinearSVC(C=c_value, class_weight="balanced", dual=False, max_iter=5000, random_state=0)
            try:
                clf.fit(x[train_idx], y[train_idx])
            except ValueError:
                continue
            val_pred = clf.predict(x[val_idx])
            val_f1 = float(f1_score(y[val_idx], val_pred, average="macro", zero_division=0))
            # Small validation folds are noisy; require a clear gain before
            # replacing the lower-variance char ngram candidate evaluated first.
            if val_f1 > float(best["val_macro_f1"]) + 0.03:
                best = {
                    "name": name,
                    "c": float(c_value),
                    "val_macro_f1": val_f1,
                    "test_pred": clf.predict(x[test_idx]).astype(np.int64),
                    "test_true": y[test_idx].astype(np.int64),
                    "text_feature_count": int(x.shape[1]),
                    "text_probe_variant": variant,
                }
    return best
