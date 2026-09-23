"""Persistent analyst review decisions and an append-only audit trail."""

from __future__ import annotations

import datetime as dt
import json
import threading
import uuid
from pathlib import Path


VALID_DECISIONS = {"confirmed", "rejected", "corrected", "insufficient"}


class ReviewStore:
    def __init__(self, review_path: Path, audit_path: Path):
        self.review_path = Path(review_path)
        self.audit_path = Path(audit_path)
        self._lock = threading.Lock()

    def _read(self) -> list[dict]:
        if not self.review_path.exists():
            return []
        try:
            data = json.loads(self.review_path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (OSError, json.JSONDecodeError):
            return []

    def get_reviews(self, result_id: str) -> list[dict]:
        with self._lock:
            rows = self._read()
        return [item for item in rows if item.get("result_id") == result_id]

    def save_review(
        self,
        *,
        result_id: str,
        sample_id: str,
        decision: str,
        corrected_label: str = "",
        note: str = "",
        reviewer: str = "",
    ) -> dict:
        if decision not in VALID_DECISIONS:
            raise ValueError("Invalid review decision")
        if decision == "corrected" and not corrected_label.strip():
            raise ValueError("Corrected label is required")
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        record = {
            "id": str(uuid.uuid4()),
            "result_id": result_id,
            "sample_id": sample_id,
            "decision": decision,
            "corrected_label": corrected_label.strip(),
            "note": note.strip(),
            "reviewer": reviewer.strip() or "anonymous-analyst",
            "updated_at": now,
        }
        with self._lock:
            rows = [
                item for item in self._read()
                if not (item.get("result_id") == result_id and item.get("sample_id") == sample_id)
            ]
            rows.append(record)
            self.review_path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.review_path.with_suffix(".tmp")
            temp.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
            temp.replace(self.review_path)
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "timestamp": now,
                            "event": "review.saved",
                            "actor": record["reviewer"],
                            "result_id": result_id,
                            "sample_id": sample_id,
                            "decision": decision,
                            "corrected_label": record["corrected_label"],
                            "note": record["note"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return record
