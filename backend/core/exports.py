"""Portable JSON and STIX 2.1 exports for attribution results."""

from __future__ import annotations

import datetime as dt
import json
import uuid
from typing import Any


def _stix_id(object_type: str, stable_value: str) -> str:
    return f"{object_type}--{uuid.uuid5(uuid.NAMESPACE_URL, stable_value)}"


def _timestamp(value: Any = None) -> str:
    if value:
        text = str(value).replace(" ", "T")
        if not text.endswith("Z"):
            text += "Z"
        return text
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def build_stix_bundle(result_id: str, result: dict[str, Any]) -> dict[str, Any]:
    created = _timestamp(result.get("created"))
    identity_id = _stix_id("identity", "GRACE APT attribution system")
    objects: list[dict[str, Any]] = [
        {
            "type": "identity",
            "spec_version": "2.1",
            "id": identity_id,
            "created": created,
            "modified": created,
            "name": "GRACE APT Attribution System",
            "identity_class": "system",
        }
    ]
    actor_ids: dict[str, str] = {}
    for sample in result.get("results", []) or []:
        label = str(sample.get("model_prediction") or sample.get("predicted_label") or "Unknown")
        if label not in actor_ids and label not in {"Unknown", "证据不足", "需人工复核"}:
            actor_id = _stix_id("threat-actor", label)
            actor_ids[label] = actor_id
            objects.append(
                {
                    "type": "threat-actor",
                    "spec_version": "2.1",
                    "id": actor_id,
                    "created": created,
                    "modified": created,
                    "created_by_ref": identity_id,
                    "name": label,
                    "threat_actor_types": ["unknown"],
                }
            )
        confidence = float(sample.get("confidence", 0) or 0)
        status = (sample.get("decision") or {}).get("status", "unknown")
        note_id = _stix_id("note", f"{result_id}:{sample.get('report_id')}")
        note = {
            "type": "note",
            "spec_version": "2.1",
            "id": note_id,
            "created": created,
            "modified": created,
            "created_by_ref": identity_id,
            "abstract": f"APT attribution for {sample.get('report_id', 'unknown')}",
            "content": json.dumps(
                {
                    "result_id": result_id,
                    "report_id": sample.get("report_id"),
                    "prediction": label,
                    "confidence": confidence,
                    "decision_status": status,
                    "top3": sample.get("top3", []),
                },
                ensure_ascii=False,
            ),
            "object_refs": [actor_ids[label]] if label in actor_ids else [identity_id],
        }
        objects.append(note)

    return {
        "type": "bundle",
        "id": _stix_id("bundle", result_id),
        "objects": objects,
    }
