from __future__ import annotations

from typing import Any

from apt_ui.services.api_client import get_json


ACTIVE_TASK_STATUSES = {"pending", "running"}


def list_tasks(
    *,
    task_type: str | None = None,
    active_only: bool = False,
    limit: int | None = None,
    ttl: str = "fast",
    timeout: float = 3,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {}
    if task_type:
        params["type"] = task_type
    if active_only:
        params["active_only"] = "true"
    if limit:
        params["limit"] = limit
    return get_json("/api/tasks", params=params or None, timeout=timeout, default=[], ttl=ttl)


def get_task_detail(task_id: str, *, timeout: float = 5, ttl: str = "slow") -> dict[str, Any]:
    if not task_id:
        return {}
    return get_json(f"/api/tasks/{task_id}", timeout=timeout, default={}, ttl=ttl)


def has_active_tasks(tasks: list[dict[str, Any]]) -> bool:
    return any(task.get("status") in ACTIVE_TASK_STATUSES for task in tasks)
