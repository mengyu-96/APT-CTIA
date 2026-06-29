"""Unified task panel used by upload / training / inference pages.

Replaces three near-identical implementations of task sorting, status text,
row rendering and fragment polling. Pages supply a title function and a list
of row-level actions; everything else (polling, refresh, empty/active hints,
cache invalidation) is handled here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import streamlit as st

from apt_ui.services import ui
from apt_ui.services.api_client import invalidate, request
from apt_ui.services.tasks import has_active_tasks, list_tasks


ACTIVE_STATUSES = {"pending", "running"}


@dataclass
class TaskAction:
    """A button rendered on each task row.

    handler(task) -> bool: return True on success (triggers toast + rerun).
    enabled(task) -> bool: whether the button is clickable for this task.
    """

    label: str
    handler: Callable[[dict], bool]
    enabled: Callable[[dict], bool] = lambda task: True
    icon: str = ""


def delete_task(task_id: str, *, timeout: float = 10) -> bool:
    if not task_id:
        return False
    try:
        return request("DELETE", f"/api/tasks/{task_id}", timeout=timeout).status_code == 200
    except Exception:
        return False


def default_delete_action() -> TaskAction:
    """Standard 'delete' action available once a task is completed/failed."""

    def _handler(task: dict) -> bool:
        return delete_task(task.get("id", ""))

    return TaskAction(
        label=ui.ACTION_LABELS["delete"],
        handler=_handler,
        enabled=lambda task: task.get("status") in {"completed", "failed"},
        icon="🗑️",
    )


def _sort_tasks(tasks: list[dict]) -> list[dict]:
    def key(task: dict) -> tuple[int, str]:
        rank = 0 if task.get("status") in ACTIVE_STATUSES else 1
        return rank, str(task.get("created", ""))

    return sorted(tasks, key=key)


def _render_row(
    task: dict,
    *,
    key_prefix: str,
    title_fn: Callable[[dict], str],
    subtitle_fn: Callable[[dict], str] | None,
    actions: list[TaskAction],
    active_message: str,
) -> None:
    status = task.get("status", "unknown")
    is_active = status in ACTIVE_STATUSES

    head_col, action_col = st.columns([6, max(1, len(actions))], gap="small")
    with head_col:
        title_html = f"**{title_fn(task)}**"
        st.markdown(
            f'{title_html} &nbsp; {ui.status_badge(status)}',
            unsafe_allow_html=True,
        )
        sub = subtitle_fn(task) if subtitle_fn else None
        meta = f"{task.get('created', '-')} · `{task.get('id', '')[:8]}`"
        st.caption(f"{sub} · {meta}" if sub else meta)
        if is_active:
            progress = int(task.get("progress", 0) or 0)
            st.progress(max(0, min(progress, 100)) / 100)
            st.caption(task.get("message") or active_message)
        if task.get("error"):
            st.error(task["error"])

    with action_col:
        btn_cols = st.columns(len(actions)) if len(actions) > 1 else [action_col]
        for action, col in zip(actions, btn_cols):
            with col:
                label = f"{action.icon} {action.label}".strip()
                if st.button(
                    label,
                    key=f"{key_prefix}_{action.label}_{task.get('id')}",
                    disabled=not action.enabled(task),
                    width="stretch",
                ):
                    if action.handler(task):
                        invalidate("tasks", "datasets", "models", "attribution_results")
                        st.toast(f"{action.label}成功", icon="✅")
                        st.rerun()
                    else:
                        st.warning(f"{action.label}失败")

    st.divider()


def _render_list(
    tasks: list[dict],
    *,
    key_prefix: str,
    title_fn,
    subtitle_fn,
    actions,
    empty_message: str,
    active_message: str,
) -> None:
    if not tasks:
        ui.empty_state(empty_message, icon="fa-list-check")
        return

    ordered = _sort_tasks(tasks)
    if any(t.get("status") in ACTIVE_STATUSES for t in ordered):
        st.caption("运行中的任务固定显示在顶部，列表会自动刷新。")
    else:
        st.caption("空闲时不自动刷新；提交、删除或手动刷新后更新。")

    for task in ordered:
        _render_row(
            task,
            key_prefix=key_prefix,
            title_fn=title_fn,
            subtitle_fn=subtitle_fn,
            actions=actions,
            active_message=active_message,
        )


def render_task_panel(
    task_type: str,
    *,
    title_fn: Callable[[dict], str],
    subtitle_fn: Callable[[dict], str] | None = None,
    actions: list[TaskAction] | None = None,
    key_prefix: str,
    poll_state_key: str,
    limit: int = 40,
    poll_seconds: int = 3,
    empty_message: str = "暂无任务记录。",
    active_message: str = "任务执行中…",
    refresh: bool = True,
) -> None:
    """Render a task list with adaptive polling.

    Idle -> no polling (default-TTL read). Active tasks -> a parallel fragment
    refreshes just this list every ``poll_seconds`` without blocking the rest
    of the page, and stops polling automatically once everything settles.
    """
    actions = actions or [default_delete_action()]

    if refresh:
        if ui.refresh_button(f"refresh_{key_prefix}"):
            invalidate("tasks")
            st.rerun()

    def _fetch(ttl: str) -> list[dict]:
        return list_tasks(task_type=task_type, limit=limit, timeout=3, ttl=ttl)

    initial = _fetch("fast")
    should_poll = has_active_tasks(initial) or st.session_state.get(poll_state_key, False)

    if should_poll:
        @st.fragment(run_every=poll_seconds)
        def _poll() -> None:
            tasks = _fetch("fast")
            if not has_active_tasks(tasks):
                st.session_state[poll_state_key] = False
                invalidate("tasks", "datasets", "models", "attribution_results")
                st.rerun()
                return
            st.session_state[poll_state_key] = True
            _render_list(
                tasks,
                key_prefix=f"{key_prefix}_live",
                title_fn=title_fn,
                subtitle_fn=subtitle_fn,
                actions=actions,
                empty_message=empty_message,
                active_message=active_message,
            )

        _poll()
    else:
        _render_list(
            _fetch("default"),
            key_prefix=f"{key_prefix}_static",
            title_fn=title_fn,
            subtitle_fn=subtitle_fn,
            actions=actions,
            empty_message=empty_message,
            active_message=active_message,
        )
