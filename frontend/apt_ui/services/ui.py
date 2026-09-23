"""Shared UI building blocks for the RGAPT frontend.

These helpers centralise the visual language (page headers, cards, toolbars,
status badges) so every page renders with consistent spacing and sizing
instead of copy-pasted inline HTML.
"""

from __future__ import annotations

import contextlib
from typing import Callable, Iterable, Iterator

import streamlit as st
import streamlit.components.v1 as components


# Human-readable, full-word action labels (replaces single-char "删/看").
ACTION_LABELS = {
    "view": "查看",
    "delete": "删除",
    "rename": "重命名",
    "refresh": "刷新",
    "open": "打开",
    "export": "导出",
}


def apply_accessibility_metadata() -> None:
    """Set document language and login autocomplete metadata in the host page."""
    components.html(
        """
        <script>
        const doc = window.parent.document;
        doc.documentElement.lang = "zh-CN";
        const apply = () => {
          const username = doc.querySelector('input[aria-label="用户名"]');
          const password = doc.querySelector('input[aria-label="密码"]');
          const main = doc.querySelector('[data-testid="stAppViewContainer"]');
          const sidebar = doc.querySelector('[data-testid="stSidebar"]');
          if (username) username.setAttribute("autocomplete", "username");
          if (password) password.setAttribute("autocomplete", "current-password");
          if (main) main.setAttribute("role", "main");
          if (sidebar) {
            sidebar.setAttribute("role", "navigation");
            sidebar.setAttribute("aria-label", "主导航");
          }
        };
        apply();
        new MutationObserver(apply).observe(doc.body, {childList: true, subtree: true});

        const page = new URL(window.parent.location.href).searchParams.get("page") || "home";
        const previousPage = window.parent.sessionStorage.getItem("grace-page");
        window.parent.sessionStorage.setItem("grace-page", page);
        if (window.parent.innerWidth <= 768 && previousPage && previousPage !== page) {
          window.parent.setTimeout(() => {
            const collapse = doc.querySelector('[data-testid="stSidebarCollapseButton"] button')
              || doc.querySelector('[data-testid="stSidebarCollapseButton"]');
            const sidebar = doc.querySelector('[data-testid="stSidebar"]');
            const bounds = sidebar ? sidebar.getBoundingClientRect() : null;
            if (collapse && bounds && bounds.x >= -1 && bounds.width > 0) collapse.click();
          }, 350);
        }
        </script>
        """,
        height=0,
        scrolling=False,
    )

_STATUS_META = {
    "pending": ("排队中", "pill-pending"),
    "running": ("运行中", "pill-running"),
    "completed": ("已完成", "pill-completed"),
    "failed": ("失败", "pill-failed"),
}


def page_header(title: str, subtitle: str = "", icon: str = "fa-shield-alt") -> None:
    """Render the standard page header (icon tile + title + uppercase subtitle)."""
    icon_class = icon if icon.startswith("fa") else f"fa-{icon}"
    sub_html = f'<p class="ph-sub">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f"""
        <div class="page-header">
            <div class="ph-icon"><i class="fas {icon_class}"></i></div>
            <div>
                <h1 class="ph-title">{title}</h1>
                {sub_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@contextlib.contextmanager
def section_card(title: str | None = None, icon: str | None = None) -> Iterator[None]:
    """A bordered native container that actually wraps the widgets inside it.

    Usage:
        with section_card("训练配置", icon="fa-sliders"):
            st.selectbox(...)
    """
    container = st.container(border=True)
    with container:
        if title:
            icon_html = f'<i class="fas {icon}"></i>' if icon else ""
            st.markdown(
                f'<div class="section-title">{icon_html}<span>{title}</span></div>',
                unsafe_allow_html=True,
            )
        yield


def status_badge(status: str) -> str:
    """Return HTML for a coloured status pill. Render with unsafe_allow_html."""
    label, cls = _STATUS_META.get(status, (status, "pill-pending"))
    return f'<span class="pill {cls}">{label}</span>'


def status_text(status: str) -> str:
    """Plain-text status label."""
    return _STATUS_META.get(status, (status, ""))[0]


def refresh_button(key: str, *, label: str | None = None, on_click=None) -> bool:
    """Compact, consistently-sized refresh button.

    Returns True when clicked (caller handles cache clear + rerun) unless an
    on_click callback is supplied.
    """
    return st.button(
        f"🔄 {label or ACTION_LABELS['refresh']}",
        key=key,
        on_click=on_click,
        help="重新从后端拉取最新数据",
    )


def confirm_delete(key: str, target: str, on_confirm: Callable[[], object]) -> None:
    """Render a two-step destructive action and execute only after confirmation."""
    state_key = f"confirm_{key}"
    if not st.session_state.get(state_key):
        if st.button(
            f"🗑️ {ACTION_LABELS['delete']}",
            key=key,
            width="stretch",
            help=f"删除{target}",
        ):
            st.session_state[state_key] = True
            st.rerun()
        return

    st.warning(f"确认永久删除{target}？此操作无法撤销。")
    confirm_col, cancel_col = st.columns(2)
    with confirm_col:
        if st.button("永久删除", key=f"{key}_confirm", type="primary", width="stretch"):
            try:
                on_confirm()
            finally:
                st.session_state.pop(state_key, None)
            st.rerun()
    with cancel_col:
        if st.button("取消", key=f"{key}_cancel", width="stretch"):
            st.session_state.pop(state_key, None)
            st.rerun()


def metric_row(items: Iterable[tuple[str, object]]) -> None:
    """Render a horizontal row of st.metric from (label, value) pairs."""
    items = list(items)
    if not items:
        return
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)


def empty_state(message: str, icon: str = "fa-inbox") -> None:
    """A friendly empty / placeholder state."""
    st.markdown(
        f"""
        <div style="text-align:center; padding:2.4rem 1rem; color:var(--text-muted);">
            <i class="fas {icon}" style="font-size:2.4rem; color:var(--primary-color);
               opacity:0.6; margin-bottom:0.8rem;"></i>
            <p style="margin:0;">{message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
