"""Small compatibility layer for Streamlit releases before the ``width`` API."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def adapt_width_argument(
    widget: str, kwargs: dict[str, Any], *, legacy: bool
) -> dict[str, Any]:
    """Translate the current ``width='stretch'`` spelling for older Streamlit."""

    adapted = dict(kwargs)
    if not legacy or adapted.get("width") != "stretch":
        return adapted

    adapted.pop("width")
    if widget == "image":
        adapted["use_column_width"] = True
    else:
        adapted["use_container_width"] = True
    return adapted


def install_streamlit_width_compatibility(streamlit: Any) -> None:
    """Patch APIs missing from older Streamlit releases used for deployment."""

    if not hasattr(streamlit, "segmented_control"):
        radio: Callable[..., Any] | None = getattr(streamlit, "radio", None)
        if radio is not None:

            def compatible_segmented_control(
                label: str,
                options: list[Any] | tuple[Any, ...],
                *,
                selection_mode: str = "single",
                **kwargs: Any,
            ) -> Any:
                if selection_mode != "single":
                    raise ValueError("旧版 Streamlit 仅支持单选分段控件。")
                return radio(label, options, horizontal=True, **kwargs)

            streamlit.segmented_control = compatible_segmented_control

    try:
        major, minor, *_ = (int(part) for part in streamlit.__version__.split(".")[:2])
    except (AttributeError, TypeError, ValueError):
        return
    if (major, minor) >= (1, 50) or getattr(streamlit, "_grace_width_compat", False):
        return

    widgets = (
        "button",
        "form_submit_button",
        "download_button",
        "link_button",
        "dataframe",
        "data_editor",
        "plotly_chart",
        "image",
    )
    for widget in widgets:
        original: Callable[..., Any] | None = getattr(streamlit, widget, None)
        if original is None:
            continue

        def compatible(*args: Any, _widget: str = widget, _original: Callable[..., Any] = original, **kwargs: Any) -> Any:
            return _original(*args, **adapt_width_argument(_widget, kwargs, legacy=True))

        setattr(streamlit, widget, compatible)
    streamlit._grace_width_compat = True
