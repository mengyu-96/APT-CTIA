"""Plotly chart configuration shared across pages.

Centralises the dark transparent theme, brand colour sequence and margins so
every figure looks consistent and we stop repeating ``update_layout`` blocks.
"""

from __future__ import annotations

from typing import Any


PLOTLY_CHART_CONFIG = {
    "displayModeBar": False,
    "displaylogo": False,
    "responsive": True,
}

# Brand-aligned categorical palette (cyan-led, cool tech feel).
BRAND_SEQUENCE = [
    "#00d4ff",
    "#28e0a6",
    "#6c8cff",
    "#ffc04d",
    "#ff5d7a",
    "#9d7bff",
    "#4dd0e1",
    "#f48fb1",
    "#80cbc4",
    "#ffab66",
]

# Default layout applied to every figure.
BASE_LAYOUT: dict[str, Any] = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#e6edf3", "family": "Segoe UI, Roboto, sans-serif"},
    "margin": {"l": 10, "r": 10, "t": 30, "b": 10},
    "colorway": BRAND_SEQUENCE,
    "legend": {"bgcolor": "rgba(0,0,0,0)", "font": {"color": "#93a4b3"}},
    "xaxis": {"gridcolor": "rgba(74,158,196,0.12)", "zerolinecolor": "rgba(74,158,196,0.2)"},
    "yaxis": {"gridcolor": "rgba(74,158,196,0.12)", "zerolinecolor": "rgba(74,158,196,0.2)"},
}


def apply_layout(fig, **overrides: Any):
    """Apply the shared base layout to a Plotly figure, then any overrides.

    Returns the same figure for chaining. Use ``title`` override sparingly —
    most charts read better with a separate Streamlit subheader.
    """
    fig.update_layout(**BASE_LAYOUT)
    if overrides:
        fig.update_layout(**overrides)
    return fig
