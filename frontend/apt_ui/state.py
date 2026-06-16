from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import streamlit as st


@dataclass(frozen=True)
class SampleMeta:
    name: str
    size_bytes: int
    file_type: str
    sha256: str
    md5: str


def ensure_session_state() -> None:
    if "samples" not in st.session_state:
        st.session_state["samples"] = []
    if "selected_sha256" not in st.session_state:
        st.session_state["selected_sha256"] = None
    if "feature_cache" not in st.session_state:
        st.session_state["feature_cache"] = {}


def get_samples() -> list[dict[str, Any]]:
    ensure_session_state()
    return st.session_state["samples"]


def upsert_sample(sample: dict[str, Any]) -> None:
    ensure_session_state()
    samples: list[dict[str, Any]] = st.session_state["samples"]
    sha256 = sample.get("sha256")
    for i, s in enumerate(samples):
        if s.get("sha256") == sha256:
            samples[i] = sample
            return
    samples.append(sample)


def set_selected_sha256(sha256: str | None) -> None:
    ensure_session_state()
    st.session_state["selected_sha256"] = sha256


def get_selected_sha256() -> str | None:
    ensure_session_state()
    return st.session_state["selected_sha256"]

