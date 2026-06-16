from __future__ import annotations

from pathlib import Path
from typing import Any
import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:5001")


def _normalise_params(params: dict[str, Any] | None) -> tuple[tuple[str, Any], ...]:
    if not params:
        return ()
    return tuple(sorted(params.items()))


def _denormalise_params(params: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    return dict(params)


@st.cache_data(ttl=2, show_spinner=False)
def _get_json_fast(path: str, params: tuple[tuple[str, Any], ...], timeout: float) -> Any:
    response = requests.get(f"{BACKEND_URL}{path}", params=_denormalise_params(params), timeout=timeout)
    if response.status_code == 200:
        return response.json()
    return None


@st.cache_data(ttl=30, show_spinner=False)
def _get_json_default(path: str, params: tuple[tuple[str, Any], ...], timeout: float) -> Any:
    response = requests.get(f"{BACKEND_URL}{path}", params=_denormalise_params(params), timeout=timeout)
    if response.status_code == 200:
        return response.json()
    return None


@st.cache_data(ttl=300, show_spinner=False)
def _get_json_slow(path: str, params: tuple[tuple[str, Any], ...], timeout: float) -> Any:
    response = requests.get(f"{BACKEND_URL}{path}", params=_denormalise_params(params), timeout=timeout)
    if response.status_code == 200:
        return response.json()
    return None


@st.cache_data(ttl=300, show_spinner=False)
def _read_local_bytes(path: str, mtime_ns: int, size: int) -> bytes | None:
    del mtime_ns, size
    return Path(path).read_bytes()


@st.cache_data(ttl=300, show_spinner=False)
def _get_bytes(path: str, params: tuple[tuple[str, Any], ...], timeout: float) -> bytes | None:
    response = requests.get(f"{BACKEND_URL}{path}", params=_denormalise_params(params), timeout=timeout)
    if response.status_code == 200 and response.content:
        return response.content
    return None


def get_json(
    path: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: float = 5,
    default: Any = None,
    ttl: str = "default",
) -> Any:
    """Cached JSON GET helper.

    ttl values:
    - fast: task/progress polling paths
    - default: registry/list views
    - slow: mostly immutable result/detail payloads
    """
    key = _normalise_params(params)
    try:
        if ttl == "fast":
            data = _get_json_fast(path, key, timeout)
        elif ttl == "slow":
            data = _get_json_slow(path, key, timeout)
        else:
            data = _get_json_default(path, key, timeout)
        return default if data is None else data
    except Exception:
        return default


def get_artifact_bytes(path: str, *, timeout: float = 5) -> bytes | None:
    if not path:
        return None

    try:
        local_path = Path(path)
        if local_path.exists() and local_path.is_file():
            stat = local_path.stat()
            return _read_local_bytes(str(local_path), stat.st_mtime_ns, stat.st_size)
    except Exception:
        pass

    return _get_bytes("/api/artifact", _normalise_params({"path": path}), timeout)


def clear_api_cache() -> None:
    _get_json_fast.clear()
    _get_json_default.clear()
    _get_json_slow.clear()
    _read_local_bytes.clear()
    _get_bytes.clear()
