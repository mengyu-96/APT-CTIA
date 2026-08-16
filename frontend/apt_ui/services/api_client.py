from __future__ import annotations

from pathlib import Path
from typing import Any
import os

import requests
from requests.adapters import HTTPAdapter
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:5001")
RequestTimeout = float | tuple[float, float]


def _normalise_params(params: dict[str, Any] | None) -> tuple[tuple[str, Any], ...]:
    if not params:
        return ()
    return tuple(sorted(params.items()))


def _denormalise_params(params: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    return dict(params)


@st.cache_resource(show_spinner=False)
def _get_http_session() -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=0)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"Accept": "application/json"})
    return session


def _request(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: RequestTimeout = 5,
    **kwargs: Any,
) -> requests.Response:
    return _get_http_session().request(
        method.upper(),
        f"{BACKEND_URL}{path}",
        params=params,
        timeout=timeout,
        **kwargs,
    )


# --- Domain-scoped cache invalidation ---------------------------------------
# Each request belongs to a "domain" derived from its path prefix. Bumping a
# domain's salt forces subsequent cached calls for that domain to miss, so a
# task submission only invalidates "tasks" instead of clearing every cache.
_DOMAIN_SALT: dict[str, int] = {}


def _domain_for(path: str) -> str:
    parts = [p for p in path.split("/") if p and p != "api"]
    return parts[0] if parts else "_root"


def _salt_for(path: str) -> int:
    return _DOMAIN_SALT.get(_domain_for(path), 0)


@st.cache_data(ttl=3, show_spinner=False)
def _get_json_fast(path: str, params: tuple[tuple[str, Any], ...], timeout: float, salt: int) -> Any:
    del salt
    response = _request("GET", path, params=_denormalise_params(params), timeout=timeout)
    if response.status_code == 200:
        return response.json()
    return None


@st.cache_data(ttl=30, show_spinner=False)
def _get_json_default(path: str, params: tuple[tuple[str, Any], ...], timeout: float, salt: int) -> Any:
    del salt
    response = _request("GET", path, params=_denormalise_params(params), timeout=timeout)
    if response.status_code == 200:
        return response.json()
    return None


@st.cache_data(ttl=300, show_spinner=False)
def _get_json_slow(path: str, params: tuple[tuple[str, Any], ...], timeout: float, salt: int) -> Any:
    del salt
    response = _request("GET", path, params=_denormalise_params(params), timeout=timeout)
    if response.status_code == 200:
        return response.json()
    return None


@st.cache_data(ttl=300, show_spinner=False)
def _read_local_bytes(path: str, mtime_ns: int, size: int) -> bytes | None:
    del mtime_ns, size
    return Path(path).read_bytes()


@st.cache_data(ttl=300, show_spinner=False)
def _get_bytes(path: str, params: tuple[tuple[str, Any], ...], timeout: float) -> bytes | None:
    response = _request("GET", path, params=_denormalise_params(params), timeout=timeout)
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
    salt = _salt_for(path)
    try:
        if ttl == "fast":
            data = _get_json_fast(path, key, timeout, salt)
        elif ttl == "slow":
            data = _get_json_slow(path, key, timeout, salt)
        else:
            data = _get_json_default(path, key, timeout, salt)
        return default if data is None else data
    except Exception:
        return default


@st.cache_data(ttl=30, show_spinner=False)
def get_runtime_config() -> dict[str, Any]:
    default_config = {
        "demo_mode": False,
        "preprocessing_enabled": True,
        "inference_enabled": True,
        "training_enabled": True,
        "training_ui_enabled": True,
        "training_report_source_enabled": True,
    }
    data = get_json("/api/runtime_config", timeout=1.5, default=default_config, ttl="default")
    return data if isinstance(data, dict) else default_config


@st.cache_data(ttl=30, show_spinner=False)
def get_dashboard_counts() -> dict[str, int]:
    return {
        "datasets": len(get_json("/api/datasets", timeout=1.5, default=[])),
        "models": len(get_json("/api/models", timeout=1.5, default=[])),
        "results": len(get_json("/api/attribution_results", timeout=1.5, default=[])),
    }


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


def get_binary(
    path: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: float = 30,
) -> bytes | None:
    if not path:
        return None
    return _get_bytes(path, _normalise_params(params), timeout)


def request(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: Any = None,
    data: Any = None,
    files: Any = None,
    timeout: RequestTimeout = 5,
) -> requests.Response:
    return _request(
        method,
        path,
        params=params,
        json=json_body,
        data=data,
        files=files,
        timeout=timeout,
    )


def build_upload_payload(field_name: str, files: list[Any]) -> list[tuple[str, tuple[str, Any, str]]]:
    payload: list[tuple[str, tuple[str, Any, str]]] = []
    for file in files:
        try:
            file.seek(0)
        except Exception:
            pass
        payload.append((field_name, (file.name, file, file.type or "application/octet-stream")))
    return payload


def invalidate(*domains: str) -> None:
    """Invalidate cached GETs for specific domains only.

    A domain is the first path segment after ``/api`` (e.g. ``"tasks"``,
    ``"datasets"``, ``"models"``, ``"attribution_results"``). Bumping the salt
    makes subsequent cached calls for that domain miss, leaving unrelated
    caches (and their warm data) intact. The dashboard summary depends on
    several domains, so it is always refreshed.

    Pass no arguments to fall back to a full clear.
    """
    if not domains:
        clear_api_cache()
        return
    for domain in domains:
        _DOMAIN_SALT[domain] = _DOMAIN_SALT.get(domain, 0) + 1
    get_dashboard_counts.clear()


def clear_api_cache() -> None:
    _get_json_fast.clear()
    _get_json_default.clear()
    _get_json_slow.clear()
    _read_local_bytes.clear()
    _get_bytes.clear()
    get_dashboard_counts.clear()
