"""Authentication configuration, password hashing, login, and registration."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import datetime as dt
import hashlib
import hmac
import json
import os
from pathlib import Path
import threading
from typing import Mapping
import unicodedata


_TRUE_VALUES = {"1", "true", "yes", "on"}
_INSECURE_PASSWORDS = {
    "admin",
    "password",
    "change-this-password",
    "replace-with-a-strong-password",
    "replace-with-a-unique-long-password",
}
_PBKDF2_ITERATIONS = 600_000
_DUMMY_SALT = b"grace-auth-dummy-salt-for-timing"


def _default_users_file() -> str:
    project_root = Path(__file__).resolve().parents[3]
    return str(project_root / "auth_data" / "users.json")


@dataclass(frozen=True)
class AuthConfig:
    enabled: bool
    username: str = ""
    password: str = ""
    session_ttl_seconds: int = 28_800
    allow_registration: bool = True
    users_file: str = ""
    error: str | None = None


def _normalize_username(username: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(username)).strip()
    if not 3 <= len(normalized) <= 32:
        raise ValueError("用户名长度必须为 3–32 个字符。")
    if not all(char.isalnum() or char in {"_", "-", "."} for char in normalized):
        raise ValueError("用户名只能包含文字、数字、下划线、短横线和点。")
    return normalized


def _validate_password(password: str) -> None:
    if len(password) < 12:
        raise ValueError("密码至少 12 个字符。")
    if password.casefold() in _INSECURE_PASSWORDS:
        raise ValueError("不能使用默认密码或常见弱密码。")


def _hash_password(password: str, salt: bytes, iterations: int) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)


class UserStore:
    """File-backed login store using salted PBKDF2-SHA256 password hashes."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def _read(self) -> dict:
        if not self.path.exists():
            return {"version": 1, "users": {}}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"version": 1, "users": {}}
        if not isinstance(payload, dict) or not isinstance(payload.get("users"), dict):
            return {"version": 1, "users": {}}
        return payload

    def _write(self, payload: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(f".{self.path.name}.tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.path)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    @staticmethod
    def _new_record(username: str, password: str, source: str) -> dict[str, str | int]:
        salt = os.urandom(32)
        password_hash = _hash_password(password, salt, _PBKDF2_ITERATIONS)
        return {
            "username": username,
            "salt": base64.b64encode(salt).decode("ascii"),
            "password_hash": base64.b64encode(password_hash).decode("ascii"),
            "iterations": _PBKDF2_ITERATIONS,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source": source,
        }

    def register(self, username: str, password: str) -> dict[str, str]:
        normalized = _normalize_username(username)
        _validate_password(password)
        key = normalized.casefold()
        with self._lock:
            payload = self._read()
            users = payload["users"]
            if key in users:
                raise ValueError("该用户名已存在。")
            record = self._new_record(normalized, password, "self-registration")
            users[key] = record
            self._write(payload)
        return {"username": normalized, "created_at": record["created_at"]}

    def import_legacy_user(self, username: str, password: str) -> dict[str, str]:
        """Import one known legacy account without weakening public registration rules.

        This path is intended for an administrator-controlled, one-time migration.  It
        accepts an existing weak password so an upgrade does not lock out the owner,
        while still storing only a salted PBKDF2 hash.
        """

        normalized = _normalize_username(username)
        if not password:
            raise ValueError("旧账号密码不能为空。")
        key = normalized.casefold()
        with self._lock:
            payload = self._read()
            users = payload["users"]
            if key in users:
                raise ValueError("该用户名已存在。")
            record = self._new_record(normalized, password, "legacy-migration")
            users[key] = record
            self._write(payload)
        return {"username": normalized, "created_at": str(record["created_at"])}

    def verify(self, username: str, password: str) -> bool:
        try:
            key = _normalize_username(username).casefold()
        except ValueError:
            key = ""
        with self._lock:
            record = self._read()["users"].get(key)
        if record:
            try:
                salt = base64.b64decode(record["salt"], validate=True)
                expected = base64.b64decode(record["password_hash"], validate=True)
                iterations = int(record.get("iterations", _PBKDF2_ITERATIONS))
            except (KeyError, TypeError, ValueError):
                return False
        else:
            salt = _DUMMY_SALT
            expected = _hash_password("invalid-user-password", salt, _PBKDF2_ITERATIONS)
            iterations = _PBKDF2_ITERATIONS
        actual = _hash_password(password, salt, iterations)
        return bool(record) and hmac.compare_digest(actual, expected)


def load_auth_config(environment: Mapping[str, str]) -> AuthConfig:
    enabled = environment.get("ENABLE_UI_AUTH", "true").strip().lower() in _TRUE_VALUES
    allow_registration = (
        environment.get("ENABLE_SELF_REGISTRATION", "true").strip().lower() in _TRUE_VALUES
    )
    users_file = environment.get("AUTH_USERS_FILE", "").strip() or _default_users_file()
    if not enabled:
        return AuthConfig(
            enabled=False,
            allow_registration=allow_registration,
            users_file=users_file,
        )

    username = environment.get("AUTH_USERNAME", "").strip()
    password = environment.get("AUTH_PASSWORD", "")
    try:
        session_ttl_seconds = int(environment.get("AUTH_SESSION_TTL_SECONDS", "28800"))
    except ValueError:
        session_ttl_seconds = 0
    base = {
        "enabled": True,
        "username": username,
        "password": password,
        "session_ttl_seconds": session_ttl_seconds,
        "allow_registration": allow_registration,
        "users_file": users_file,
    }
    if session_ttl_seconds < 300:
        return AuthConfig(**base, error="AUTH_SESSION_TTL_SECONDS 必须是至少 300 秒的整数。")
    if bool(username) != bool(password):
        return AuthConfig(**base, error="AUTH_USERNAME 和 AUTH_PASSWORD 必须同时配置。")
    if not username and not allow_registration:
        return AuthConfig(**base, error="未配置登录账号，并且用户注册已关闭。")
    if password:
        try:
            _validate_password(password)
        except ValueError:
            return AuthConfig(**base, error="检测到默认或示例密码，请设置新的登录凭据。")
    return AuthConfig(**base)


def verify_credentials(
    username: str,
    password: str,
    config: AuthConfig,
    user_store: UserStore | None = None,
) -> bool:
    if not config.enabled or config.error:
        return False
    bootstrap_match = bool(config.username) and (
        hmac.compare_digest(username, config.username)
        and hmac.compare_digest(password, config.password)
    )
    return bootstrap_match or bool(user_store and user_store.verify(username, password))
