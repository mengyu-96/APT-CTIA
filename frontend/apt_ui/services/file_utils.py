from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class FileDigests:
    md5: str
    sha256: str


def compute_digests(data: bytes) -> FileDigests:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    md5.update(data)
    sha256.update(data)
    return FileDigests(md5=md5.hexdigest(), sha256=sha256.hexdigest())


def guess_file_type(filename: str, data: bytes) -> str:
    # Best-effort: use python-magic if available; fall back to signature/name.
    try:
        import magic  # type: ignore

        m = magic.Magic(mime=True)
        mime = m.from_buffer(data)
        return f"{mime} (magic)"
    except Exception:
        lower = filename.lower()
        if data.startswith(b"MZ"):
            return "PE executable (signature)"
        if data.startswith(b"\x7fELF"):
            return "ELF executable (signature)"
        if lower.endswith((".doc", ".docm", ".docx", ".xls", ".xlsx", ".ppt", ".pptx")):
            return "Office document (extension)"
        if lower.endswith(".pdf") or data.startswith(b"%PDF"):
            return "PDF (extension/signature)"
        if lower.endswith(".rtf") or data.startswith(b"{\\rtf"):
            return "RTF (extension/signature)"
        return "Unknown"

