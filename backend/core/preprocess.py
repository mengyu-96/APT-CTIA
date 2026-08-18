"""
APT归因数据集预处理脚本
将所有PDF和TXT报告转换为PyG图神经网络所需的格式
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union
import os
import sys
import io

# 尝试导入可选依赖
try:
    import fitz  # type: ignore
except ImportError:
    fitz = None

try:
    import pdfplumber  # type: ignore
except ImportError:
    pdfplumber = None

try:
    import chardet  # type: ignore
except ImportError:
    chardet = None

try:
    import easyocr  # type: ignore
except ImportError:
    easyocr = None

try:
    from langdetect import DetectorFactory, detect_langs  # type: ignore
except ImportError:
    DetectorFactory = None
    detect_langs = None

try:
    import torch
except ImportError:
    torch = None

try:
    from torch_geometric.data import Data
except ImportError:
    Data = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:
    SentenceTransformer = None

try:
    from backend.core.device_utils import preferred_sentence_transformer_device
except ImportError:
    try:
        from core.device_utils import preferred_sentence_transformer_device  # type: ignore
    except ImportError:
        def preferred_sentence_transformer_device() -> str:
            return "cpu"

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    import numpy as np
except ImportError:
    TfidfVectorizer = None
    TruncatedSVD = None
    np = None

# 设置标准输出编码为UTF-8
if sys.platform == 'win32':
    # Only wrap if not already utf-8
    if getattr(sys.stdout, 'encoding', '') != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURE_SCHEMA_VERSION = "apt-ctia-feature-schema-v5"
ENTITY_EXTRACTOR_VERSION = "rule-extractor-v5"


def resolve_embedding_model_path(model_spec: str) -> str:
    """Prefer a local model directory before falling back to a hub identifier."""
    if not model_spec:
        return model_spec

    candidate = Path(model_spec)
    if candidate.exists():
        return str(candidate.resolve())

    search_roots = [
        REPO_ROOT,
        Path.cwd(),
    ]
    for root in search_roots:
        local_dir = root / model_spec
        if local_dir.exists() and local_dir.is_dir():
            return str(local_dir.resolve())

    return model_spec

# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class ReportMetadata:
    """报告元数据"""
    report_id: str
    apt_group: str
    source_vendor: Optional[str]
    published_date: Optional[str]
    file_path: str
    file_type: str
    raw_name: str
    extra_tags: List[str]

    def to_row(self) -> dict:
        data = asdict(self)
        data["extra_tags"] = "|".join(self.extra_tags)
        return data


@dataclass
class Entity:
    """实体"""
    text: str
    label: str
    span: Tuple[int, int]
    model_eligible: bool = True
    source: str = "explicit"
    mapping_reason: Optional[str] = None


# ============================================================================
# 文件名解析
# ============================================================================

DATE_PATTERNS = [
    (re.compile(r"\((\d{2})-(\d{2})-(\d{4})\)"), "%m-%d-%Y"),
    (re.compile(r"\((\d{4})-(\d{2})-(\d{2})\)"), "%Y-%m-%d"),
    (re.compile(r"\((\d{2})-(\d{2})-(\d{2})\)"), "%m-%d-%y"),
    (re.compile(r"_(\d{4})(\d{2})(\d{2})_"), "%Y%m%d"),
    (re.compile(r"(\d{4})-(\d{2})-(\d{2})"), "%Y-%m-%d"),
    (re.compile(r"(\d{2})-(\d{2})-(\d{4})"), "%m-%d-%Y"),
]

VENDOR_HINTS = [
    "kaspersky", "mandiant", "paloalto", "paloaltonetworks", "symantec",
    "trendmicro", "microsoft", "proofpoint", "cisco", "cisctal",
    "sentinelone", "eset", "ahnlab", "drweb", "checkpoint", "fortinet",
    "nccgroup", "crowdstrike", "fireeye", "recordedfuture", "qianxin",
    "qax", "cert", "trellix", "group-ib", "hvconsulting", "lab52",
    "nsfocus", "telsy", "thedfirreport", "zscaler", "malwarebytes",
    "inquest", "netskope", "cyble", "avast", "alyac",
]


def parse_report_filename(name: str) -> dict:
    """从文件名提取元数据"""
    clean_name = name.strip()
    apt_group = _extract_group(clean_name)
    publication_date = _extract_date(clean_name)
    vendor = _extract_vendor(clean_name, apt_group)
    extra_tags = _extract_extra_tags(clean_name, apt_group, vendor)
    return {
        "apt_group": apt_group,
        "date": publication_date,
        "vendor": vendor,
        "extra_tags": extra_tags,
    }


def _extract_group(name: str) -> str:
    """提取APT组织名称"""
    tokens = re.split(r"[\s_\-]+", name)
    if not tokens:
        return "UNKNOWN"
    first = tokens[0]
    aliases = {
        "未知": "UNKNOWN",
        "APT": "UNKNOWN",
        "GROUP": "UNKNOWN",
    }
    group = aliases.get(first.upper(), first.upper())
    if re.fullmatch(r"(APT)?\d{1,3}", group, flags=re.IGNORECASE):
        group = group.upper().replace("APTAPT", "APT")
        if not group.startswith("APT"):
            group = f"APT{group}"
    return group or "UNKNOWN"


def _extract_date(name: str) -> Optional[str]:
    """提取发布日期"""
    for pattern, fmt in DATE_PATTERNS:
        match = pattern.search(name)
        if match:
            try:
                token = match.group(0).strip("()_")
                date_obj = datetime.strptime(token, fmt)
                return date_obj.date().isoformat()
            except ValueError:
                continue
    return None


def _extract_vendor(name: str, group: str) -> Optional[str]:
    """提取厂商名称"""
    lowered = name.lower()
    for hint in VENDOR_HINTS:
        if hint in lowered:
            return hint.upper()
    parts = re.split(r"[()\s]+", name)
    for part in parts[1:]:
        if not part:
            continue
        if part.upper().startswith(group):
            continue
        if part.isupper() and part.isalpha() and len(part) > 2:
            return part.upper()
    return None


def _extract_extra_tags(name: str, group: str, vendor: Optional[str]) -> List[str]:
    """提取额外标签"""
    parts = re.split(r"[()\s_]+", name)
    tags = []
    for part in parts:
        cleaned = part.strip()
        if not cleaned:
            continue
        upper = cleaned.upper()
        if upper in {group, vendor}:
            continue
        if re.fullmatch(r"T\d{4}(?:\.\d{3})?", upper):
            tags.append(upper)
            continue
        if upper in {"REPORT", "FINAL", "SUMMARY"}:
            continue
        if len(cleaned) > 2:
            tags.append(cleaned)
    return tags


def _build_report_id(group: str, date_str: Optional[str], vendor: Optional[str], raw_name: str) -> str:
    """构建报告ID"""
    segments = [group or "UNKNOWN"]
    if date_str:
        segments.append(date_str.replace("-", ""))
    if vendor:
        segments.append(vendor)
    slug_base = _slugify("_".join(segments))
    hash_suffix = hashlib.md5(raw_name.encode("utf-8")).hexdigest()[:8]
    if slug_base:
        return f"{slug_base}_{hash_suffix}"
    return f"report_{hash_suffix}"


def _slugify(value: str) -> str:
    """转换为URL友好的slug"""
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


# ============================================================================
# 文件加载
# ============================================================================

BLOCK_MIN_CHARS = 15
FOOTER_PATTERNS = [
    re.compile(r"^\s*page\s+\d+\s+of\s+\d+\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*/\s*\d+\s*$"),
]
HEADER_PATTERNS = [
    re.compile(r"^\s*table\s+of\s+contents\s*$", re.IGNORECASE),
]


def load_document(path: Path) -> str:
    """加载文档（PDF或TXT）"""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf_text(path)
    if suffix == ".txt":
        return load_txt_text(path)
    if suffix == ".json":
        # 兼容 JSON 格式的 Sysmon 日志，将其转为伪文本
        return load_json_text(path)
    # 尝试作为文本读取
    return load_txt_text(path)

def load_json_text(path: Path) -> str:
    """从JSON文件加载文本（简单的转储）"""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return json.dumps(data, ensure_ascii=False)
    except Exception:
        return load_txt_text(path)


# OCR fallback configuration (for scanned / image-only PDFs)
OCR_DPI = 200          # 渲染分辨率，兼顾准确率与速度
OCR_MAX_PAGES = 50     # 安全上限，避免超大扫描件无限运行
OCR_LANGUAGES = ["en", "ch_sim"]
_OCR_READER = None     # 模块级懒加载单例


def _get_ocr_reader():
    """惰性构建共享的 EasyOCR Reader（若 torch 检测到 CUDA 则使用 GPU）。"""
    global _OCR_READER
    if easyocr is None or np is None:
        return None
    if _OCR_READER is None:
        use_gpu = bool(torch is not None and torch.cuda.is_available())
        try:
            _OCR_READER = easyocr.Reader(OCR_LANGUAGES, gpu=use_gpu)
        except Exception as exc:
            LOGGER.warning("Failed to initialize EasyOCR reader: %s", exc)
            return None
    return _OCR_READER


def ocr_pdf_text(path: Path) -> str:
    """对扫描型 PDF 逐页渲染为图像并做 OCR，作为最终兜底。"""
    reader = _get_ocr_reader()
    if reader is None or fitz is None or np is None:
        return ""
    try:
        doc = fitz.open(str(path))
    except Exception as e:
        LOGGER.error("OCR: failed to open %s: %s", path, e)
        return ""

    parts: List[str] = []
    try:
        zoom = OCR_DPI / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        for page_no, page in enumerate(doc):
            if page_no >= OCR_MAX_PAGES:
                LOGGER.warning("OCR: %s exceeds %d pages; truncating.", path, OCR_MAX_PAGES)
                break
            try:
                pix = page.get_pixmap(matrix=matrix)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                if pix.n == 4:  # 去掉 alpha 通道
                    img = img[:, :, :3]
                lines = reader.readtext(img, detail=0, paragraph=True)
            except Exception as e:
                LOGGER.warning("OCR: failed on page %d of %s: %s", page_no, path, e)
                continue
            if lines:
                parts.append(_normalize_block("\n".join(lines)))
    finally:
        doc.close()

    return "\n\n".join(p for p in parts if p)


def load_pdf_text(path: Path) -> str:
    """从PDF提取文本"""
    # 优先使用 pdfplumber，因为它通常能更好地保持文本顺序
    if pdfplumber is not None:
        try:
            text_parts = []
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
            
            if text_parts:
                combined = "\n\n".join(_remove_repeated_page_margins(text_parts))
                return combined
        except Exception as e:
            LOGGER.warning(f"pdfplumber extraction failed for {path}: {e}. Falling back to fitz/PyMuPDF.")

    if fitz is None:
        if pdfplumber is None:
            # 没有任何文本层提取库，尝试 OCR 兜底
            LOGGER.warning("PDF extraction requires 'pdfplumber' or 'pymupdf' (fitz); attempting OCR.")
        else:
            # pdfplumber failed/empty and fitz is missing
            LOGGER.warning("pdfplumber produced no text and fitz is not available; attempting OCR.")
    else:
        try:
            doc = fitz.open(str(path))
            page_texts: List[str] = []
            scanned_pages = 0

            for page in doc:
                page_blocks = page.get_text("blocks")
                if not page_blocks:
                    scanned_pages += 1
                    continue
                blocks: List[str] = []
                for block in page_blocks:
                    if len(block) < 5:
                        continue
                    text = block[4].strip()
                    if len(text) < BLOCK_MIN_CHARS:
                        continue
                    if _is_header_or_footer(text):
                        continue
                    blocks.append(text)
                if blocks:
                    page_texts.append("\n".join(blocks))

            doc.close()

            if page_texts:
                combined = "\n\n".join(_remove_repeated_page_margins(page_texts))
                return combined

            LOGGER.warning("No text blocks extracted from %s using fitz; attempting OCR.", path)
        except Exception as e:
            LOGGER.error(f"Failed to extract PDF text with fitz for {path}: {e}")

    # 最终兜底：对扫描型 / 图像型 PDF 做 OCR
    ocr_text = ocr_pdf_text(path)
    if ocr_text.strip():
        LOGGER.info("OCR recovered %d chars from %s", len(ocr_text), path)
        return ocr_text

    LOGGER.warning("OCR produced no text for %s; treating as empty.", path)
    return ""


def load_txt_text(path: Path) -> str:
    """从TXT文件加载文本"""
    try:
        raw = path.read_bytes()
        encoding: Optional[str] = "utf-8"
        if chardet is not None:
            detected = chardet.detect(raw)
            if detected and detected.get("encoding"):
                encoding = detected["encoding"]
        return raw.decode(encoding or "utf-8", errors="ignore")
    except Exception as e:
        LOGGER.error(f"Failed to read text file {path}: {e}")
        return ""


def segment_paragraphs(text: str, min_length: int = 20, max_length: int = 2000) -> List[str]:
    """Split text into bounded paragraphs while preserving sentence boundaries."""
    normalized = _normalize_whitespace(text)
    paragraphs = [p.strip() for p in normalized.split("\n\n")]
    result: List[str] = []
    sentence_pattern = re.compile(r"(?<=[.!?。！？])\s+|\n+")
    for paragraph in paragraphs:
        if len(paragraph) <= max_length:
            if len(paragraph) >= min_length:
                result.append(paragraph)
            continue
        current: List[str] = []
        current_length = 0
        for sentence in (part.strip() for part in sentence_pattern.split(paragraph)):
            if not sentence:
                continue
            if current and current_length + len(sentence) + 1 > max_length:
                result.append(" ".join(current))
                current = []
                current_length = 0
            current.append(sentence)
            current_length += len(sentence) + 1
        if current:
            result.append(" ".join(current))
    return [p for p in result if len(p) >= min_length]
def detect_language(text: str, default: str = "unknown") -> str:
    """检测文本语言"""
    if detect_langs is None or DetectorFactory is None:
        return default
    DetectorFactory.seed = 0
    trimmed = text.strip()
    if not trimmed:
        return default
    try:
        langs = detect_langs(trimmed[:5000])
    except Exception:
        return default
    if not langs:
        return default
    best = max(langs, key=lambda item: item.prob)
    return best.lang


def _normalize_block(text: str) -> str:
    """规范化文本块"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(
        r"\b([A-Za-z0-9_-]+)\.\n(js|py|exe|dll|bat|cmd|ps1|vbs|jar|sh|bin)\b",
        r"\1.\2",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _remove_repeated_page_margins(page_texts: Sequence[str]) -> List[str]:
    """Remove only lines repeatedly occurring near PDF page boundaries."""
    normalized_pages = [_normalize_block(text) for text in page_texts]
    if len(normalized_pages) < 3:
        return [text for text in normalized_pages if text]

    boundary_counts: Counter[str] = Counter()
    boundary_lines: List[List[str]] = []
    for page in normalized_pages:
        lines = [line.strip() for line in page.splitlines() if line.strip()]
        boundary_lines.append(lines)
        candidates = set(lines[:4] + lines[-4:])
        for line in candidates:
            if 2 <= len(line) <= 160:
                boundary_counts[line] += 1

    threshold = max(3, math.ceil(len(normalized_pages) * 0.30))
    repeated = {line for line, count in boundary_counts.items() if count >= threshold}
    cleaned: List[str] = []
    for lines in boundary_lines:
        last_index = len(lines) - 1
        kept = [
            line for index, line in enumerate(lines)
            if not (line in repeated and (index < 4 or index > last_index - 4))
        ]
        text = _normalize_block("\n".join(kept))
        if text:
            cleaned.append(text)
    return cleaned


def _normalize_whitespace(text: str) -> str:
    """规范化空白字符"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\t", " ", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_header_or_footer(text: str) -> bool:
    """判断是否为页眉或页脚"""
    target = text.strip()
    for pattern in FOOTER_PATTERNS + HEADER_PATTERNS:
        if pattern.match(target.lower()):
            return True
    return False


# ============================================================================
# 实体提取
# ============================================================================

class EntityExtractor:
    """实体提取器（基于正则表达式和词典匹配）"""

    DOMAIN_TLDS = {
        "app", "biz", "cloud", "cn", "co", "com", "dev", "edu", "gov", "info",
        "io", "me", "mil", "net", "org", "ru", "site", "tech", "top", "uk",
        "us", "xyz",
    }
    DOMAIN_DENYLIST = {"date.now", "request.post"}
    MALWARE_TERM_DENYLIST = {
        "media", "module", "node", "obfuscated", "office", "old", "on", "one",
        "online", "only", "open", "operation", "org", "origin", "other", "over",
        "appears", "are", "been", "campaign", "contains", "including", "implant",
        "implants", "operating", "that", "the", "used", "using", "variant", "variants",
        "was", "which", "with", "and", "from", "into", "itself", "receives", "begins",
        "repo", "repository", "capabilities",
    }
    ENTITY_PRIORITY = {
        "DIRECT_ACTOR_MENTION": 110,
        "URL": 100,
        "EMAIL": 95,
        "IP": 90,
        "HASH_SHA256": 90,
        "HASH_SHA1": 90,
        "HASH_MD5": 90,
        "CVE": 85,
        "MITRE_TECH": 85,
        "FILE_PATH": 80,
        "FILE_NAME": 80,
        "REGISTRY": 80,
        "ORG": 70,
        "MALWARE": 65,
        "TOOL": 65,
    }

    # 改进的IP地址模式：排除版本号、本地地址等
    IP_PATTERN = re.compile(r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b")
    
    # 改进的域名模式：排除文件路径中的点
    DOMAIN_PATTERN = re.compile(
        r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[A-Za-z]{2,}\b"
    )
    
    EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    
    # 改进的HASH模式
    MD5_PATTERN = re.compile(r"\b(?<![\da-fA-F])[a-fA-F0-9]{32}(?![\da-fA-F])\b")
    SHA1_PATTERN = re.compile(r"\b(?<![\da-fA-F])[a-fA-F0-9]{40}(?![\da-fA-F])\b")
    SHA256_PATTERN = re.compile(r"\b(?<![\da-fA-F])[a-fA-F0-9]{64}(?![\da-fA-F])\b")
    
    MITRE_PATTERN = re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.IGNORECASE)
    CVE_PATTERN = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)
    
    # 新增实体类型
    URL_PATTERN = re.compile(
        r"\b(?:https?|hxxps?)://[^\s<>\"{}|\\^`\[\]]*[^\s<>\"{}|\\^`\[\]().,;:]",
        re.IGNORECASE,
    )
    FILE_PATH_PATTERN = re.compile(
        r'(?<!\w)(?:[A-Za-z]:[\\/]|/)(?:[^\\/\s:*?"<>|\r\n]+[\\/])+'
        r'[^\\/\s:*?"<>|\r\n]+\.(?:exe|dll|bat|cmd|ps1|vbs|js|jar|py|sh|bin|scr|com|pif|lnk)\b',
        re.IGNORECASE,
    )
    REGISTRY_PATTERN = re.compile(r"\b(?:HKEY_|HKLM|HKCU|HKCR|HKU|HKCC)[\\\w\s\-_.]+", re.IGNORECASE)
    PORT_PATTERN = re.compile(r"\b(?:port|端口)[\s:：]?(\d{1,5})\b", re.IGNORECASE)
    PROCESS_NAME_PATTERN = re.compile(r"\b(?:process|进程)[\s:：]?([A-Za-z][\w\s\-_.]+\.(?:exe|dll|bat|cmd|ps1))\b", re.IGNORECASE)
    SERVICE_NAME_PATTERN = re.compile(r"\b(?:service|服务)[\s:：]?([A-Za-z][\w\s\-_.]+)\b", re.IGNORECASE)
    USER_AGENT_PATTERN = re.compile(r"\bUser-Agent[\s:：][^\n\r]+", re.IGNORECASE)
    FILE_EXTENSION_PATTERN = re.compile(r"\.(?:exe|dll|bat|cmd|ps1|vbs|js|jar|py|sh|bin|scr|com|pif|lnk|doc|docx|xls|xlsx|pdf|zip|rar|7z)\b", re.IGNORECASE)
    FILE_NAME_PATTERN = re.compile(
        r"\b[A-Za-z0-9][A-Za-z0-9_.-]{0,80}\.(?:exe|dll|bat|cmd|ps1|vbs|js|jar|py|sh|bin|scr|lnk)\b",
        re.IGNORECASE,
    )
    
    DEFAULT_MALWARE_TERMS = [
        "plugx", "poisonivy", "gh0st", "darkcomet", "njrat", "zeus", "spyeye", "citadel",
        "dridex", "emotet", "trickbot", "ryuk", "wannacry", "notpetya", "stuxnet", "duqu",
        "flame", "gauss", "ragnarok", "sodinokibi", "revil", "conti", "maze", "egregor",
        "doppelpaymer", "hammertoss", "seaduke", "cozyduke", "onionduke", "miniduke",
        "dukes", "cozybear", "fancybear", "equationgroup", "lazarus", "turla", "oilrig",
        "apt28", "apt29", "sofacy", "sednit", "sandworm", "hades", "badrabbit", "olympic destroyer",
        "xagent", "x-agent", "chopstick", "coreshell", "oldbait", "splm", "kage", "komplex",
        "dealerschoice", "downdelph", "uroburos", "agent.btz", "comrat", "carbon", "gazer",
        "skiper", "pfinet", "mosquito", "whitebear", "kopiluwak", "epic turla", "snake",
        "ouroboros", "penquin", "helminth", "ismdoor", "bondupdater", "twoface", "quadagent",
        "oopsie", "karkoff", "murky top", "hikit", "derusbi", "winnti", "thor", "rxbot",
        "sso", "pupy", "cobalt strike", "sakula", "scanbox", "pirpi", "9002 rat", "bbk",
        "blackcoffee", "bubblewrap", "build_tracker", "casper", "china chopper", "chownet",
        "cobaltstrike", "crosswalk", "dnsmessenger", "dogcall", "dorthel", "driftnet",
        "dropshot", "elise", "emissary", "evilgrab", "fake m", "felismus", "final1stspy",
        "fin4", "firedance", "flowerpot", "formbook", "fox panda", "fysbis", "gameover zeus",
        "ghost", "gold dragon", "gongda exploit kit", "gorsh", "gref", "greyenergy",
        "grizzly steppe", "groundbait", "group 72", "h1n1", "hackingteam", "halcyon",
        "halfbaked", "hancitor", "happy", "hardrain", "havij", "hawkeye", "hdmr", "hellokitty",
        "hermes", "hi-z", "hidden cobra", "hidden lynx", "highnoise", "him", "hancitor",
        "httpbrowser", "hummingbad", "hydra", "icefog", "icedid", "invisimole", "iron tiger",
        "ismagent", "ivory", "jackpos", "jaku", "japan", "jcss", "jebena", "jhuhat",
        "jigsaw", "jripbot", "jsrbot", "kaba", "karagany", "kasidet", "kazuar", "keyboy",
        "keylogger", "khrat", "killdisk", "kingslayer", "kivars", "klez", "klosief", "knockout",
        "komplex", "konni", "kovter", "kpot", "kremlin", "kriptovor", "kryptik", "krypton",
        "ktm", "kuzzle", "l0phtcrack", "l3mon", "ladon", "ladyboy", "lambert", "lazarus group",
        "leafminer", "legion", "level4", "liberate", "lightneuron", "linton", "linux.duk",
        "lion", "lite power", "lizar", "locky", "loki", "lokibot", "longwatch", "looty",
        "lost door", "lotus blossom", "luckycat", "luminousmoth", "lux", "maccontrol",
        "macdownloader", "machbot", "macspy", "magic hound", "magniber", "mailslot",
        "malumpos", "mamid", "man1", "manitsme", "manuscrypt", "maple", "marap", "markir",
        "mars joke", "mars stealer", "master of war", "matrix", "maui", "maya", "maze",
        "mbroot", "mdmbot", "mealybug", "medusa", "megacortex", "megalodon", "megumin",
        "mekotio", "meltdown", "menupass", "merom", "metamorfo", "meteor", "methbot",
        "microcin", "microp", "middle out", "mimi", "mimikatz", "mindspiders", "minidionis",
        "miniduke", "mirage", "mirai", "mis-type", "mivast", "mobef", "mobileorder", "mobspy",
        "module", "molerats", "moneytaker", "monsoon", "moonlight", "moonwind", "more_eggs",
        "mosquito", "mothra", "mount locker", "mouse", "mouth", "mpk", "ms17-010", "msblaster",
        "msf", "msfvenom", "msil/kryptik", "msr", "mssql", "multigrain", "murky", "murkytop",
        "musca", "muscle", "music", "mustang panda", "mute", "mykings", "mysterious werewolf",
        "mysterybot", "mythic", "naikon", "nanocore", "nanohail", "navrat", "navy", "nbtscan",
        "ndisk", "necurs", "nemesis", "nemty", "nephils", "nerbian", "netbot", "netc",
        "netcat", "neteagle", "netfilter", "nethravel", "netjob", "netrepser", "netsupport",
        "nettraveler", "netwalker", "netwire", "netwiredrc", "neuron", "newscaster", "newtab",
        "next generation", "nfs", "ngrok", "niapy", "nibiru", "nic", "night dragon", "nightclub",
        "nightdoor", "nightmare", "nightrose", "nightshade", "nightsky", "nihao", "nim",
        "nimble", "nimda", "nimzoloader", "ninja", "nippon", "nitro", "nix", "njrat",
        "nk", "nls_933w.dll", "nm", "no-ip", "noah", "nobelium", "nocturnal", "node", "nofat",
        "nokki", "nomad", "non-sucking service manager", "noone", "nordic", "normandy",
        "north korea", "notpetya", "noun", "nova", "novetta", "nps", "nptel", "nrg", "nrs",
        "nsa", "nsrl", "nssm", "ntfs", "ntlm", "ntp", "nuclear", "null", "nullcon", "nuos",
        "nup", "nuts", "nyl", "o94", "oak", "oath", "obfuscated", "oblivion", "ocean",
        "oceanlotus", "octopus", "odinaff", "office", "office365", "offline", "ogre", "oilrig",
        "ok", "old", "ole", "ollydbg", "olympic destroyer", "omega", "omni", "on", "one",
        "onedrive", "onion", "onionduke", "oneline", "onemillion", "oneshark", "onions",
        "online", "only", "oopsie", "opal", "open", "openc2", "openssh", "openssl", "opera",
        "operation", "ophcrack", "optic", "oracle", "orange", "orbit", "orca", "ore",
        "org", "origin", "orion", "ork", "os", "os x", "osiris", "osox", "ospatch", "ospf",
        "ostap", "ot", "other", "outlook", "outrider", "over", "owl", "owa", "owasp",
        "owner", "ox", "oxygen", "oy", "ozone"
    ]
    
    DEFAULT_TOOL_TERMS = [
        "mshta", "regsvr32", "cmstp", "installutil", "msbuild", "cscript", "wscript",
        "at.exe", "schtasks.exe", "net.exe", "whoami.exe", "ipconfig.exe",
        "mimikatz", "pssexec", "psexec", "powershell", "cmd", "rundll32", "wmic", "net", "sc",
        "reg", "certutil", "bitsadmin", "cobalt strike", "cobaltstrike", "metasploit", "empire",
        "bloodhound", "nmap", "wireshark", "ida", "ghidra", "ollydbg", "x64dbg", "procmon",
        "procexp", "autoruns", "sysinternals", "tor", "onion", "bitcoin", "monero",
        "china chopper", "caidao", "burpsuite", "sqlmap", "fofa", "shodan", "censys",
        "winrar", "7zip", "putty", "winscp", "teamviewer", "anydesk", "beef", "webbug",
        "netsh", "schtasks", "tasklist", "taskkill", "ipconfig", "whoami", "systeminfo",
        "nbtstat", "netstat", "route", "arp", "nslookup", "ping", "tracert", "telnet", "ftp",
        "tftp", "powershell.exe", "cmd.exe", "bash", "sh", "python", "perl", "ruby", "gcc",
        "make", "cknife", "antsword", "behinder", "godzilla", "dnscat2", "frp", "gost", "ngrok",
        "chisel", "plink", "regeorg", "tunna", "lcx", "htran", "ew", "termite", "ladon",
        "crackmapexec", "impacket", "responder", "mitm6", "rubeus", "seatbelt", "sharpound",
        "sharpdump", "sharpwmi", "sharpssh", "sharpchrome", "safetykatz", "dumpert",
        "procdump", "lsassy", "nanodump", "minidump", "powerploit", "nishang", "empire",
        "poshc2", "covenant", "sliver", "mythic", "brute ratel", "havoc", "merlin", "shad0w"
    ]

    DEFAULT_COUNTRY_TERMS = [
        "China", "North Korea", "Russia", "USA", "United States", "Iran", "Vietnam", "India", 
        "Pakistan", "South Korea", "Japan", "Germany", "UK", "France", "Israel", "Thailand",
        "Philippines", "Taiwan", "Hong Kong", "Ukraine", "Belarus", "Syria", "Iraq", "Afghanistan",
        "Saudi Arabia", "Turkey", "Brazil", "Mexico", "Canada", "Australia", "Netherlands",
        "Belgium", "Switzerland", "Sweden", "Norway", "Finland", "Denmark", "Poland",
        "Czech Republic", "Romania", "Bulgaria", "Greece", "Italy", "Spain", "Portugal",
        "Egypt", "UAE", "Qatar", "Kuwait", "Jordan", "Lebanon", "Malaysia", "Indonesia", "Singapore",
        "Estonia", "Latvia", "Lithuania", "Moldova", "Georgia", "Kazakhstan", "Uzbekistan",
        "Azerbaijan", "Armenia", "Venezuela", "Colombia", "Argentina", "Chile", "Peru",
        "South Africa", "Nigeria", "Kenya", "Ethiopia"
    ]

    DEFAULT_ORG_TERMS = [
        "Flax Typhoon", "Volt Typhoon", "Salt Typhoon", "Midnight Blizzard", "Star Blizzard",
        "Google", "Microsoft", "Adobe", "Oracle", "Cisco", "FireEye", "CrowdStrike", "Mandiant",
        "Kaspersky", "Symantec", "Trend Micro", "Palo Alto Networks", "Sony", "Sony Pictures",
        "Boyusec", "Hacking Team", "Gamma Group", "NSO Group", "FBI", "NSA", "CIA", "MSS",
        "PLA", "Unit 61398", "Lazarus", "Equation Group", "Shadow Brokers", "Barium", "Lead", "Winnti",
        "Axiom", "Comment Crew", "Putter Panda", "Stone Panda", "Deep Panda", "Numbered Panda",
        "Samurai Panda", "Gothic Panda", "Anchor Panda", "Mustang Panda", "Aquatic Panda",
        "Red Delta", "Wicked Panda", "Emissary Panda", "Judgement Panda", "Karma Panda",
        "Maverick Panda", "Nomad Panda", "Sneaky Panda", "Union Panda", "Violin Panda",
        "Vixen Panda", "Wet Panda", "Energetic Bear", "Berserk Bear", "Cozy Bear", "Fancy Bear",
        "Venomous Bear", "Voodoo Bear", "Primitive Bear", "Static Kitten", "Charming Kitten",
        "Magic Kitten", "Rocket Kitten", "Refined Kitten", "Helix Kitten", "Remix Kitten",
        "Pioneer Kitten", "Vampire Kitten", "Nemesis Kitten", "Muren Shark", "Mythic Leopard",
        "OceanLotus", "DarkHotel", "Kimsuky", "Konni", "Reaper", "Scarcruft", "Group123",
        "Tonto Team", "Tick", "Rancor", "Gallmaker", "Naikon", "APT1", "APT3", "APT10",
        "APT12", "APT16", "APT17", "APT18", "APT19", "APT21", "APT27", "APT28", "APT29",
        "APT30", "APT32", "APT33", "APT34", "APT35", "APT37", "APT38", "APT39", "APT40", "APT41",
        "Lockheed Martin", "Boeing", "Raytheon", "Northrop Grumman", "General Dynamics", "BAE Systems",
        "Thales", "Airbus", "Safran", "Mitsubishi Heavy Industries", "Kawasaki Heavy Industries",
        "IHI Corporation", "Hitachi", "Toshiba", "NEC", "Fujitsu", "NTT", "KDDI", "SoftBank",
        "SK Hynix", "Samsung", "LG", "Hyundai", "Daewoo", "Posco", "Lotte", "Hanwha",
        "Petrobras", "Pemex", "Saudi Aramco", "Qatar Petroleum", "ADNOC", "Gazprom", "Rosneft",
        "Lukoil", "Sberbank", "VTB Bank", "Central Bank of Russia", "SWIFT", "IMF", "World Bank"
    ]
    
    OPERATION_PATTERN = re.compile(
        r"\bOperation[ \t]+[A-Z0-9][A-Za-z0-9-]*(?:[ \t]+[A-Z0-9][A-Za-z0-9-]*){0,5}\b"
    )
    CAMPAIGN_PATTERN = re.compile(
        r"\bCampaign[ \t:：]+([A-Z0-9][A-Za-z0-9-]*(?:[ \t]+[A-Z0-9][A-Za-z0-9-]*){0,5})\b"
    )
    
    SSL_CERT_PATTERN = re.compile(r"\b(?:SSL|TLS|证书|certificate)[\s:：]?(?:serial|序列号)?[\s:：]?([a-fA-F0-9:]{20,})\b", re.IGNORECASE)
    
    CWE_PATTERN = re.compile(r"\bCWE-\d+\b", re.IGNORECASE)
    
    INDUSTRY_PATTERN = re.compile(r"\b(?:金融|能源|医疗|政府|教育|制造业|technology|financial|energy|healthcare|government|education|manufacturing|defense|telecommunications|aerospace|transportation|retail)\b", re.IGNORECASE)
    
    CITY_PATTERN = re.compile(r"\b(?:北京|上海|广州|深圳|杭州|南京|武汉|成都|西安|重庆|New York|London|Moscow|Tokyo|Berlin|Paris|Washington|Beijing|Shanghai|Bangkok|Seoul|Pyongyang|Tehran|Hanoi|Taipei|Hong Kong)\b", re.IGNORECASE)
    
    HOSTNAME_PATTERN = re.compile(
        r"\b(?:hostname|主机名)[ \t]*[:：=][ \t]*([A-Za-z0-9][A-Za-z0-9.-]{1,252})\b",
        re.IGNORECASE,
    )
    
    USER_ACCOUNT_PATTERN = re.compile(
        r"\b(?:username|user[ \t]+account|account[ \t]+name|用户名|用户账户|账户名)"
        r"[ \t]*[:：=][ \t]*([a-zA-Z0-9_.@-]+)\b",
        re.IGNORECASE,
    )
    
    THREAT_INTEL_SOURCE_PATTERN = re.compile(r"\b(?:FireEye|CrowdStrike|Mandiant|Kaspersky|Symantec|Trend Micro|Palo Alto|奇安信|360|安天|微步在线|VirusTotal|AlienVault|OTX|MITRE|ATT&CK|Unit 42|Talos|Cylance|Proofpoint|Secureworks)\b", re.IGNORECASE)

    DIRECT_ACTOR_PATTERN = re.compile(
        r"\b(?:APT\s?\d{1,3}|Lazarus(?:\s+Group)?|Hidden\s+Cobra|Fancy\s+Bear|"
        r"Cozy\s+Bear|Sandworm|Turla|OilRig|Mustang\s+Panda|OceanLotus|Nobelium|"
        r"FIN7|Deep\s+Panda|MenuPass|Winnti|Rocket\s+Kitten|Voodoo\s+Bear|"
        r"Sofacy|Sednit|Strontium|Kimsuky|MuddyWater|Gamaredon|"
        r"North\s+Korea(?=[’']s)|North\s+Korean)\b",
        re.IGNORECASE,
    )
    DYNAMIC_MALWARE_PATTERNS = (
        re.compile(r"\b(?:code[- ]?named|dubbed|tracked as|known as|named)[ \t,:-]+[\"'“”]?([A-Za-z][A-Za-z0-9_.-]{2,63})[\"'“”]?", re.IGNORECASE),
        re.compile(r"\b(?i:implant|malware|backdoor|trojan|payload)[ \t,:-]+[\"'“”]?([A-Z][A-Za-z0-9_.-]{2,63})[\"'“”]?"),
    )
    ATTACK_BEHAVIOR_RULES = (
        ("T1027", re.compile(r"\b(?:obfuscat(?:e|ed|ion)|control[- ]flow flattening|base(?:64|85)(?:[- ]encoded| encoding)?)\b", re.IGNORECASE), "obfuscated or encoded content"),
        ("T1140", re.compile(r"\b(?:decode[sd]?|decrypt(?:s|ed|ion)?|base85|xor(?:ed|ing)?)\b", re.IGNORECASE), "decoded or decrypted content"),
        ("T1082", re.compile(r"\b(?:system (?:details|information)|hostname|operating system|platform information)\b", re.IGNORECASE), "system information discovery"),
        ("T1059.007", re.compile(r"\b(?:javascript|node\.js|js script)\b", re.IGNORECASE), "JavaScript execution"),
        ("T1195", re.compile(r"\b(?:supply[- ]chain|malicious npm package|compromised package)\b", re.IGNORECASE), "software supply-chain compromise"),
        ("T1041", re.compile(r"\b(?:exfiltrat(?:e|ed|ion)|HTTP POST request)\b", re.IGNORECASE), "data exfiltration over a command-and-control channel"),
        ("T1105", re.compile(r"\b(?:download(?:s|ed|ing)? (?:a )?(?:remote )?payload|retriev(?:e|ed|ing) (?:a )?(?:second|2nd)[- ]stage implant)\b", re.IGNORECASE), "payload transfer from command-and-control infrastructure"),
        ("T1083", re.compile(r"\b(?:list(?:ing)? files in (?:a )?directory|enumerate all files|scan(?:ning)? (?:the )?system .{0,30}(?:files|directories))\b", re.IGNORECASE), "file and directory discovery"),
        ("T1005", re.compile(r"\b(?:read(?:s|ing)? (?:the )?(?:file|contents)|gathering data from every file|collect(?:s|ed|ing)? (?:the )?file data)\b", re.IGNORECASE), "collection of data from local files"),
        ("T1622", re.compile(r"\b(?:anti[- ]debugging|anti[- ]tamp?oring|detect tampering|debugging or automated analysis)\b", re.IGNORECASE), "debugger evasion"),
    )
    PORT_LIST_PATTERN = re.compile(
        r"\bports?[ \t:]+((?:\d{1,5}[ \t]*(?:(?:,|and)[ \t]*)?){1,6})",
        re.IGNORECASE,
    )
    GITHUB_ACCOUNT_PATTERN = re.compile(
        r"\b(?:GitHub|Github)\s+(?:profile|account|repository)"
        r"\s+(?:associated with|for|belonging to)\s+([A-Za-z0-9-]{3,39})\b",
        re.IGNORECASE,
    )
    NAMELY_MALWARE_PATTERN = re.compile(
        r"\b(?:components?|implants?)[^\n.;:]{0,30}\bnamely[ \t]+([A-Za-z][A-Za-z0-9_.-]{2,63})[ \t]+and[ \t]+([A-Za-z][A-Za-z0-9_.-]{2,63})",
        re.IGNORECASE,
    )

    BLACKLIST = {
        "cnc", "is", "the", "has", "are", "and", "or", "uses", "use", "to", "for", "with", "encoded", "victim", "ip", "address", "id", "name", "data", "list", "file",
        "url", "domain", "host", "server", "user", "account", "password", "email", "hash",
        "unknown", "none", "null", "true", "false", "test", "sample", "example",
        "com", "net", "org", "cn", "gov", "edu", "www", "http", "https"
    }

    def __init__(
        self,
        malware_terms: Optional[Sequence[str]] = None,
        tool_terms: Optional[Sequence[str]] = None,
        organization_terms: Optional[Sequence[str]] = None,
        country_terms: Optional[Sequence[str]] = None,
        operation_terms: Optional[Sequence[str]] = None,
        industry_terms: Optional[Sequence[str]] = None,
    ) -> None:
        if malware_terms is None:
            malware_terms = self.DEFAULT_MALWARE_TERMS
        if tool_terms is None:
            tool_terms = self.DEFAULT_TOOL_TERMS
        tool_terms = list(tool_terms) + ["Node.js", "JavaScript"]
        if country_terms is None:
            country_terms = self.DEFAULT_COUNTRY_TERMS
        if organization_terms is None:
            organization_terms = self.DEFAULT_ORG_TERMS

        organization_keys = {term.strip().lower() for term in organization_terms if term.strip()}
        country_keys = {term.strip().lower() for term in country_terms if term.strip()}
        malware_terms = [
            term
            for term in malware_terms
            if term.strip().lower() not in self.MALWARE_TERM_DENYLIST
            and term.strip().lower() not in organization_keys
            and term.strip().lower() not in country_keys
        ]

        self.malware_terms = self._normalize_terms(malware_terms)
        self.tool_terms = self._normalize_terms(tool_terms)
        self.organization_terms = self._normalize_terms(organization_terms)
        self.country_terms = self._normalize_terms(country_terms)
        self.operation_terms = self._normalize_terms(operation_terms)
        self.industry_terms = self._normalize_terms(industry_terms)

        self.malware_pattern = self._compile_phrase_pattern(self.malware_terms)
        self.tool_pattern = self._compile_phrase_pattern(self.tool_terms)
        self.organization_pattern = self._compile_phrase_pattern(self.organization_terms)
        self.country_pattern = self._compile_phrase_pattern(self.country_terms)
        self.operation_pattern = self._compile_phrase_pattern(self.operation_terms)
        self.industry_pattern = self._compile_phrase_pattern(self.industry_terms)

    def extract(self, text: str) -> List[Entity]:
        """提取所有实体"""
        entities: List[Entity] = []
        
        # 基础IOC实体
        entities.extend(self._regex_to_entities(self.IP_PATTERN, text, "IP"))
        entities.extend(self._regex_to_entities(self.DOMAIN_PATTERN, text, "DOMAIN"))
        entities.extend(self._regex_to_entities(self.EMAIL_PATTERN, text, "EMAIL"))
        entities.extend(self._regex_to_entities(self.MD5_PATTERN, text, "HASH_MD5"))
        entities.extend(self._regex_to_entities(self.SHA1_PATTERN, text, "HASH_SHA1"))
        entities.extend(self._regex_to_entities(self.SHA256_PATTERN, text, "HASH_SHA256"))
        
        # 攻击技术实体
        entities.extend(self._regex_to_entities(self.MITRE_PATTERN, text, "MITRE_TECH"))
        entities.extend(self._regex_to_entities(self.CVE_PATTERN, text, "CVE"))
        
        # 新增实体类型
        entities.extend(self._regex_to_entities(self.URL_PATTERN, text, "URL"))
        entities.extend(self._regex_to_entities(self.FILE_PATH_PATTERN, text, "FILE_PATH"))
        entities.extend(self._regex_to_entities(self.REGISTRY_PATTERN, text, "REGISTRY"))
        entities.extend(self._regex_to_entities(self.PORT_PATTERN, text, "PORT"))
        entities.extend(self._regex_to_entities(self.PROCESS_NAME_PATTERN, text, "PROCESS"))
        entities.extend(self._regex_to_entities(self.SERVICE_NAME_PATTERN, text, "SERVICE"))
        entities.extend(self._regex_to_entities(self.USER_AGENT_PATTERN, text, "USER_AGENT"))
        entities.extend(self._regex_to_entities(self.FILE_EXTENSION_PATTERN, text, "FILE_EXT"))
        entities.extend(self._regex_to_entities(self.FILE_NAME_PATTERN, text, "FILE_NAME"))
        
        entities.extend(self._regex_to_entities(self.OPERATION_PATTERN, text, "OPERATION"))
        entities.extend(self._regex_to_entities(self.CAMPAIGN_PATTERN, text, "CAMPAIGN"))
        entities.extend(self._regex_to_entities(self.SSL_CERT_PATTERN, text, "SSL_CERT"))
        entities.extend(self._regex_to_entities(self.CWE_PATTERN, text, "CWE"))
        entities.extend(self._regex_to_entities(self.INDUSTRY_PATTERN, text, "INDUSTRY"))
        entities.extend(self._regex_to_entities(self.CITY_PATTERN, text, "CITY"))
        entities.extend(self._regex_to_entities(self.HOSTNAME_PATTERN, text, "HOSTNAME"))
        entities.extend(self._regex_to_entities(self.USER_ACCOUNT_PATTERN, text, "USER_ACCOUNT"))
        entities.extend(self._regex_to_entities(self.THREAT_INTEL_SOURCE_PATTERN, text, "THREAT_INTEL_SOURCE"))
        entities.extend(self._direct_actor_entities(text))
        entities.extend(self._dynamic_malware_entities(text))
        entities.extend(self._attack_behavior_entities(text))
        entities.extend(self._port_list_entities(text))
        entities.extend(self._regex_to_entities(self.GITHUB_ACCOUNT_PATTERN, text, "USER_ACCOUNT"))

        # 词典匹配实体
        if self.malware_pattern is not None:
            entities.extend(self._regex_to_entities(self.malware_pattern, text, "MALWARE"))
        if self.tool_pattern is not None:
            entities.extend(self._regex_to_entities(self.tool_pattern, text, "TOOL"))
        if self.organization_pattern is not None:
            entities.extend(self._regex_to_entities(self.organization_pattern, text, "ORG"))
        if self.country_pattern is not None:
            entities.extend(self._regex_to_entities(self.country_pattern, text, "COUNTRY"))
        if self.operation_pattern is not None:
            entities.extend(self._regex_to_entities(self.operation_pattern, text, "OPERATION"))
        if self.industry_pattern is not None:
            entities.extend(self._regex_to_entities(self.industry_pattern, text, "INDUSTRY"))

        return self._deduplicate(entities)

    def _direct_actor_entities(self, text: str) -> List[Entity]:
        return [
            Entity(match.group(0), "DIRECT_ACTOR_MENTION", match.span(), model_eligible=False)
            for match in self.DIRECT_ACTOR_PATTERN.finditer(text)
        ]

    def _dynamic_malware_entities(self, text: str) -> List[Entity]:
        entities: List[Entity] = []
        for pattern in self.DYNAMIC_MALWARE_PATTERNS:
            for match in pattern.finditer(text):
                name = match.group(1).strip("\"'“”.,;:")
                if name.lower() in self.BLACKLIST or name.lower() in self.MALWARE_TERM_DENYLIST:
                    continue
                entities.append(Entity(name, "MALWARE", match.span(1), source="context_rule"))
        for match in self.NAMELY_MALWARE_PATTERN.finditer(text):
            for group_index in (1, 2):
                name = match.group(group_index).strip("\"'“”.,;:")
                if name.lower() not in self.MALWARE_TERM_DENYLIST:
                    entities.append(Entity(name, "MALWARE", match.span(group_index), source="context_rule"))
        return entities

    def _port_list_entities(self, text: str) -> List[Entity]:
        entities: List[Entity] = []
        for match in self.PORT_LIST_PATTERN.finditer(text):
            offset = match.start(1)
            for number in re.finditer(r"\d{1,5}", match.group(1)):
                value = number.group(0)
                if 0 < int(value) <= 65535:
                    span = (offset + number.start(), offset + number.end())
                    entities.append(Entity(value, "PORT", span, source="port_list"))
        return entities

    def _attack_behavior_entities(self, text: str) -> List[Entity]:
        entities: List[Entity] = []
        for technique_id, pattern, reason in self.ATTACK_BEHAVIOR_RULES:
            for match in pattern.finditer(text):
                entities.append(
                    Entity(
                        technique_id,
                        "MITRE_TECH",
                        match.span(),
                        source="behavior_mapping",
                        mapping_reason=f"{reason}: {match.group(0)}",
                    )
                )
        return entities

    def _regex_to_entities(self, pattern: re.Pattern, text: str, label: str) -> List[Entity]:
        """将正则匹配转换为实体列表"""
        entities = []
        for match in pattern.finditer(text):
            if match.lastindex and match.lastindex >= 1:
                entity_text = match.group(1)
                start = match.start(1)
                end = match.end(1)
                span = (start, end)
            else:
                entity_text = match.group(0)
                span = match.span()
            
            if len(entity_text) < 2 or len(entity_text) > 100:
                continue
                
            if entity_text.lower() in self.BLACKLIST:
                continue

            if not self._is_valid_entity(entity_text, label):
                continue
                
            entities.append(Entity(entity_text, label, span))
        return entities

    def _is_valid_entity(self, entity_text: str, label: str) -> bool:
        normalized = entity_text.strip().lower().rstrip(".,;:)")
        if label == "DOMAIN":
            if normalized in self.DOMAIN_DENYLIST or "." not in normalized:
                return False
            return normalized.rsplit(".", 1)[-1] in self.DOMAIN_TLDS
        if label == "PORT":
            try:
                return 0 < int(normalized) <= 65535
            except ValueError:
                return False
        if label == "FILE_NAME":
            return normalized not in {"node.js"}
        if label in {"OPERATION", "CAMPAIGN"}:
            return "\n" not in entity_text and len(entity_text) <= 80
        return True

    @staticmethod
    def _compile_phrase_pattern(terms: Sequence[str]) -> Optional[re.Pattern]:
        """编译短语匹配模式"""
        if not terms:
            return None
        escaped = sorted({re.escape(term) for term in terms if term})
        if not escaped:
            return None
        joined = "|".join(escaped)
        return re.compile(rf"\b(?:{joined})\b", re.IGNORECASE)

    @staticmethod
    def _normalize_terms(terms: Optional[Sequence[str]]) -> List[str]:
        """规范化术语列表"""
        if not terms:
            return []
        normalized = []
        for term in terms:
            cleaned = term.strip()
            if cleaned:
                normalized.append(cleaned)
        return normalized

    @staticmethod
    def _spans_overlap(first: Tuple[int, int], second: Tuple[int, int]) -> bool:
        return first[0] < second[1] and second[0] < first[1]

    def _deduplicate(self, entities: List[Entity]) -> List[Entity]:
        """去重并优先保留更具体的重叠实体。"""
        seen = set()
        unique: List[Entity] = []
        ordered = sorted(
            entities,
            key=lambda entity: (
                -self.ENTITY_PRIORITY.get(entity.label, 50),
                entity.span[0],
                -(entity.span[1] - entity.span[0]),
            ),
        )
        for entity in ordered:
            key = (entity.span, entity.label, entity.text.lower())
            if key in seen:
                continue
            if any(
                self._spans_overlap(entity.span, kept.span)
                and entity.source != "behavior_mapping"
                and kept.source != "behavior_mapping"
                for kept in unique
            ):
                continue
            seen.add(key)
            unique.append(entity)
        return sorted(unique, key=lambda entity: (entity.span[0], entity.span[1], entity.label))


# ============================================================================
# 索引构建
# ============================================================================

def _scan_directory(root: Path, suffixes: Iterable[str]) -> Iterator[Path]:
    """扫描目录查找指定后缀的文件"""
    if not root.exists():
        return iter(())

    def generator() -> Iterator[Path]:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in suffixes:
                yield path

    return generator()


def build_raw_index(
    pdf_root: Optional[Path],
    txt_root: Path,
    output_csv: Path,
    only_txt: bool = False,
) -> List[ReportMetadata]:
    """扫描PDF/TXT目录并生成统一元数据索引"""
    pdf_entries = []
    if pdf_root:
        pdf_entries.extend(list(_scan_directory(pdf_root, {".pdf"})))
    
    # 即使没指定pdf_root，也检查txt_root下是否有PDF（如果是混合存放）
    if not only_txt:
        extra_pdfs = list(_scan_directory(txt_root, {".pdf"}))
        pdf_entries.extend(extra_pdfs)
    
    # 去重
    pdf_entries = sorted(list(set(pdf_entries)))

    # 支持 .txt 和 .json (Sysmon logs treated as text)
    txt_entries = list(_scan_directory(txt_root, {".txt", ".json"}))

    rows: List[ReportMetadata] = []
    paths = txt_entries if only_txt else (pdf_entries + txt_entries)
    for path in paths:
        meta = parse_report_filename(path.stem)
        report_id = _build_report_id(meta["apt_group"], meta["date"], meta["vendor"], path.stem)
        rows.append(
            ReportMetadata(
                report_id=report_id,
                apt_group=meta["apt_group"],
                source_vendor=meta["vendor"],
                published_date=meta["date"],
                file_path=str(path),
                file_type=path.suffix.lower(),
                raw_name=path.stem,
                extra_tags=meta["extra_tags"],
            )
        )

    if not rows:
        raise ValueError("No files discovered under the provided directories.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "report_id",
                "apt_group",
                "source_vendor",
                "published_date",
                "file_path",
                "file_type",
                "raw_name",
                "extra_tags",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_row())
    return rows


# ============================================================================
# 主预处理流程
# ============================================================================

def _normalize_report_text_for_dedup(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "").lower()).strip()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff.:-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _report_shingles(text: str, width: int = 8) -> Set[str]:
    tokens = [token for token in re.split(r"\s+", text) if token]
    if len(tokens) < width:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[idx:idx + width]) for idx in range(0, len(tokens) - width + 1)}


def _jaccard(left: Set[str], right: Set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


@dataclass
class PreprocessConfig:
    """预处理配置"""
    pdf_root: Optional[Path]
    txt_root: Path
    output_dir: Path
    vocabulary_dir: Optional[Path] = None
    min_paragraph_length: int = 20
    max_paragraph_length: int = 2000
    skip_empty: bool = True
    only_txt: bool = False
    deduplicate_reports: bool = False
    near_duplicate_jaccard: float = 0.92


class DatasetPreprocessor:
    """数据集预处理器"""

    def __init__(self, config: PreprocessConfig, progress_callback=None) -> None:
        self.config = config
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.output_dir / "raw_index.csv"
        self.paragraphs_path = self.output_dir / "paragraphs.jsonl"
        self.entities_path = self.output_dir / "entities.jsonl"
        self.skipped_path = self.output_dir / "skipped_reports.jsonl"
        self.quality_path = self.output_dir / "preprocess_quality.json"
        self.extractor = self._build_extractor()
        self.progress_callback = progress_callback

    def run(self) -> None:
        """运行预处理流程"""
        if self.progress_callback:
            self.progress_callback(0, 100, "Scanning files...")
            
        metadata = build_raw_index(self.config.pdf_root, self.config.txt_root, self.metadata_path, only_txt=self.config.only_txt)
        LOGGER.info("发现 %d 个报告文件", len(metadata))
        self._process_reports(metadata)
        LOGGER.info("预处理完成！输出目录: %s", self.output_dir)

    def _process_reports(self, metadata: List[ReportMetadata]) -> None:
        """处理所有报告"""
        processed_count = 0
        skipped_count = 0
        total_files = len(metadata)
        seen_hashes: Dict[str, str] = {}
        seen_shingles: List[Tuple[str, Set[str]]] = []
        report_quality: List[dict] = []

        with self.paragraphs_path.open("w", encoding="utf-8") as para_fp, self.entities_path.open(
            "w", encoding="utf-8"
        ) as entity_fp, self.skipped_path.open("w", encoding="utf-8") as skipped_fp:
            for idx, item in enumerate(metadata):
                if self.progress_callback:
                    # Update progress every file (or every N files if too many)
                    self.progress_callback(idx, total_files, f"Processing {item.raw_name}")
                    
                path = Path(item.file_path)
                try:
                    text = load_document(path)
                except Exception as exc:
                    LOGGER.exception("加载文件失败 %s: %s", path, exc)
                    skipped_fp.write(
                        json.dumps(
                            {
                                "report_id": item.report_id,
                                "file_path": item.file_path,
                                "reason": str(exc),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    skipped_count += 1
                    continue

                if not text.strip():
                    LOGGER.warning("文档为空: %s", path)
                    if self.config.skip_empty:
                        skipped_fp.write(
                            json.dumps(
                                {
                                    "report_id": item.report_id,
                                    "file_path": item.file_path,
                                    "reason": "empty_text",
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        skipped_count += 1
                        continue

                if self.config.deduplicate_reports:
                    normalized_text = _normalize_report_text_for_dedup(text)
                    content_hash = hashlib.sha256(normalized_text.encode("utf-8", errors="ignore")).hexdigest()
                    duplicate_of = seen_hashes.get(content_hash)
                    if duplicate_of is not None:
                        skipped_fp.write(
                            json.dumps(
                                {
                                    "report_id": item.report_id,
                                    "file_path": item.file_path,
                                    "reason": "exact_duplicate_report",
                                    "duplicate_of": duplicate_of,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        skipped_count += 1
                        continue

                    shingles = _report_shingles(normalized_text)
                    near_duplicate_of = None
                    near_duplicate_score = 0.0
                    for previous_report_id, previous_shingles in seen_shingles:
                        score = _jaccard(shingles, previous_shingles)
                        if score >= self.config.near_duplicate_jaccard:
                            near_duplicate_of = previous_report_id
                            near_duplicate_score = score
                            break
                    if near_duplicate_of is not None:
                        skipped_fp.write(
                            json.dumps(
                                {
                                    "report_id": item.report_id,
                                    "file_path": item.file_path,
                                    "reason": "near_duplicate_report",
                                    "duplicate_of": near_duplicate_of,
                                    "jaccard": round(float(near_duplicate_score), 6),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        skipped_count += 1
                        continue
                    seen_hashes[content_hash] = item.report_id
                    seen_shingles.append((item.report_id, shingles))

                language = detect_language(text)
                paragraphs = segment_paragraphs(
                    text,
                    min_length=self.config.min_paragraph_length,
                    max_length=self.config.max_paragraph_length,
                )
                entity_counts: Counter[str] = Counter()
                source_counts: Counter[str] = Counter()
                excluded_from_model = 0
                
                for idx, paragraph in enumerate(paragraphs):
                    paragraph_record = {
                        "report_id": item.report_id,
                        "paragraph_index": idx,
                        "language": language,
                        "text": paragraph,
                        "apt_group": item.apt_group,
                        "source_vendor": item.source_vendor,
                        "published_date": item.published_date,
                        "file_path": item.file_path,
                    }
                    para_fp.write(json.dumps(paragraph_record, ensure_ascii=False) + "\n")

                    entities = self.extractor.extract(paragraph)
                    for entity in entities:
                        entity_counts[entity.label] += 1
                        source_counts[entity.source] += 1
                        if not entity.model_eligible:
                            excluded_from_model += 1
                        entity_record = {
                            "report_id": item.report_id,
                            "paragraph_index": idx,
                            "label": entity.label,
                            "text": entity.text,
                            "start": entity.span[0],
                            "end": entity.span[1],
                            "model_eligible": entity.model_eligible,
                            "source": entity.source,
                            "mapping_reason": entity.mapping_reason,
                        }
                        entity_fp.write(json.dumps(entity_record, ensure_ascii=False) + "\n")

                paragraph_lengths = [len(paragraph) for paragraph in paragraphs]
                report_quality.append(
                    {
                        "report_id": item.report_id,
                        "file_path": item.file_path,
                        "language": language,
                        "extracted_characters": len(text),
                        "paragraph_count": len(paragraphs),
                        "paragraph_length_min": min(paragraph_lengths, default=0),
                        "paragraph_length_max": max(paragraph_lengths, default=0),
                        "entity_count": sum(entity_counts.values()),
                        "entity_counts": dict(sorted(entity_counts.items())),
                        "entity_source_counts": dict(sorted(source_counts.items())),
                        "excluded_from_model": excluded_from_model,
                        "quality_flags": [
                            flag
                            for flag, triggered in (
                                ("no_paragraphs", not paragraphs),
                                ("no_entities", not entity_counts),
                                ("paragraph_over_limit", any(length > self.config.max_paragraph_length for length in paragraph_lengths)),
                            )
                            if triggered
                        ],
                    }
                )

                processed_count += 1

        quality_summary = {
            "total_files": total_files,
            "processed_files": processed_count,
            "skipped_files": skipped_count,
            "min_paragraph_length": self.config.min_paragraph_length,
            "max_paragraph_length": self.config.max_paragraph_length,
            "reports": report_quality,
        }
        self.quality_path.write_text(
            json.dumps(quality_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        LOGGER.info("处理完成: 成功 %d 个, 跳过 %d 个", processed_count, skipped_count)

    def _build_extractor(self) -> EntityExtractor:
        """构建实体提取器"""
        return EntityExtractor()


# ============================================================================
# 图构建
# ============================================================================

@dataclass
class GraphBuilderConfig:
    """图构建配置"""

    min_entities_per_graph: int = 3
    feature_dim: int = 64
    add_report_node: bool = True
    undirected: bool = True
    use_text_embedding: bool = False
    embedding_model: Optional[str] = None
    max_edges_per_node: Optional[int] = None
    cooccurrence_window: int = 4
    root_edges_per_paragraph: int = 1
    embed_context: bool = False
    context_window_size: int = 0
    embed_paragraph: bool = False
    embed_document: bool = False
    include_paragraph_hash: bool = False
    include_context_hash: bool = True
    use_tfidf: bool = False
    tfidf_dim: int = 300
    min_paragraph_length: int = 20
    max_paragraph_length: int = 2000


class GraphDatasetBuilder:
    """将实体JSONL转换为PyG图列表"""

    DEFAULT_ENTITY_TYPES = [
        "IP", "DOMAIN", "EMAIL", "URL", "SSL_CERT", "HOSTNAME",
        "HASH_MD5", "HASH_SHA1", "HASH_SHA256", "MALWARE", "TOOL",
        "OPERATION", "CAMPAIGN",
        "ORG", "INDUSTRY", "COUNTRY", "CITY", "USER_ACCOUNT",
        "MITRE_TECH",
        "CVE", "CWE",
        "FILE_PATH", "FILE_NAME", "REGISTRY", "PORT", "PROCESS", "SERVICE", "USER_AGENT", "FILE_EXT",
        "THREAT_INTEL_SOURCE",
    ]

    def __init__(
        self,
        metadata_path: Path,
        entities_path: Path,
        output_dir: Path,
        config: GraphBuilderConfig,
        paragraphs_path: Optional[Path] = None,
    ) -> None:
        self.metadata_path = metadata_path
        self.entities_path = entities_path
        self.output_dir = output_dir
        self.config = config
        self.label_map_path = self.output_dir / "label_mapping.json"
        # Changed: We will store graphs in a 'graphs' subdirectory
        self.graphs_dir = self.output_dir / "graphs"
        self.graph_stats_path = self.output_dir / "graph_stats.json"
        self.feature_manifest_path = self.output_dir / "feature_manifest.json"
        self.entity_types = self.DEFAULT_ENTITY_TYPES
        self.type_vector_dim = len(self.entity_types) + 1  # 额外一维表示根节点/未知
        self.entity_type_to_idx = {label: idx for idx, label in enumerate(self.entity_types)}
        self.embedder = None
        self.embedding_dim = 0
        self._embedding_cache: Dict[str, torch.Tensor] = {}
        if self.config.use_text_embedding:
            if SentenceTransformer is None or not self.config.embedding_model:
                raise RuntimeError(
                    "Text embeddings are enabled but sentence-transformers or embedding_model is unavailable."
                )
            try:
                st_device = preferred_sentence_transformer_device()
                resolved_model = resolve_embedding_model_path(self.config.embedding_model)
                LOGGER.info(f"Loading embedding model: {resolved_model} on {st_device}")
                self.embedder = SentenceTransformer(resolved_model, device=st_device)
                self.embedding_dim = int(self.embedder.get_sentence_embedding_dimension())
            except Exception as e:
                raise RuntimeError(f"Failed to load required embedding model: {e}") from e

        self.paragraphs_path = paragraphs_path
        self.paragraph_texts: Dict[str, Dict[int, str]] = {}
        if self.paragraphs_path and self.paragraphs_path.exists():
            with self.paragraphs_path.open("r", encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    rid = rec.get("report_id")
                    pidx = int(rec.get("paragraph_index", 0))
                    txt = rec.get("text", "")
                    self.paragraph_texts.setdefault(rid, {})[pidx] = txt

    def run(self) -> None:
        if torch is None or Data is None:
            raise ImportError(
                "运行图构建需要安装 torch 以及 torch-geometric。请执行 pip install torch torch-geometric"
            )
        LOGGER.info("开始构建图数据集...")
        metadata = self._load_metadata()
        entities = self._load_entities()
        label_map = self._build_label_map(metadata)
        self._prepare_embedding_cache(entities)

        # 计算TF-IDF特征
        entity_embedding_map = {}
        if self.config.use_tfidf and TfidfVectorizer is not None:
            LOGGER.info("正在计算全局TF-IDF特征...")
            corpus = []
            keys = []
            entity_contexts = defaultdict(list)

            for report_id, report_ents in entities.items():
                for para_idx, para_ents in report_ents.items():
                    para_txt = self.paragraph_texts.get(report_id, {}).get(para_idx, "")
                    for entry in para_ents:
                        text = entry.get("text", "").strip()
                        label = entry.get("label", "UNKNOWN")
                        if not text:
                            continue
                        
                        start = entry.get("start")
                        end = entry.get("end")
                        ctx = ""
                        if isinstance(start, int) and isinstance(end, int) and para_txt:
                            ctx = self._extract_context_window(para_txt, start, end, self.config.context_window_size)
                        
                        key = (report_id, text.lower(), label)
                        entity_contexts[key].append(f"{text} {ctx}")
            
            for key, ctx_list in entity_contexts.items():
                doc = " ".join(ctx_list)
                corpus.append(doc)
                keys.append(key)
            
            if corpus:
                try:
                    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
                    X = vectorizer.fit_transform(corpus)
                    
                    n_components = min(self.config.tfidf_dim, X.shape[1] - 1)
                    if n_components > 50:
                        svd = TruncatedSVD(n_components=n_components, random_state=42)
                        X_reduced = svd.fit_transform(X)
                        for i, key in enumerate(keys):
                            entity_embedding_map[key] = self._coerce_tfidf_vector(X_reduced[i])
                    else:
                         X_dense = X.toarray()
                         for i, key in enumerate(keys):
                            entity_embedding_map[key] = self._coerce_tfidf_vector(X_dense[i])

                except Exception as e:
                    LOGGER.error(f"TF-IDF计算失败: {e}")

        # Ensure graphs directory exists
        self.graphs_dir.mkdir(parents=True, exist_ok=True)
        
        saved_count = 0
        actual_feature_dim: Optional[int] = None
        stats: List[dict] = []

        for report_id, meta in metadata.items():
            report_entities = entities.get(report_id, {})
            
            # 统计实体类型
            entity_type_counts = defaultdict(int)
            for para_ents in report_entities.values():
                for ent in para_ents:
                    label = ent.get("label", "UNKNOWN")
                    entity_type_counts[label] += 1
            
            graph = self._build_graph_for_report(report_id, meta, report_entities, label_map, entity_embedding_map)
            if graph is None:
                continue
            graph_feature_dim = int(graph.x.size(1))
            if actual_feature_dim is None:
                actual_feature_dim = graph_feature_dim
            elif graph_feature_dim != actual_feature_dim:
                raise RuntimeError(
                    f"Feature dimension drift detected: expected {actual_feature_dim}, got {graph_feature_dim} "
                    f"for {report_id}."
                )
            if not torch.isfinite(graph.x).all():
                raise RuntimeError(f"Non-finite feature values detected for {report_id}.")
            if graph.edge_index.numel() and (
                int(graph.edge_index.min()) < 0 or int(graph.edge_index.max()) >= int(graph.num_nodes)
            ):
                raise RuntimeError(f"Out-of-range edge index detected for {report_id}.")
            degrees = torch.bincount(graph.edge_index[0], minlength=graph.num_nodes)
            
            # Save individual graph file
            try:
                graph_path = self.graphs_dir / f"{report_id}.pt"
                torch.save(graph, graph_path)
                saved_count += 1
            except Exception as e:
                LOGGER.error(f"Failed to save graph for {report_id}: {e}")
                continue

            stats.append(
                {
                    "report_id": report_id,
                    "apt_group": meta.get("apt_group", "UNKNOWN"),
                    "source_vendor": meta.get("source_vendor", "UNKNOWN"),
                    "published_date": meta.get("published_date", "UNKNOWN"),
                    "num_nodes": int(graph.num_nodes),
                    "num_edges": int(graph.num_edges),
                    "feature_dim": graph_feature_dim,
                    "features_finite": True,
                    "max_out_degree": int(degrees.max()) if degrees.numel() else 0,
                    "entity_counts": dict(entity_type_counts),
                    "edge_relation_counts": {
                        self._edge_relation_name(int(rel)): int((graph.edge_type == rel).sum())
                        for rel in torch.unique(graph.edge_type).tolist()
                    },
                }
            )

        if saved_count == 0:
            LOGGER.warning("没有构建出任何图，请检查实体抽取结果。")
            return

        with self.label_map_path.open("w", encoding="utf-8") as fp:
            json.dump(label_map, fp, ensure_ascii=False, indent=2)
        with self.graph_stats_path.open("w", encoding="utf-8") as fp:
            json.dump(stats, fp, ensure_ascii=False, indent=2)
        self._write_feature_manifest(actual_feature_dim or 0)
        
        LOGGER.info("图构建完成，共输出 %d 个图。", saved_count)
        LOGGER.info("Graphs directory: %s", self.graphs_dir)
        LOGGER.info("label_mapping.json: %s", self.label_map_path)

    def _write_feature_manifest(self, actual_feature_dim: int) -> None:
        contract = {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "entity_extractor_version": ENTITY_EXTRACTOR_VERSION,
            "entity_types": self.entity_types,
            "root_feature": "constant_REPORT",
            "graph": {
                "undirected": self.config.undirected,
                "cooccurrence_window": self.config.cooccurrence_window,
                "root_edges_per_paragraph": self.config.root_edges_per_paragraph,
                "max_edges_per_node": self.config.max_edges_per_node,
                "relation_schema": {
                    "0": "local_cooccurrence",
                    "1": "report_anchor",
                    "2": "self_evidence",
                    "3": "behavior_evidence",
                    "4": "malware_artifact",
                    "5": "operation_campaign",
                    "6": "entity_context",
                },
            },
            "features": {
                "actual_dim": actual_feature_dim,
                "hash_dim": self.config.feature_dim,
                "use_text_embedding": self.config.use_text_embedding,
                "embedding_model": self.config.embedding_model,
                "embedding_dim": self.embedding_dim,
                "embed_context": self.config.embed_context,
                "context_window_size": self.config.context_window_size,
                "embed_paragraph": self.config.embed_paragraph,
                "embed_document": self.config.embed_document,
                "include_paragraph_hash": self.config.include_paragraph_hash,
                "include_context_hash": self.config.include_context_hash,
                "use_tfidf": self.config.use_tfidf,
                "tfidf_dim": self.config.tfidf_dim if self.config.use_tfidf else 0,
            },
            "preprocessing": {
                "min_paragraph_length": self.config.min_paragraph_length,
                "max_paragraph_length": self.config.max_paragraph_length,
            },
        }
        serialized = json.dumps(contract, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        manifest = {
            **contract,
            "fingerprint": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        }
        self.feature_manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_metadata(self) -> Dict[str, dict]:
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"metadata 文件不存在: {self.metadata_path}")
        metadata: Dict[str, dict] = {}
        with self.metadata_path.open("r", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                metadata[row["report_id"]] = row
        return metadata

    def _load_entities(self) -> Dict[str, Dict[int, List[dict]]]:
        if not self.entities_path.exists():
            raise FileNotFoundError(f"entities 文件不存在: {self.entities_path}")
        grouped: Dict[str, Dict[int, List[dict]]] = defaultdict(lambda: defaultdict(list))
        with self.entities_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("model_eligible", True) is False:
                    continue
                report_id = record.get("report_id")
                paragraph_index = int(record.get("paragraph_index", 0))
                grouped[report_id][paragraph_index].append(record)
        return grouped

    def _build_label_map(self, metadata: Dict[str, dict]) -> Dict[str, int]:
        labels = {row.get("apt_group", "UNKNOWN") or "UNKNOWN" for row in metadata.values()}
        labels.add("UNKNOWN")
        ordered = sorted(labels)
        return {label: idx for idx, label in enumerate(ordered)}

    def _build_graph_for_report(
        self,
        report_id: str,
        meta: dict,
        report_entities: Dict[int, List[dict]],
        label_map: Dict[str, int],
        entity_embedding_map: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None,
    ) -> Optional[Data]:
        entity_nodes: Dict[Tuple[str, str], int] = {}
        node_features: List[torch.Tensor] = []
        node_labels: List[str] = []

        root_offset = 0
        if self.config.add_report_node:
            # Keep identifiers and filename-derived labels out of model features.
            node_features.append(self._build_feature_vector(text="REPORT", label=None))
            entity_nodes[("__root__", "REPORT")] = 0
            root_offset = 1
            node_labels.append("REPORT")

        # 收集所有实体
        for para_idx, paragraph_entities in report_entities.items():
            for entry in paragraph_entities:
                if entry.get("model_eligible", True) is False:
                    continue
                text = entry.get("text", "").strip()
                label = entry.get("label", "UNKNOWN")
                if not text:
                    continue
                key = (text.lower(), label)
                if key in entity_nodes:
                    continue
                para_txt = self.paragraph_texts.get(report_id, {}).get(para_idx, "")
                start = entry.get("start")
                end = entry.get("end")
                ctx = ""
                if self.config.embed_context and isinstance(start, int) and isinstance(end, int) and para_txt:
                    ctx = self._extract_context_window(para_txt, start, end, self.config.context_window_size)
                
                tfidf_vec = None
                if entity_embedding_map:
                    key_for_map = (report_id, text.lower(), label)
                    tfidf_vec = entity_embedding_map.get(key_for_map)

                node_features.append(self._build_feature_vector(text=text, label=label, paragraph_text=para_txt, context_text=ctx, tfidf_vec=tfidf_vec))
                entity_nodes[key] = len(node_features) - 1
                node_labels.append(label)

        num_entities = len(node_features) - root_offset
        if num_entities < self.config.min_entities_per_graph:
            return None

        edge_set: Set[Tuple[int, int]] = set()
        paragraph_node_ids: Dict[int, List[int]] = {}

        # Nearby co-occurrence preserves local structure without making cliques.
        for para_idx, paragraph_entities in report_entities.items():
            node_ids: List[int] = []
            for entry in paragraph_entities:
                if entry.get("model_eligible", True) is False:
                    continue
                text = entry.get("text", "").strip()
                label = entry.get("label", "UNKNOWN")
                idx = entity_nodes.get((text.lower(), label))
                if idx is not None and idx not in node_ids:
                    node_ids.append(idx)
            paragraph_node_ids[para_idx] = node_ids
            window = max(1, int(self.config.cooccurrence_window))
            for position, source in enumerate(node_ids):
                for target in node_ids[position + 1 : position + 1 + window]:
                    edge_set.add((source, target))
                    if self.config.undirected:
                        edge_set.add((target, source))
            if len(node_ids) == 1:
                edge_set.add((node_ids[0], node_ids[0]))

        # A shared entity already bridges paragraphs. Connect only paragraph anchors
        # to the report root instead of connecting the root to every entity.
        if self.config.add_report_node and len(node_features) > 1:
            anchor_count = max(1, int(self.config.root_edges_per_paragraph))
            for node_ids in paragraph_node_ids.values():
                for idx in node_ids[:anchor_count]:
                    edge_set.add((0, idx))
                    if self.config.undirected:
                        edge_set.add((idx, 0))

        edge_set = self._limit_edge_degree(edge_set)
        
        if not edge_set:
            if num_entities >= 1:
                entity_idx = root_offset
                edge_set.add((entity_idx, entity_idx))
                if self.config.add_report_node:
                    edge_set.add((0, entity_idx))
                    edge_set.add((entity_idx, 0))
            else:
                return None

        edges = sorted(edge_set)
        x = torch.stack(node_features)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        # Preserve the semantic reason for each connection.  The relation ids
        # are deliberately small and stable so they can be consumed by RGCN
        # while remaining compatible with the existing type-inference fallback.
        edge_type = torch.tensor(
            [self._infer_edge_relation(node_labels, src, dst) for src, dst in edges],
            dtype=torch.long,
        )
        apt_group = meta.get("apt_group", "UNKNOWN") or "UNKNOWN"
        label_idx = label_map.get(apt_group, label_map["UNKNOWN"])

        doc_emb_tensor = None
        if self.embedder and self.config.embed_document:
            try:
                paragraphs = self.paragraph_texts.get(report_id, {})
                full_text = " ".join([p for p in paragraphs.values() if isinstance(p, str)])
                if full_text:
                    doc_emb = self.embedder.encode(full_text[:10000])
                    doc_emb_tensor = torch.tensor(doc_emb, dtype=torch.float).unsqueeze(0)
            except Exception as exc:
                raise RuntimeError(f"Failed to generate document embedding for {report_id}: {exc}") from exc

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            y=torch.tensor([label_idx], dtype=torch.long),
        )
        if doc_emb_tensor is not None:
            data.doc_emb = doc_emb_tensor
            
        data.report_id = report_id
        data.apt_group = apt_group
        data.node_texts = self._build_node_texts(entity_nodes, self.config.add_report_node)
        data.node_labels = node_labels
        data.node_paragraph_indices = self._build_node_paragraph_indices(entity_nodes, report_entities, self.config.add_report_node)
        data.edge_relation_names = [self._edge_relation_name(int(v)) for v in edge_type.tolist()]
        return data

    @staticmethod
    def _edge_relation_name(relation_id: int) -> str:
        return {
            0: "local_cooccurrence",
            1: "report_anchor",
            2: "self_evidence",
            3: "behavior_evidence",
            4: "malware_artifact",
            5: "operation_campaign",
            6: "entity_context",
        }.get(int(relation_id), "entity_context")

    @classmethod
    def _infer_edge_relation(cls, node_labels: List[str], source: int, target: int) -> int:
        src = node_labels[source] if 0 <= source < len(node_labels) else "UNKNOWN"
        dst = node_labels[target] if 0 <= target < len(node_labels) else "UNKNOWN"
        if src == "REPORT" or dst == "REPORT":
            return 1
        if source == target:
            return 2
        if "MITRE_TECH" in (src, dst):
            return 3
        artifact_labels = {"MALWARE", "FILE_NAME", "HASH", "URL", "DOMAIN", "IP", "PORT", "EMAIL"}
        if src in artifact_labels or dst in artifact_labels:
            return 4
        if src in {"OPERATION", "CAMPAIGN"} or dst in {"OPERATION", "CAMPAIGN"}:
            return 5
        return 6

    def _limit_edge_degree(self, edges: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        limit = self.config.max_edges_per_node
        if limit is None or int(limit) <= 0:
            return edges

        pairs = {tuple(sorted((src, dst))) for src, dst in edges}
        ordered = sorted(
            pairs,
            key=lambda pair: (0 if 0 in pair else 1, abs(pair[0] - pair[1]), pair),
        )
        degree: Dict[int, int] = defaultdict(int)
        kept: Set[Tuple[int, int]] = set()
        for src, dst in ordered:
            if src == dst:
                kept.add((src, dst))
                continue
            if degree[src] >= int(limit) or degree[dst] >= int(limit):
                continue
            kept.add((src, dst))
            if self.config.undirected:
                kept.add((dst, src))
            degree[src] += 1
            degree[dst] += 1
        return kept

    @staticmethod
    def _build_node_texts(entity_nodes: Dict[Tuple[str, str], int], add_report_node: bool) -> List[str]:
        ordered = sorted(entity_nodes.items(), key=lambda item: item[1])
        texts: List[str] = []
        for (text, _label), idx in ordered:
            if add_report_node and idx == 0 and text == "__root__":
                texts.append("REPORT")
            else:
                texts.append(text)
        return texts

    @staticmethod
    def _build_node_paragraph_indices(
        entity_nodes: Dict[Tuple[str, str], int],
        report_entities: Dict[int, List[dict]],
        add_report_node: bool,
    ) -> List[int]:
        para_map: Dict[int, int] = {}
        if add_report_node:
            para_map[0] = -1
        for para_idx, paragraph_entities in report_entities.items():
            for entry in paragraph_entities:
                text = entry.get("text", "").strip().lower()
                label = entry.get("label", "UNKNOWN")
                idx = entity_nodes.get((text, label))
                if idx is not None and idx not in para_map:
                    para_map[idx] = int(para_idx)
        return [para_map.get(idx, -1) for idx in range(len(entity_nodes))]

    def _build_feature_vector(self, text: str, label: Optional[str], paragraph_text: str = "", context_text: str = "", tfidf_vec: Optional[torch.Tensor] = None) -> torch.Tensor:
        type_vector = torch.zeros(self.type_vector_dim, dtype=torch.float)
        if label and label in self.entity_type_to_idx:
            type_vector[self.entity_type_to_idx[label]] = 1.0
        else:
            type_vector[-1] = 1.0
        hashed = self._hash_text(text)
        p_hash = (
            self._hash_text(paragraph_text)
            if self.config.include_paragraph_hash and paragraph_text
            else torch.empty(0, dtype=torch.float)
        )
        c_hash = (
            self._hash_text(context_text)
            if self.config.include_context_hash and context_text
            else (torch.zeros_like(hashed) if self.config.include_context_hash else torch.empty(0, dtype=torch.float))
        )
        stats = self._text_stats(text)
        
        expected_dim = self.embedding_dim if self.embedding_dim > 0 else 384
        
        if self.embedder is not None:
            try:
                emb_t = self._cached_embedding(text)
            except Exception as exc:
                raise RuntimeError(f"Failed to embed entity text {text!r}: {exc}") from exc
        elif self.config.use_text_embedding:
             raise RuntimeError("Text embeddings are configured but the embedding model is unavailable.")
        else:
            emb_t = torch.empty(0, dtype=torch.float)

        if self.embedder is not None and self.config.embed_paragraph and paragraph_text:
            try:
                pemb_t = self._cached_embedding(paragraph_text)
            except Exception as exc:
                raise RuntimeError(f"Failed to embed paragraph context: {exc}") from exc
        elif self.config.embed_paragraph and self.config.use_text_embedding:
            pemb_t = torch.zeros(expected_dim, dtype=torch.float)
        else:
            pemb_t = torch.empty(0, dtype=torch.float)

        if self.embedder is not None and self.config.embed_context and context_text:
            try:
                cemb_t = self._cached_embedding(context_text)
            except Exception as exc:
                raise RuntimeError(f"Failed to embed local context for {text!r}: {exc}") from exc
        elif self.config.embed_context and self.config.use_text_embedding:
            cemb_t = torch.zeros(expected_dim, dtype=torch.float)
        else:
            cemb_t = torch.empty(0, dtype=torch.float)
        
        if tfidf_vec is not None:
            tfidf_t = self._coerce_tfidf_vector(tfidf_vec)
        elif self.config.use_tfidf:
            tfidf_t = torch.zeros(self.config.tfidf_dim, dtype=torch.float)
        else:
            tfidf_t = torch.empty(0, dtype=torch.float)

        return torch.cat([type_vector, hashed, p_hash, c_hash, stats, emb_t, pemb_t, cemb_t, tfidf_t])

    def _prepare_embedding_cache(self, entities: Dict[str, Dict[int, List[dict]]]) -> None:
        if self.embedder is None:
            return
        texts: Set[str] = set()
        for report_id, report_entities in entities.items():
            for para_idx, paragraph_entities in report_entities.items():
                paragraph_text = self.paragraph_texts.get(report_id, {}).get(para_idx, "")
                if self.config.embed_paragraph and paragraph_text:
                    texts.add(paragraph_text)
                for entry in paragraph_entities:
                    if entry.get("model_eligible", True) is False:
                        continue
                    text = entry.get("text", "").strip()
                    if text:
                        texts.add(text)
                    if self.config.embed_context and paragraph_text:
                        start, end = entry.get("start"), entry.get("end")
                        if isinstance(start, int) and isinstance(end, int):
                            context = self._extract_context_window(
                                paragraph_text, start, end, self.config.context_window_size
                            )
                            if context:
                                texts.add(context)
        if not texts:
            return
        ordered = sorted(texts)
        LOGGER.info("Batch-encoding %d unique graph feature texts...", len(ordered))
        vectors = self.embedder.encode(
            ordered,
            batch_size=128,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        self._embedding_cache = {
            text: torch.as_tensor(vector, dtype=torch.float).clone()
            for text, vector in zip(ordered, vectors)
        }

    def _cached_embedding(self, text: str) -> torch.Tensor:
        cached = self._embedding_cache.get(text)
        if cached is not None:
            return cached
        # This path is expected only for dynamically generated inputs not seen
        # during cache preparation (for example a report root at inference).
        vector = self.embedder.encode(text, normalize_embeddings=True)
        tensor = torch.as_tensor(vector, dtype=torch.float).clone()
        self._embedding_cache[text] = tensor
        return tensor

    def _coerce_tfidf_vector(self, vector: Union[torch.Tensor, Sequence[float]]) -> torch.Tensor:
        """Keep TF-IDF features at the fixed dimension expected by the model."""
        target_dim = max(0, int(self.config.tfidf_dim))
        result = torch.zeros(target_dim, dtype=torch.float)
        if target_dim == 0:
            return result
        source = torch.as_tensor(vector, dtype=torch.float).flatten()
        result[: min(target_dim, source.numel())] = source[:target_dim]
        return result

    @staticmethod
    def _extract_context_window(paragraph_text: str, start: int, end: int, window_chars: int) -> str:
        if not paragraph_text or window_chars is None or window_chars <= 0:
            return ""
        s = max(0, start - window_chars)
        e = min(len(paragraph_text), end + window_chars)
        return paragraph_text[s:e]

    def _hash_text(self, text: str) -> torch.Tensor:
        dim = max(1, self.config.feature_dim)
        vector = torch.zeros(dim, dtype=torch.float)
        if not text:
            return vector
        tokens = re.findall(r"[a-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff]", text.lower())
        for token in tokens:
            if not token:
                continue
            digest = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = digest % dim
            vector[idx] += 1.0
        norm = torch.norm(vector, p=2)
        if norm > 0:
            vector = vector / norm
        return vector

    @staticmethod
    def _text_stats(text: str) -> torch.Tensor:
        length = float(len(text))
        if length == 0:
            return torch.zeros(3, dtype=torch.float)
        digits = sum(1 for ch in text if ch.isdigit())
        alphabetic = sum(1 for ch in text if ch.isalpha())
        stats = torch.tensor(
            [
                min(length / 100.0, 1.0),
                digits / length,
                alphabetic / length,
            ],
            dtype=torch.float,
        )
        return stats


def run_preprocessing_pipeline(
    raw_data_dir: Union[str, Path],
    base_output_dir: Union[str, Path],
    skip_graphs: bool = False,
    progress_callback=None
) -> Path:
    """
    Executes the full preprocessing pipeline.
    
    Args:
        raw_data_dir (str or Path): Directory containing raw files (PDF/TXT/JSON).
        base_output_dir (str or Path): Directory to save processed outputs.
        skip_graphs (bool): If True, skips graph construction.
        progress_callback (callable): Optional callback(current, total, message).
    
    Returns:
        Path: The path to the final processed graphs.pt.
    """
    raw_data_dir = Path(raw_data_dir)
    base_output_dir = Path(base_output_dir)
    
    config = PreprocessConfig(
        pdf_root=None, # Treat raw_data_dir as unified root
        txt_root=raw_data_dir,
        output_dir=base_output_dir,
        only_txt=False, # Allow scanning for PDFs in txt_root
        min_paragraph_length=20,
        skip_empty=True
    )
    
    LOGGER.info("Starting preprocessing...")
    if progress_callback:
        progress_callback(0, 100, "Initializing...")
        
    preprocessor = DatasetPreprocessor(config, progress_callback=progress_callback)
    preprocessor.run()
    
    if skip_graphs:
        if progress_callback:
            progress_callback(100, 100, "Done (Graphs skipped)")
        return base_output_dir
        
    if progress_callback:
        progress_callback(90, 100, "Building Graphs...")
        
    graph_config = GraphBuilderConfig(
        min_entities_per_graph=3,
        feature_dim=64,
        add_report_node=True,
        use_text_embedding=True, # Enable Semantic Enhancement (Innovation Point)
        embedding_model="all-MiniLM-L6-v2", # Use a lightweight, high-performance model
        max_edges_per_node=12,
        cooccurrence_window=4,
        root_edges_per_paragraph=1,
        embed_context=True,
        context_window_size=240,
        embed_paragraph=False,
        embed_document=False,
        include_paragraph_hash=False,
        include_context_hash=True,
        min_paragraph_length=config.min_paragraph_length,
        max_paragraph_length=config.max_paragraph_length,
        # Dataset-local TF-IDF axes are not reusable at inference time.
        use_tfidf=False,
    )
    
    builder = GraphDatasetBuilder(
        metadata_path=preprocessor.metadata_path,
        entities_path=preprocessor.entities_path,
        output_dir=config.output_dir,
        config=graph_config,
        paragraphs_path=preprocessor.paragraphs_path,
    )
    builder.run()
    
    if progress_callback:
        progress_callback(100, 100, "Completed")
        
    return builder.graphs_dir
