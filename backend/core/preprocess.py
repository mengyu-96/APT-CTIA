"""
APT归因数据集预处理脚本
将所有PDF和TXT报告转换为PyG图神经网络所需的格式
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from itertools import combinations
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
                combined = "\n\n".join(_normalize_block(block) for block in text_parts)
                return combined
        except Exception as e:
            LOGGER.warning(f"pdfplumber extraction failed for {path}: {e}. Falling back to fitz/PyMuPDF.")

    if fitz is None:
        if pdfplumber is None:
            # 如果没有PDF库，返回空或报错，这里选择返回空字符串避免崩溃
            LOGGER.warning("PDF extraction requires 'pdfplumber' or 'pymupdf' (fitz). Please install one of them.")
            return ""
        else:
             # pdfplumber failed and fitz is missing
             LOGGER.warning("pdfplumber failed and fitz is not available.")
             return ""

    try:
        doc = fitz.open(str(path))
        blocks: List[str] = []
        scanned_pages = 0

        for page in doc:
            page_blocks = page.get_text("blocks")
            if not page_blocks:
                scanned_pages += 1
                continue
            for block in page_blocks:
                if len(block) < 5:
                    continue
                text = block[4].strip()
                if len(text) < BLOCK_MIN_CHARS:
                    continue
                if _is_header_or_footer(text):
                    continue
                blocks.append(text)

        doc.close()

        if blocks:
            combined = "\n\n".join(_normalize_block(block) for block in blocks)
            return combined

        LOGGER.warning("No text blocks extracted from %s using fitz; likely scanned.", path)
        return ""
    except Exception as e:
        LOGGER.error(f"Failed to extract PDF text with fitz for {path}: {e}")
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


def segment_paragraphs(text: str, min_length: int = 20) -> List[str]:
    """将文本分割为段落"""
    normalized = _normalize_whitespace(text)
    paragraphs = [p.strip() for p in normalized.split("\n\n")]
    return [p for p in paragraphs if len(p) >= min_length]


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
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


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
    URL_PATTERN = re.compile(r"\bhttps?://[^\s<>\"{}|\\^`\[\]]+", re.IGNORECASE)
    FILE_PATH_PATTERN = re.compile(r"\b(?:[A-Za-z]:)?(?:[/\\][\w\s\-_.]+)+\.(?:exe|dll|bat|cmd|ps1|vbs|js|jar|py|sh|bin|scr|com|pif|lnk)\b", re.IGNORECASE)
    REGISTRY_PATTERN = re.compile(r"\b(?:HKEY_|HKLM|HKCU|HKCR|HKU|HKCC)[\\\w\s\-_.]+", re.IGNORECASE)
    PORT_PATTERN = re.compile(r"\b(?:port|端口)[\s:：]?(\d{1,5})\b", re.IGNORECASE)
    PROCESS_NAME_PATTERN = re.compile(r"\b(?:process|进程)[\s:：]?([A-Za-z][\w\s\-_.]+\.(?:exe|dll|bat|cmd|ps1))\b", re.IGNORECASE)
    SERVICE_NAME_PATTERN = re.compile(r"\b(?:service|服务)[\s:：]?([A-Za-z][\w\s\-_.]+)\b", re.IGNORECASE)
    USER_AGENT_PATTERN = re.compile(r"\bUser-Agent[\s:：][^\n\r]+", re.IGNORECASE)
    FILE_EXTENSION_PATTERN = re.compile(r"\.(?:exe|dll|bat|cmd|ps1|vbs|js|jar|py|sh|bin|scr|com|pif|lnk|doc|docx|xls|xlsx|pdf|zip|rar|7z)\b", re.IGNORECASE)
    
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
    
    OPERATION_PATTERN = re.compile(r"\bOperation\s+[A-Z][a-zA-Z0-9\s-]+\b", re.IGNORECASE)
    CAMPAIGN_PATTERN = re.compile(r"\b(?:Campaign|行动|活动)[\s:：]?([A-Z][a-zA-Z0-9\s-]+)\b", re.IGNORECASE)
    
    SSL_CERT_PATTERN = re.compile(r"\b(?:SSL|TLS|证书|certificate)[\s:：]?(?:serial|序列号)?[\s:：]?([a-fA-F0-9:]{20,})\b", re.IGNORECASE)
    
    CWE_PATTERN = re.compile(r"\bCWE-\d+\b", re.IGNORECASE)
    
    INDUSTRY_PATTERN = re.compile(r"\b(?:金融|能源|医疗|政府|教育|制造业|technology|financial|energy|healthcare|government|education|manufacturing|defense|telecommunications|media|aerospace|transportation|retail)\b", re.IGNORECASE)
    
    CITY_PATTERN = re.compile(r"\b(?:北京|上海|广州|深圳|杭州|南京|武汉|成都|西安|重庆|New York|London|Moscow|Tokyo|Berlin|Paris|Washington|Beijing|Shanghai|Bangkok|Seoul|Pyongyang|Tehran|Hanoi|Taipei|Hong Kong)\b", re.IGNORECASE)
    
    HOSTNAME_PATTERN = re.compile(r"\b(?:host|主机|server|服务器)[\s:：]?([a-zA-Z0-9][a-zA-Z0-9\-\.]+)\b", re.IGNORECASE)
    
    USER_ACCOUNT_PATTERN = re.compile(r"\b(?:user|用户|account|账户|username)[\s:：]?([a-zA-Z0-9_\.\-@]+)\b", re.IGNORECASE)
    
    THREAT_INTEL_SOURCE_PATTERN = re.compile(r"\b(?:FireEye|CrowdStrike|Mandiant|Kaspersky|Symantec|Trend Micro|Palo Alto|奇安信|360|安天|微步在线|VirusTotal|AlienVault|OTX|MITRE|ATT&CK|Unit 42|Talos|Cylance|Proofpoint|Secureworks)\b", re.IGNORECASE)

    BLACKLIST = {
        "cnc", "is", "encoded", "victim", "ip", "address", "id", "name", "data", "list", "file",
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
        if country_terms is None:
            country_terms = self.DEFAULT_COUNTRY_TERMS
        if organization_terms is None:
            organization_terms = self.DEFAULT_ORG_TERMS

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
        
        entities.extend(self._regex_to_entities(self.OPERATION_PATTERN, text, "OPERATION"))
        entities.extend(self._regex_to_entities(self.CAMPAIGN_PATTERN, text, "CAMPAIGN"))
        entities.extend(self._regex_to_entities(self.SSL_CERT_PATTERN, text, "SSL_CERT"))
        entities.extend(self._regex_to_entities(self.CWE_PATTERN, text, "CWE"))
        entities.extend(self._regex_to_entities(self.INDUSTRY_PATTERN, text, "INDUSTRY"))
        entities.extend(self._regex_to_entities(self.CITY_PATTERN, text, "CITY"))
        entities.extend(self._regex_to_entities(self.HOSTNAME_PATTERN, text, "HOSTNAME"))
        entities.extend(self._regex_to_entities(self.USER_ACCOUNT_PATTERN, text, "USER_ACCOUNT"))
        entities.extend(self._regex_to_entities(self.THREAT_INTEL_SOURCE_PATTERN, text, "THREAT_INTEL_SOURCE"))

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
                
            entities.append(Entity(entity_text, label, span))
        return entities

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
    def _deduplicate(entities: List[Entity]) -> List[Entity]:
        """去重实体"""
        seen = set()
        unique: List[Entity] = []
        for entity in entities:
            key = (entity.span, entity.label, entity.text.lower())
            if key in seen:
                continue
            seen.add(key)
            unique.append(entity)
        return unique


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
    skip_empty: bool = True
    only_txt: bool = False
    deduplicate_reports: bool = True
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
                paragraphs = segment_paragraphs(text, min_length=self.config.min_paragraph_length)
                
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
                        entity_record = {
                            "report_id": item.report_id,
                            "paragraph_index": idx,
                            "label": entity.label,
                            "text": entity.text,
                            "start": entity.span[0],
                            "end": entity.span[1],
                        }
                        entity_fp.write(json.dumps(entity_record, ensure_ascii=False) + "\n")

                processed_count += 1

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
    embed_context: bool = False
    context_window_size: int = 0
    use_tfidf: bool = False
    tfidf_dim: int = 300


class GraphDatasetBuilder:
    """将实体JSONL转换为PyG图列表"""

    DEFAULT_ENTITY_TYPES = [
        "IP", "DOMAIN", "EMAIL", "URL", "SSL_CERT", "HOSTNAME",
        "HASH_MD5", "HASH_SHA1", "HASH_SHA256", "MALWARE", "TOOL",
        "OPERATION", "CAMPAIGN",
        "ORG", "INDUSTRY", "COUNTRY", "CITY", "USER_ACCOUNT",
        "MITRE_TECH",
        "CVE", "CWE",
        "FILE_PATH", "REGISTRY", "PORT", "PROCESS", "SERVICE", "USER_AGENT", "FILE_EXT",
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
        self.entity_types = self.DEFAULT_ENTITY_TYPES
        self.type_vector_dim = len(self.entity_types) + 1  # 额外一维表示根节点/未知
        self.entity_type_to_idx = {label: idx for idx, label in enumerate(self.entity_types)}
        self.embedder = None
        self.embedding_dim = 0
        if self.config.use_text_embedding and SentenceTransformer is not None and self.config.embedding_model:
            try:
                st_device = preferred_sentence_transformer_device()
                LOGGER.info(f"Loading embedding model: {self.config.embedding_model} on {st_device}")
                self.embedder = SentenceTransformer(self.config.embedding_model, device=st_device)
                self.embedding_dim = int(self.embedder.get_sentence_embedding_dimension())
            except Exception as e:
                LOGGER.warning(f"Failed to load embedding model: {e}")

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
                            entity_embedding_map[key] = torch.tensor(X_reduced[i], dtype=torch.float)
                    else:
                         X_dense = X.toarray()
                         for i, key in enumerate(keys):
                            entity_embedding_map[key] = torch.tensor(X_dense[i], dtype=torch.float)

                except Exception as e:
                    LOGGER.error(f"TF-IDF计算失败: {e}")

        # Ensure graphs directory exists
        self.graphs_dir.mkdir(parents=True, exist_ok=True)
        
        saved_count = 0
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
                    "entity_counts": dict(entity_type_counts)
                }
            )

        if saved_count == 0:
            LOGGER.warning("没有构建出任何图，请检查实体抽取结果。")
            return

        with self.label_map_path.open("w", encoding="utf-8") as fp:
            json.dump(label_map, fp, ensure_ascii=False, indent=2)
        with self.graph_stats_path.open("w", encoding="utf-8") as fp:
            json.dump(stats, fp, ensure_ascii=False, indent=2)
        
        LOGGER.info("图构建完成，共输出 %d 个图。", saved_count)
        LOGGER.info("Graphs directory: %s", self.graphs_dir)
        LOGGER.info("label_mapping.json: %s", self.label_map_path)

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
            node_features.append(self._build_feature_vector(text=report_id, label=None))
            entity_nodes[("__root__", "REPORT")] = 0
            root_offset = 1
            node_labels.append("REPORT")

        # 收集所有实体
        for para_idx, paragraph_entities in report_entities.items():
            for entry in paragraph_entities:
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
        
        # 策略1: 段落内共现边
        for paragraph_entities in report_entities.values():
            node_ids = []
            for entry in paragraph_entities:
                text = entry.get("text", "").strip()
                label = entry.get("label", "UNKNOWN")
                key = (text.lower(), label)
                idx = entity_nodes.get(key)
                if idx is not None:
                    node_ids.append(idx)
            unique_ids = sorted(set(node_ids))
            if len(unique_ids) >= 2:
                for i, j in combinations(unique_ids, 2):
                    edge_set.add((i, j))
                    if self.config.undirected:
                        edge_set.add((j, i))
            elif len(unique_ids) == 1:
                idx = unique_ids[0]
                edge_set.add((idx, idx))
        
        # 策略2: 跨段落连接
        entity_to_paragraphs: Dict[int, List[int]] = defaultdict(list)
        for para_idx, paragraph_entities in report_entities.items():
            for entry in paragraph_entities:
                text = entry.get("text", "").strip()
                label = entry.get("label", "UNKNOWN")
                key = (text.lower(), label)
                idx = entity_nodes.get(key)
                if idx is not None:
                    entity_to_paragraphs[idx].append(para_idx)
        
        for entity_idx, para_list in entity_to_paragraphs.items():
            if len(para_list) > 1:
                related_entities = set()
                for para_idx in para_list:
                    for entry in report_entities.get(para_idx, []):
                        text = entry.get("text", "").strip()
                        label = entry.get("label", "UNKNOWN")
                        key = (text.lower(), label)
                        idx = entity_nodes.get(key)
                        if idx is not None:
                            related_entities.add(idx)
                related_list = sorted(related_entities)
                if len(related_list) >= 2:
                    for i, j in combinations(related_list, 2):
                        edge_set.add((i, j))
                        if self.config.undirected:
                            edge_set.add((j, i))
        
        # 策略3: 报告根节点连接
        if self.config.add_report_node and len(node_features) > 1:
            root_idx = 0
            for idx in range(1, len(node_features)):
                edge_set.add((root_idx, idx))
                if self.config.undirected:
                    edge_set.add((idx, root_idx))
        
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
        apt_group = meta.get("apt_group", "UNKNOWN") or "UNKNOWN"
        label_idx = label_map.get(apt_group, label_map["UNKNOWN"])

        doc_emb_tensor = None
        if self.embedder:
            try:
                paragraphs = self.paragraph_texts.get(report_id, {})
                full_text = " ".join([p for p in paragraphs.values() if isinstance(p, str)])
                if full_text:
                    doc_emb = self.embedder.encode(full_text[:10000])
                    doc_emb_tensor = torch.tensor(doc_emb, dtype=torch.float).unsqueeze(0)
            except Exception as e:
                LOGGER.warning(f"Failed to generate doc embedding for {report_id}: {e}")

        data = Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor([label_idx], dtype=torch.long),
        )
        if doc_emb_tensor is not None:
            data.doc_emb = doc_emb_tensor
            
        data.report_id = report_id
        data.apt_group = apt_group
        data.node_texts = self._build_node_texts(entity_nodes, self.config.add_report_node)
        data.node_labels = node_labels
        data.node_paragraph_indices = self._build_node_paragraph_indices(entity_nodes, report_entities, self.config.add_report_node)
        return data

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
        p_hash = self._hash_text(paragraph_text) if paragraph_text else torch.zeros_like(hashed)
        c_hash = self._hash_text(context_text) if context_text else torch.zeros_like(hashed)
        stats = self._text_stats(text)
        
        expected_dim = self.embedding_dim if self.embedding_dim > 0 else 384
        
        if self.embedder is not None:
            try:
                emb = self.embedder.encode(text, normalize_embeddings=True)
                emb_t = torch.tensor(emb, dtype=torch.float)
            except Exception:
                emb_t = torch.zeros(expected_dim, dtype=torch.float)
        elif self.config.use_text_embedding:
             emb_t = torch.zeros(expected_dim, dtype=torch.float)
        else:
            emb_t = torch.empty(0, dtype=torch.float)

        if self.embedder is not None and self.config.embed_context and paragraph_text:
            try:
                pemb = self.embedder.encode(paragraph_text, normalize_embeddings=True)
                pemb_t = torch.tensor(pemb, dtype=torch.float)
            except Exception:
                pemb_t = torch.zeros(expected_dim, dtype=torch.float)
        elif self.config.embed_context and self.config.use_text_embedding:
            pemb_t = torch.zeros(expected_dim, dtype=torch.float)
        else:
            pemb_t = torch.empty(0, dtype=torch.float)

        if self.embedder is not None and self.config.embed_context and context_text:
            try:
                cemb = self.embedder.encode(context_text, normalize_embeddings=True)
                cemb_t = torch.tensor(cemb, dtype=torch.float)
            except Exception:
                cemb_t = torch.zeros(expected_dim, dtype=torch.float)
        elif self.config.embed_context and self.config.use_text_embedding:
            cemb_t = torch.zeros(expected_dim, dtype=torch.float)
        else:
            cemb_t = torch.empty(0, dtype=torch.float)
        
        if tfidf_vec is not None:
            tfidf_t = tfidf_vec
        elif self.config.use_tfidf:
            tfidf_t = torch.zeros(self.config.tfidf_dim, dtype=torch.float)
        else:
            tfidf_t = torch.empty(0, dtype=torch.float)

        return torch.cat([type_vector, hashed, p_hash, c_hash, stats, emb_t, pemb_t, cemb_t, tfidf_t])

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
        tokens = re.split(r"[^a-zA-Z0-9]+", text.lower())
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
        embed_context=False,
        use_tfidf=True
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
