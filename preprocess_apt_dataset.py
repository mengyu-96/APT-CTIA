"""
APT归因数据集预处理脚本
将所有PDF和TXT报告转换为PyG图神经网络所需的格式
"""

from __future__ import annotations

import argparse
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
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

# 尝试导入可选依赖
import os

# 移除强制离线模式，允许尝试下载或连接
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["HF_HUB_OFFLINE"] = "1"

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
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    import numpy as np
except ImportError:
    TfidfVectorizer = None
    TruncatedSVD = None
    np = None

# 设置标准输出编码为UTF-8，解决Windows终端中文乱码问题
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 配置日志，确保使用UTF-8编码
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
# 抑制 pdfminer 的警告日志
logging.getLogger("pdfminer").setLevel(logging.ERROR)

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
    raise ValueError(f"Unsupported file type: {suffix}")


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
            raise ImportError("PDF extraction requires 'pdfplumber' or 'pymupdf' (fitz). Please install one of them (pip install pdfplumber).")
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
    raw = path.read_bytes()
    encoding: Optional[str] = "utf-8"
    if chardet is not None:
        detected = chardet.detect(raw)
        if detected and detected.get("encoding"):
            encoding = detected["encoding"]
    return raw.decode(encoding or "utf-8", errors="ignore")


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
    
    # 改进的HASH模式：避免匹配普通十六进制字符串（要求前后有分隔符或特定上下文）
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
        # 2024-2025 New/Updated
        "dodgebox", "moonwalk", "cookieplus", "todoswift", "lightspy", "flexibleferret",
        "socgholish", "landupdate808", "clearfake", "zerolot", "sting", "acidbox", "magicscroll",
        "metador", "tajmahal", "darkuniverse", "puzzlemaker", "projectsauron", "usb thief",
        "plexingeagle", "sinsono", "tensho", "white tur",
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
        # LOLBins & Others
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
        # New APT Groups
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
    
    # 新增实体类型 - 攻击活动
    OPERATION_PATTERN = re.compile(r"\bOperation\s+[A-Z][a-zA-Z0-9\s-]+\b", re.IGNORECASE)
    CAMPAIGN_PATTERN = re.compile(r"\b(?:Campaign|行动|活动)[\s:：]?([A-Z][a-zA-Z0-9\s-]+)\b", re.IGNORECASE)
    
    # SSL/TLS证书
    SSL_CERT_PATTERN = re.compile(r"\b(?:SSL|TLS|证书|certificate)[\s:：]?(?:serial|序列号)?[\s:：]?([a-fA-F0-9:]{20,})\b", re.IGNORECASE)
    
    # CWE弱点
    CWE_PATTERN = re.compile(r"\bCWE-\d+\b", re.IGNORECASE)
    
    # 行业类型
    INDUSTRY_PATTERN = re.compile(r"\b(?:金融|能源|医疗|政府|教育|制造业|technology|financial|energy|healthcare|government|education|manufacturing|defense|telecommunications|media|aerospace|transportation|retail)\b", re.IGNORECASE)
    
    # 城市
    CITY_PATTERN = re.compile(r"\b(?:北京|上海|广州|深圳|杭州|南京|武汉|成都|西安|重庆|New York|London|Moscow|Tokyo|Berlin|Paris|Washington|Beijing|Shanghai|Bangkok|Seoul|Pyongyang|Tehran|Hanoi|Taipei|Hong Kong)\b", re.IGNORECASE)
    
    # 主机名/IP组合
    HOSTNAME_PATTERN = re.compile(r"\b(?:host|主机|server|服务器)[\s:：]?([a-zA-Z0-9][a-zA-Z0-9\-\.]+)\b", re.IGNORECASE)
    
    # 用户账户
    USER_ACCOUNT_PATTERN = re.compile(r"\b(?:user|用户|account|账户|username)[\s:：]?([a-zA-Z0-9_\.\-@]+)\b", re.IGNORECASE)
    
    # 威胁情报来源
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
        
        # 新增实体类型 - 基础设施和指标
        entities.extend(self._regex_to_entities(self.URL_PATTERN, text, "URL"))
        entities.extend(self._regex_to_entities(self.FILE_PATH_PATTERN, text, "FILE_PATH"))
        entities.extend(self._regex_to_entities(self.REGISTRY_PATTERN, text, "REGISTRY"))
        entities.extend(self._regex_to_entities(self.PORT_PATTERN, text, "PORT"))
        entities.extend(self._regex_to_entities(self.PROCESS_NAME_PATTERN, text, "PROCESS"))
        entities.extend(self._regex_to_entities(self.SERVICE_NAME_PATTERN, text, "SERVICE"))
        entities.extend(self._regex_to_entities(self.USER_AGENT_PATTERN, text, "USER_AGENT"))
        entities.extend(self._regex_to_entities(self.FILE_EXTENSION_PATTERN, text, "FILE_EXT"))
        
        # 新增实体类型 - 攻击活动、漏洞、目标等
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
            # 如果有捕获组，优先使用第一个捕获组的内容作为实体文本
            if match.lastindex and match.lastindex >= 1:
                entity_text = match.group(1)
                # 计算捕获组在原字符串中的位置
                start = match.start(1)
                end = match.end(1)
                span = (start, end)
            else:
                entity_text = match.group(0)
                span = match.span()
            
            # 过滤过长或过短的实体
            if len(entity_text) < 2 or len(entity_text) > 100:
                continue
                
            # 黑名单过滤
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


def load_terms_from_json(path: Path, key: str) -> List[str]:
    """从JSON文件加载术语列表"""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if isinstance(data, dict):
        values = data.get(key, [])
    else:
        values = data
    if not isinstance(values, list):
        return []
    return [str(item) for item in values]


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
        LOGGER.info(f"DEBUG: Found {len(extra_pdfs)} PDFs in txt_root {txt_root}")
        pdf_entries.extend(extra_pdfs)
    
    # 去重
    pdf_entries = sorted(list(set(pdf_entries)))

    txt_entries = list(_scan_directory(txt_root, {".txt"}))

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


class DatasetPreprocessor:
    """数据集预处理器"""

    def __init__(self, config: PreprocessConfig) -> None:
        self.config = config
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.output_dir / "raw_index.csv"
        self.paragraphs_path = self.output_dir / "paragraphs.jsonl"
        self.entities_path = self.output_dir / "entities.jsonl"
        self.skipped_path = self.output_dir / "skipped_reports.jsonl"
        self.extractor = self._build_extractor()

    def run(self) -> None:
        """运行预处理流程"""
        metadata = build_raw_index(self.config.pdf_root, self.config.txt_root, self.metadata_path, only_txt=self.config.only_txt)
        LOGGER.info("发现 %d 个报告文件", len(metadata))
        self._process_reports(metadata)
        LOGGER.info("预处理完成！输出目录: %s", self.output_dir)

    def _process_reports(self, metadata: List[ReportMetadata]) -> None:
        """处理所有报告"""
        processed_count = 0
        skipped_count = 0

        with self.paragraphs_path.open("w", encoding="utf-8") as para_fp, self.entities_path.open(
            "w", encoding="utf-8"
        ) as entity_fp, self.skipped_path.open("w", encoding="utf-8") as skipped_fp:
            for item in metadata:
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
                if processed_count % 10 == 0:
                    LOGGER.info("已处理 %d/%d 个报告", processed_count, len(metadata))

        LOGGER.info("处理完成: 成功 %d 个, 跳过 %d 个", processed_count, skipped_count)

    def _build_extractor(self) -> EntityExtractor:
        """构建实体提取器"""
        malware_terms: Optional[List[str]] = None
        tool_terms: Optional[List[str]] = None
        organization_terms: Optional[List[str]] = None
        country_terms: Optional[List[str]] = None
        operation_terms: Optional[List[str]] = None
        industry_terms: Optional[List[str]] = None

        if self.config.vocabulary_dir and self.config.vocabulary_dir.exists():
            malware_terms = load_terms_from_json(self.config.vocabulary_dir / "malware.json", "malware")
            tool_terms = load_terms_from_json(self.config.vocabulary_dir / "tools.json", "tools")
            organization_terms = load_terms_from_json(
                self.config.vocabulary_dir / "organizations.json", "organizations"
            )
            country_terms = load_terms_from_json(self.config.vocabulary_dir / "countries.json", "countries")
            operation_terms = load_terms_from_json(self.config.vocabulary_dir / "operations.json", "operations")
            industry_terms = load_terms_from_json(self.config.vocabulary_dir / "industries.json", "industries")

        return EntityExtractor(
            malware_terms=malware_terms,
            tool_terms=tool_terms,
            organization_terms=organization_terms,
            country_terms=country_terms,
            operation_terms=operation_terms,
            industry_terms=industry_terms,
        )


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
        # 基础设施实体
        "IP",
        "DOMAIN",
        "EMAIL",
        "URL",
        "SSL_CERT",
        "HOSTNAME",
        # 恶意软件与工具
        "HASH_MD5",
        "HASH_SHA1",
        "HASH_SHA256",
        "MALWARE",
        "TOOL",
        # 攻击活动
        "OPERATION",
        "CAMPAIGN",
        # 目标与受害者
        "ORG",
        "INDUSTRY",
        "COUNTRY",
        "CITY",
        "USER_ACCOUNT",
        # 战术、技术与程序
        "MITRE_TECH",
        # 漏洞
        "CVE",
        "CWE",
        # 指标
        "FILE_PATH",
        "REGISTRY",
        "PORT",
        "PROCESS",
        "SERVICE",
        "USER_AGENT",
        "FILE_EXT",
        # 情报来源
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
        self.graph_output_path = self.output_dir / "graphs.pt"
        self.graph_stats_path = self.output_dir / "graph_stats.json"
        self.entity_types = self.DEFAULT_ENTITY_TYPES
        self.type_vector_dim = len(self.entity_types) + 1  # 额外一维表示根节点/未知
        self.entity_type_to_idx = {label: idx for idx, label in enumerate(self.entity_types)}
        self.embedder = None
        self.embedding_dim = 0
        if self.config.use_text_embedding and SentenceTransformer is not None and self.config.embedding_model:
            model_name = self.config.embedding_model
            LOGGER.info(f"Preparing to load embedding model: {model_name}")
            
            # Define potential local paths to search
            search_paths = [
                model_name, # As is
                os.path.join(os.getcwd(), model_name), # Relative to CWD
                os.path.join(os.getcwd(), model_name.replace("/", os.sep)), # Handle slash diffs
                os.path.join(os.getcwd(), "models", model_name.split("/")[-1]), # In models folder
            ]
            
            # Attempt to load
            for path in search_paths:
                try:
                    if os.path.exists(path) or "/" in path: # Only try if path exists or it looks like a model ID
                         LOGGER.info(f"Trying to load model from: {path}")
                         self.embedder = SentenceTransformer(path)
                         self.embedding_dim = int(self.embedder.get_sentence_embedding_dimension())
                         LOGGER.info(f"Successfully loaded embedding model from: {path}")
                         break
                except Exception as e:
                    LOGGER.warning(f"Failed to load from {path}: {e}")
            
            # If still not loaded, try one last desperation load with the raw string
            if self.embedder is None:
                try:
                    LOGGER.info(f"Trying default load for: {model_name}")
                    self.embedder = SentenceTransformer(model_name)
                    self.embedding_dim = int(self.embedder.get_sentence_embedding_dimension())
                    LOGGER.info(f"Successfully loaded embedding model: {model_name}")
                except Exception as e:
                    LOGGER.error(f"FATAL: All attempts to load SentenceTransformer failed: {e}")
                    raise RuntimeError(f"Could not load embedding model {model_name}. Please ensure internet connection or download model locally.")

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
            
            # 临时存储：(report_id, text_lower, label) -> list of contexts
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
            
            # 构建语料库
            for key, ctx_list in entity_contexts.items():
                doc = " ".join(ctx_list)
                corpus.append(doc)
                keys.append(key)
            
            if corpus:
                try:
                    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
                    X = vectorizer.fit_transform(corpus)
                    
                    n_components = min(self.config.tfidf_dim, X.shape[1] - 1)
                    if n_components > 50: # 至少保留50维
                        svd = TruncatedSVD(n_components=n_components, random_state=42)
                        X_reduced = svd.fit_transform(X)
                        
                        for i, key in enumerate(keys):
                            entity_embedding_map[key] = torch.tensor(X_reduced[i], dtype=torch.float)
                        
                        LOGGER.info(f"TF-IDF特征计算完成，维度: {n_components}")
                    else:
                         LOGGER.warning(f"TF-IDF特征维度不足({X.shape[1]})，使用原始TF-IDF")
                         X_dense = X.toarray()
                         for i, key in enumerate(keys):
                            entity_embedding_map[key] = torch.tensor(X_dense[i], dtype=torch.float)

                except Exception as e:
                    LOGGER.error(f"TF-IDF计算失败: {e}")

        graphs: List[Data] = []
        stats: List[dict] = []

        for report_id, meta in metadata.items():
            report_entities = entities.get(report_id, {})
            graph = self._build_graph_for_report(report_id, meta, report_entities, label_map, entity_embedding_map)
            if graph is None:
                continue
            graphs.append(graph)
            stats.append(
                {
                    "report_id": report_id,
                    "apt_group": meta.get("apt_group", "UNKNOWN"),
                    "num_nodes": int(graph.num_nodes),
                    "num_edges": int(graph.num_edges),
                    "feature_dim": int(getattr(graph, "feature_dim", graph.x.size(1) if graph.x is not None else 0)),
                    "max_degree": float(getattr(graph, "max_degree", 0.0)),
                    "avg_degree": float(getattr(graph, "avg_degree", 0.0)),
                    "type_counts": getattr(graph, "type_counts", {}),
                    "use_text_embedding": bool(getattr(graph, "use_text_embedding", False)),
                    "embedding_dim": int(getattr(graph, "embedding_dim", 0)),
                    "context_window_size": int(getattr(graph, "context_window_size", 0)),
                    "num_paragraphs": int(getattr(graph, "num_paragraphs", 0)),
                    "num_entities": int(getattr(graph, "num_entities", max(0, int(graph.num_nodes) - 1))),
                }
            )

        if not graphs:
            LOGGER.warning("没有构建出任何图，请检查实体抽取结果。")
            return

        torch.save(graphs, self.graph_output_path)
        with self.label_map_path.open("w", encoding="utf-8") as fp:
            json.dump(label_map, fp, ensure_ascii=False, indent=2)
        with self.graph_stats_path.open("w", encoding="utf-8") as fp:
            json.dump(stats, fp, ensure_ascii=False, indent=2)
        try:
            group_stats: Dict[str, dict] = {}
            for item in stats:
                grp = item.get("apt_group", "UNKNOWN") or "UNKNOWN"
                entry = group_stats.get(grp)
                if entry is None:
                    entry = {
                        "num_graphs": 0,
                        "total_nodes": 0,
                        "total_edges": 0,
                        "total_entities": 0,
                        "total_feature_dim": 0,
                    }
                    group_stats[grp] = entry
                entry["num_graphs"] += 1
                entry["total_nodes"] += int(item.get("num_nodes", 0))
                entry["total_edges"] += int(item.get("num_edges", 0))
                entry["total_entities"] += int(item.get("num_entities", 0))
                entry["total_feature_dim"] += int(item.get("feature_dim", 0))
            summary = {
                "per_group": {},
                "global": {
                    "num_graphs": len(graphs),
                    "feature_dim": int(stats[0].get("feature_dim", 0)) if stats else 0,
                },
            }
            for grp, entry in group_stats.items():
                n = max(entry["num_graphs"], 1)
                summary["per_group"][grp] = {
                    "num_graphs": entry["num_graphs"],
                    "avg_num_nodes": entry["total_nodes"] / n,
                    "avg_num_edges": entry["total_edges"] / n,
                    "avg_num_entities": entry["total_entities"] / n,
                    "avg_feature_dim": entry["total_feature_dim"] / n,
                }
            group_stats_path = self.output_dir / "graph_group_stats.json"
            with group_stats_path.open("w", encoding="utf-8") as fp:
                json.dump(summary, fp, ensure_ascii=False, indent=2)
            LOGGER.info("图按APT组织统计已保存: %s", group_stats_path)
        except Exception as e:
            LOGGER.error("生成图统计汇总失败: %s", e)
        LOGGER.info("图构建完成，共输出 %d 个图。", len(graphs))
        LOGGER.info("graphs.pt: %s", self.graph_output_path)
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
                
                # 获取TF-IDF特征
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
        
        # 策略1: 段落内共现边（原有逻辑）
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
                # 段落内实体两两连接
                for i, j in combinations(unique_ids, 2):
                    edge_set.add((i, j))
                    if self.config.undirected:
                        edge_set.add((j, i))
            elif len(unique_ids) == 1:
                # 单实体段落：添加自环边
                idx = unique_ids[0]
                edge_set.add((idx, idx))
        
        # 策略2: 跨段落连接（如果实体在多个段落出现，连接这些段落的所有实体）
        entity_to_paragraphs: Dict[int, List[int]] = defaultdict(list)
        for para_idx, paragraph_entities in report_entities.items():
            for entry in paragraph_entities:
                text = entry.get("text", "").strip()
                label = entry.get("label", "UNKNOWN")
                key = (text.lower(), label)
                idx = entity_nodes.get(key)
                if idx is not None:
                    entity_to_paragraphs[idx].append(para_idx)
        
        # 对于出现在多个段落的实体，连接这些段落的所有实体
        for entity_idx, para_list in entity_to_paragraphs.items():
            if len(para_list) > 1:  # 实体出现在多个段落
                # 找到这些段落中的所有实体
                related_entities = set()
                for para_idx in para_list:
                    for entry in report_entities.get(para_idx, []):
                        text = entry.get("text", "").strip()
                        label = entry.get("label", "UNKNOWN")
                        key = (text.lower(), label)
                        idx = entity_nodes.get(key)
                        if idx is not None:
                            related_entities.add(idx)
                # 连接这些实体
                related_list = sorted(related_entities)
                if len(related_list) >= 2:
                    for i, j in combinations(related_list, 2):
                        edge_set.add((i, j))
                        if self.config.undirected:
                            edge_set.add((j, i))
        
        # 策略3: 报告根节点连接（如果启用）
        if self.config.add_report_node and len(node_features) > 1:
            root_idx = 0
            for idx in range(1, len(node_features)):
                edge_set.add((root_idx, idx))
                if self.config.undirected:
                    edge_set.add((idx, root_idx))
        
        # 策略4: 单实体图的处理（添加自环边和虚拟连接）
        if num_entities == 1 and not edge_set:
            # 只有一个实体时，添加自环边
            entity_idx = root_offset  # 第一个实体节点
            edge_set.add((entity_idx, entity_idx))
            if self.config.add_report_node:
                edge_set.add((0, entity_idx))
                edge_set.add((entity_idx, 0))
        
        # 策略5: 基于语义关系的边构建（新增）
        # 构建实体类型到节点索引的映射
        entity_type_to_nodes: Dict[str, List[int]] = defaultdict(list)
        for (text, label), idx in entity_nodes.items():
            if label != "REPORT":
                entity_type_to_nodes[label].append(idx)
        
        # 5.1 归属/隶属于关系
        # OPERATION/CAMPAIGN -> APT组织（通过报告根节点）
        # IP/DOMAIN -> 攻击者（通过共现）
        operation_nodes = entity_type_to_nodes.get("OPERATION", []) + entity_type_to_nodes.get("CAMPAIGN", [])
        if operation_nodes and self.config.add_report_node:
            for op_idx in operation_nodes:
                edge_set.add((0, op_idx))  # 报告 -> 攻击活动
                if self.config.undirected:
                    edge_set.add((op_idx, 0))
        
        # 5.2 使用关系
        # MALWARE/TOOL -> MITRE_TECH（恶意软件使用攻击技术）
        malware_nodes = entity_type_to_nodes.get("MALWARE", [])
        tool_nodes = entity_type_to_nodes.get("TOOL", [])
        mitre_nodes = entity_type_to_nodes.get("MITRE_TECH", [])
        for mal_idx in malware_nodes + tool_nodes:
            for mitre_idx in mitre_nodes:
                edge_set.add((mal_idx, mitre_idx))
                if self.config.undirected:
                    edge_set.add((mitre_idx, mal_idx))
        
        # OPERATION/CAMPAIGN -> CVE（攻击活动利用漏洞）
        cve_nodes = entity_type_to_nodes.get("CVE", [])
        for op_idx in operation_nodes:
            for cve_idx in cve_nodes:
                edge_set.add((op_idx, cve_idx))
                if self.config.undirected:
                    edge_set.add((cve_idx, op_idx))
        
        # 5.3 通信/连接关系
        # HASH -> IP/DOMAIN（恶意软件样本连接C2服务器）
        hash_nodes = entity_type_to_nodes.get("HASH_MD5", []) + entity_type_to_nodes.get("HASH_SHA1", []) + entity_type_to_nodes.get("HASH_SHA256", [])
        ip_nodes = entity_type_to_nodes.get("IP", [])
        domain_nodes = entity_type_to_nodes.get("DOMAIN", [])
        url_nodes = entity_type_to_nodes.get("URL", [])
        for hash_idx in hash_nodes:
            for infra_idx in ip_nodes + domain_nodes + url_nodes:
                edge_set.add((hash_idx, infra_idx))
                if self.config.undirected:
                    edge_set.add((infra_idx, hash_idx))
        
        # 5.4 投放/传播关系
        # EMAIL -> HASH（钓鱼邮件投放恶意软件）
        email_nodes = entity_type_to_nodes.get("EMAIL", [])
        for email_idx in email_nodes:
            for hash_idx in hash_nodes:
                edge_set.add((email_idx, hash_idx))
                if self.config.undirected:
                    edge_set.add((hash_idx, email_idx))
        
        # CVE -> HASH（漏洞利用传播恶意软件）
        for cve_idx in cve_nodes:
            for hash_idx in hash_nodes:
                edge_set.add((cve_idx, hash_idx))
                if self.config.undirected:
                    edge_set.add((hash_idx, cve_idx))
        
        # 5.5 关联/匹配关系
        # IP <-> DOMAIN（IP解析为域名）
        for ip_idx in ip_nodes:
            for domain_idx in domain_nodes:
                edge_set.add((ip_idx, domain_idx))
                if self.config.undirected:
                    edge_set.add((domain_idx, ip_idx))
        
        # HASH <-> HASH（相似恶意软件样本）
        if len(hash_nodes) >= 2:
            for i, j in combinations(hash_nodes, 2):
                edge_set.add((i, j))
                if self.config.undirected:
                    edge_set.add((j, i))
        
        # 5.6 持久化关系
        # HASH -> REGISTRY（恶意软件通过注册表持久化）
        registry_nodes = entity_type_to_nodes.get("REGISTRY", [])
        for hash_idx in hash_nodes:
            for reg_idx in registry_nodes:
                edge_set.add((hash_idx, reg_idx))
                if self.config.undirected:
                    edge_set.add((reg_idx, hash_idx))
        
        # HASH -> FILE_PATH（恶意软件文件路径）
        file_path_nodes = entity_type_to_nodes.get("FILE_PATH", [])
        for hash_idx in hash_nodes:
            for path_idx in file_path_nodes:
                edge_set.add((hash_idx, path_idx))
                if self.config.undirected:
                    edge_set.add((path_idx, hash_idx))
        
        # 5.7 目标关系
        # OPERATION/CAMPAIGN -> INDUSTRY/COUNTRY/CITY/ORG（攻击活动目标）
        industry_nodes = entity_type_to_nodes.get("INDUSTRY", [])
        country_nodes = entity_type_to_nodes.get("COUNTRY", [])
        city_nodes = entity_type_to_nodes.get("CITY", [])
        org_nodes = entity_type_to_nodes.get("ORG", [])
        target_nodes = industry_nodes + country_nodes + city_nodes + org_nodes
        for op_idx in operation_nodes:
            for target_idx in target_nodes:
                edge_set.add((op_idx, target_idx))
                if self.config.undirected:
                    edge_set.add((target_idx, op_idx))
        
        # 5.8 执行关系
        # TOOL -> MITRE_TECH（工具执行攻击技术）
        for tool_idx in tool_nodes:
            for mitre_idx in mitre_nodes:
                edge_set.add((tool_idx, mitre_idx))
                if self.config.undirected:
                    edge_set.add((mitre_idx, tool_idx))
        
        # 5.9 指示关系
        # FILE_PATH/REGISTRY/PROCESS -> HASH（指标指示恶意软件）
        indicator_nodes = file_path_nodes + registry_nodes + entity_type_to_nodes.get("PROCESS", [])
        for ind_idx in indicator_nodes:
            for hash_idx in hash_nodes:
                edge_set.add((ind_idx, hash_idx))
                if self.config.undirected:
                    edge_set.add((hash_idx, ind_idx))
        
        # 5.10 隶属于同一活动关系
        # 同一段落内的所有实体隶属于同一活动
        for paragraph_entities in report_entities.values():
            para_node_ids = []
            for entry in paragraph_entities:
                text = entry.get("text", "").strip()
                label = entry.get("label", "UNKNOWN")
                key = (text.lower(), label)
                idx = entity_nodes.get(key)
                if idx is not None:
                    para_node_ids.append(idx)
            # 如果段落中有OPERATION/CAMPAIGN，连接该段落所有实体
            has_operation = any(
                entry.get("label") in ("OPERATION", "CAMPAIGN")
                for entry in paragraph_entities
            )
            if has_operation and len(para_node_ids) >= 2:
                for i, j in combinations(para_node_ids, 2):
                    edge_set.add((i, j))
                    if self.config.undirected:
                        edge_set.add((j, i))
        
        # 策略6: 小图增强（节点数<5时，添加更多连接）
        if num_entities < 5 and num_entities > 1:
            # 为小图添加额外的全连接（降低权重，但增加连通性）
            entity_indices = list(range(root_offset, len(node_features)))
            if len(entity_indices) >= 2:
                # 添加所有实体之间的连接（如果还没有）
                for i, j in combinations(entity_indices, 2):
                    if (i, j) not in edge_set:
                        edge_set.add((i, j))
                        if self.config.undirected:
                            edge_set.add((j, i))

        # 如果仍然没有边，尝试最后的手段
        if not edge_set:
            # 如果至少有一个实体，添加自环边
            if num_entities >= 1:
                entity_idx = root_offset
                edge_set.add((entity_idx, entity_idx))
                if self.config.add_report_node:
                    edge_set.add((0, entity_idx))
                    edge_set.add((entity_idx, 0))
            else:
                return None

        # 可选：按每节点最多保留K条边裁剪，优先保留不同类型连接
        if self.config.max_edges_per_node and self.config.max_edges_per_node > 0:
            per_node: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
            for u, v in edge_set:
                per_node[u].append((u, v))
            pruned: Set[Tuple[int, int]] = set()
            for u, edges_u in per_node.items():
                # 排序：不同类型优先
                edges_u_sorted = sorted(
                    edges_u,
                    key=lambda e: 0 if node_labels[e[0]] != node_labels[e[1]] else 1
                )
                for e in edges_u_sorted[: self.config.max_edges_per_node]:
                    pruned.add(e)
            edge_set = pruned
        edges = sorted(edge_set)
        x = torch.stack(node_features)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        apt_group = meta.get("apt_group", "UNKNOWN") or "UNKNOWN"
        label_idx = label_map.get(apt_group, label_map["UNKNOWN"])

        # 计算文档级 Embedding (Mean Pooling of Sentences or Full Doc Encoding)
        doc_emb_tensor = None
        if self.embedder:
            try:
                # 获取该报告的所有段落文本
                paragraphs = self.paragraph_texts.get(report_id, {})
                full_text = " ".join([p for p in paragraphs.values() if isinstance(p, str)])
                
                # 如果文本太长，SentenceTransformer可能会截断，但通常会自动处理
                # 这里我们直接编码整篇文档（或者取前N个字符）
                if full_text:
                    # encode返回 numpy array
                    doc_emb = self.embedder.encode(full_text[:10000]) # 限制长度防止过慢
                    doc_emb_tensor = torch.tensor(doc_emb, dtype=torch.float).unsqueeze(0) # [1, dim]
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
        deg = torch.bincount(edge_index[0], minlength=x.size(0)).float()
        data.max_degree = float(deg.max().item()) if deg.numel() > 0 else 0.0
        data.avg_degree = float(deg.mean().item()) if deg.numel() > 0 else 0.0
        type_counts: Dict[str, int] = {}
        for lbl in node_labels:
            if lbl == "REPORT":
                continue
            type_counts[lbl] = type_counts.get(lbl, 0) + 1
        data.type_counts = type_counts
        data.feature_dim = int(x.size(1))
        data.use_text_embedding = bool(self.embedder is not None)
        data.embedding_dim = int(self.embedding_dim)
        data.context_window_size = int(self.config.context_window_size)
        data.num_paragraphs = int(len(self.paragraph_texts.get(report_id, {})))
        data.num_entities = int(len(node_features) - root_offset)
        return data

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
        
        # 即使embedder加载失败，也要保证维度对齐，填充零向量
        # 384是all-MiniLM-L6-v2的标准维度
        expected_dim = self.embedding_dim if self.embedding_dim > 0 else 384 
        
        if self.embedder is not None:
            try:
                emb = self.embedder.encode(text, normalize_embeddings=True)
                emb_t = torch.tensor(emb, dtype=torch.float)
            except Exception as e:
                # LOGGER.error(f"Text embedding failed: {e}") # 减少日志噪音
                emb_t = torch.zeros(expected_dim, dtype=torch.float)
        elif self.config.use_text_embedding:
             # 用户要求embedding但模型未加载，填充零向量
             emb_t = torch.zeros(expected_dim, dtype=torch.float)
        else:
            emb_t = torch.empty(0, dtype=torch.float)

        if self.embedder is not None and self.config.embed_context and paragraph_text:
            try:
                pemb = self.embedder.encode(paragraph_text, normalize_embeddings=True)
                pemb_t = torch.tensor(pemb, dtype=torch.float)
            except Exception as e:
                pemb_t = torch.zeros(expected_dim, dtype=torch.float)
        elif self.config.embed_context and self.config.use_text_embedding:
            pemb_t = torch.zeros(expected_dim, dtype=torch.float)
        else:
            pemb_t = torch.empty(0, dtype=torch.float)

        if self.embedder is not None and self.config.embed_context and context_text:
            try:
                cemb = self.embedder.encode(context_text, normalize_embeddings=True)
                cemb_t = torch.tensor(cemb, dtype=torch.float)
            except Exception as e:
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


# ============================================================================
# 命令行入口
# ============================================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="预处理APT归因数据集，转换为PyG图神经网络所需格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python preprocess_apt_dataset.py --pdf-root dataset_PDF --txt-root dataset_TXT --output-dir processed
  
依赖安装:
  pip install pymupdf chardet langdetect
        """,
    )
    parser.add_argument("--pdf-root", type=Path, required=False, help="PDF报告目录")
    parser.add_argument("--txt-root", type=Path, default=Path("dataset_TXT"), help="TXT报告目录 (默认: dataset_TXT)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results_archive/processed_data/latest"),
        help="输出目录 (默认: results_archive/processed_data/latest)",
    )
    parser.add_argument(
        "--vocabulary-dir",
        type=Path,
        default=None,
        help="可选：包含JSON词典的目录（malware.json, tools.json, organizations.json, countries.json）",
    )
    parser.add_argument(
        "--min-paragraph-length",
        type=int,
        default=20,
        help="段落最小字符数（默认: 20）",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="保留空文档（默认会跳过）",
    )
    parser.add_argument(
        "--skip-graphs",
        action="store_true",
        help="仅执行文本预处理，跳过图构建阶段",
    )
    parser.add_argument(
        "--min-entities",
        type=int,
        default=3,
        help="构图时每个报告至少需要的实体数量（默认: 3）",
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=64,
        help="实体哈希特征的维度（默认: 64）",
    )
    parser.add_argument(
        "--no-report-node",
        action="store_true",
        help="构图时不添加报告根节点",
    )
    parser.add_argument(
        "--only-txt",
        action="store_true",
        help="仅处理TXT数据",
    )
    parser.add_argument(
        "--embed-text",
        action="store_true",
        help="启用句向量嵌入以增强节点特征",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="句向量模型标识",
    )
    parser.add_argument(
        "--embed-context",
        action="store_true",
        help="为实体加入上下文窗口嵌入",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=64,
        help="实体上下文窗口字符数",
    )
    parser.add_argument(
        "--max-edges-per-node",
        type=int,
        default=0,
        help="每个节点保留的最多边数量（0表示不限制）",
    )
    parser.add_argument(
        "--use-tfidf",
        action="store_true",
        help="启用全局TF-IDF特征作为节点属性",
    )
    parser.add_argument(
        "--tfidf-dim",
        type=int,
        default=300,
        help="TF-IDF特征SVD降维后的维度",
    )
    return parser.parse_args()


def main() -> None:
    """主函数"""
    args = parse_args()
    
    # 验证输入目录
    if args.pdf_root and not args.pdf_root.exists():
        LOGGER.error("PDF目录不存在: %s", args.pdf_root)
        return
    if not args.txt_root.exists():
        LOGGER.error("TXT目录不存在: %s", args.txt_root)
        return

    config = PreprocessConfig(
        pdf_root=args.pdf_root,
        txt_root=args.txt_root,
        output_dir=args.output_dir,
        vocabulary_dir=args.vocabulary_dir,
        min_paragraph_length=args.min_paragraph_length,
        skip_empty=not args.keep_empty,
        only_txt=args.only_txt,
    )
    
    LOGGER.info("开始预处理...")
    LOGGER.info("PDF目录: %s", config.pdf_root)
    LOGGER.info("TXT目录: %s", config.txt_root)
    LOGGER.info("输出目录: %s", config.output_dir)
    
    preprocessor = DatasetPreprocessor(config)
    preprocessor.run()
    LOGGER.info("预处理完成！")

    if args.skip_graphs:
        LOGGER.info("已按参数要求跳过图构建阶段。")
        return

    graph_config = GraphBuilderConfig(
        min_entities_per_graph=max(1, args.min_entities),
        feature_dim=max(1, args.feature_dim),
        add_report_node=not args.no_report_node,
        use_text_embedding=bool(args.embed_text),
        embedding_model=args.embed_model if args.embed_text else None,
        max_edges_per_node=(args.max_edges_per_node if args.max_edges_per_node and args.max_edges_per_node > 0 else None),
        embed_context=bool(args.embed_context),
        context_window_size=int(args.context_window) if args.embed_context else 0,
        use_tfidf=bool(args.use_tfidf),
        tfidf_dim=int(args.tfidf_dim),
    )
    builder = GraphDatasetBuilder(
        metadata_path=preprocessor.metadata_path,
        entities_path=preprocessor.entities_path,
        output_dir=config.output_dir,
        config=graph_config,
        paragraphs_path=preprocessor.paragraphs_path,
    )
    builder.run()
    LOGGER.info("图构建流程完成。")


if __name__ == "__main__":
    main()
