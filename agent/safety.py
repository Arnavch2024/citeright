"""Treat fetched bytes as hostile data.

The model never sees raw HTML without delimiters, and common prompt-injection
phrases are flagged so the synthesizer can list them as limitations. SSRF
checks live here (not in the LLM) so they cannot be talked out of.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from pydantic import BaseModel, Field

from agent.tools.base import ToolError

UNTRUSTED_BEGIN = "<<UNTRUSTED_SOURCE_CONTENT>>"
UNTRUSTED_END = "<<END_UNTRUSTED_SOURCE_CONTENT>>"

# WHY: these are the strings models are most often jailbroken with in web pages.
_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ignore_previous", re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.I)),
    ("disregard_previous", re.compile(r"disregard\s+(the\s+)?(previous|above)", re.I)),
    ("system_role", re.compile(r"(?:^|\n)\s*system\s*:", re.I)),
    ("im_start", re.compile(r"<\|im_start\|>", re.I)),
    ("inst_tag", re.compile(r"\[INST\]", re.I)),
    ("you_are_now", re.compile(r"\byou are now\b", re.I)),
    ("tool_call_forge", re.compile(r"<tool_call>|<function_call>", re.I)),
]


class InjectionReport(BaseModel):
    flagged: bool
    patterns: list[str] = Field(default_factory=list)


def wrap_untrusted(text: str) -> str:
    """Delimiter wrapper so the model is instructed to treat this as data only."""
    return f"{UNTRUSTED_BEGIN}\n{text}\n{UNTRUSTED_END}"


def scan_injection(text: str) -> InjectionReport:
    hits = [name for name, pat in _INJECTION_PATTERNS if pat.search(text)]
    return InjectionReport(flagged=bool(hits), patterns=hits)


def strip_injection_markers(text: str) -> str:
    """Neutralize marker tokens; keep the surrounding prose for BM25."""
    cleaned = text
    cleaned = re.sub(r"<\|im_start\|>|<\|im_end\|>", " ", cleaned)
    cleaned = re.sub(r"\[INST\]|\[/INST\]", " ", cleaned)
    cleaned = re.sub(r"</?tool_call>|</?function_call>", " ", cleaned)
    return cleaned


def _ip_is_blocked(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if ip.is_private or ip.is_loopback or ip.is_link_local:
        return True
    if ip.is_multicast or ip.is_reserved or ip.is_unspecified:
        return True
    # Cloud metadata (AWS/GCP/Azure) — classic SSRF target.
    if ip.version == 4 and ip in ipaddress.ip_network("169.254.0.0/16"):
        return True
    if ip.version == 6:
        mapped = getattr(ip, "ipv4_mapped", None)
        if mapped is not None:
            return _ip_is_blocked(mapped)
    return False


def _host_ips(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    host = host.strip("[]").lower()
    if host == "localhost" or host.endswith(".localhost") or host.endswith(".local"):
        raise ToolError("localhost/mDNS hosts are blocked", tool="fetch_page", retryable=False)
    # Decimal IPv4 (http://2130706433/) used to bypass string filters.
    if host.isdigit():
        return [ipaddress.IPv4Address(int(host))]
    try:
        return [ipaddress.ip_address(host)]
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ToolError(f"DNS failed for {host}: {exc}", tool="fetch_page", retryable=True) from exc
    ips: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for info in infos:
        addr = info[4][0]
        ips.append(ipaddress.ip_address(addr))
    if not ips:
        raise ToolError(f"no addresses for {host}", tool="fetch_page", retryable=True)
    return ips


def resolve_fixture_file(url: str, fixture_root: Path) -> Path:
    """Allow file:// only if the resolved path sits under eval fixtures. Not general file SSRF."""
    parsed = urlparse(url)
    if parsed.scheme != "file":
        raise ToolError("not a file URL", tool="fetch_page", retryable=False)
    raw = url2pathname(unquote(parsed.path))
    path = Path(raw).resolve()
    root = fixture_root.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ToolError(f"file URL outside fixtures root: {path}", tool="fetch_page", retryable=False) from exc
    if not path.is_file():
        raise ToolError(f"fixture file missing: {path}", tool="fetch_page", retryable=False)
    return path


def assert_public_http_url(url: str, *, tool: str = "fetch_page") -> None:
    """Reject anything that could hit this machine or RFC1918 space."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ToolError(f"rejected scheme {parsed.scheme!r}", tool=tool, retryable=False)
    host = parsed.hostname
    if not host:
        raise ToolError("URL has no hostname", tool=tool, retryable=False)
    if parsed.username or parsed.password:
        raise ToolError("URLs with userinfo are rejected", tool=tool, retryable=False)
    for ip in _host_ips(host):
        if _ip_is_blocked(ip):
            raise ToolError(f"blocked non-public address {ip} for host {host}", tool=tool, retryable=False)
