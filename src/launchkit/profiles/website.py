"""Fetch a public website and reduce its HTML to readable text for AI extraction."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import httpx

from launchkit.core.exceptions import DomainError

logger = logging.getLogger(__name__)

WEBSITE_FETCH_TIMEOUT_SECONDS = 20.0
WEBSITE_MAX_BYTES = 2 * 1024 * 1024
WEBSITE_MAX_REDIRECTS = 3
_GENERIC_FETCH_ERROR = "The website could not be processed."
_PRIVATE_NETWORK_ERROR = "This address points to a private network and cannot be scanned."

_BLOCKED_HOSTNAMES = {"localhost", "localhost.localdomain"}
# Defense-in-depth: known DNS-rebinding / wildcard-to-arbitrary-IP services.
_REBINDING_HOST_SUFFIXES = (
    ".nip.io",
    ".sslip.io",
    ".xip.io",
    ".traefik.me",
    ".localtest.me",
    ".lacolhost.com",
    ".lvh.me",
    ".vcap.me",
    ".localdomain",
)
_SKIPPED_ELEMENTS = {"script", "style", "noscript", "svg", "template", "iframe", "head"}
# Block ends force line breaks so headings and paragraphs stay separated.
_BLOCK_ELEMENTS = {
    "p", "div", "section", "article", "header", "footer", "main", "aside", "nav",
    "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "br", "blockquote", "figcaption",
}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0
        self._in_title = False
        self.title = ""
        self.meta_description = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIPPED_ELEMENTS:
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
        if tag == "meta":
            attributes = dict(attrs)
            name = (attributes.get("name") or attributes.get("property") or "").lower()
            if name in {"description", "og:description"} and not self.meta_description:
                self.meta_description = (attributes.get("content") or "").strip()
        if tag in _BLOCK_ELEMENTS:
            self._chunks.append("\n")
        # Alt text often carries the only description of hero imagery.
        if tag == "img":
            alt = (dict(attrs).get("alt") or "").strip()
            if alt:
                self._chunks.append(f" {alt} ")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIPPED_ELEMENTS and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if tag == "title":
            self._in_title = False
        if tag in _BLOCK_ELEMENTS:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        # The title lives inside <head>, which is otherwise skipped wholesale.
        if self._in_title:
            self.title += data
            return
        if self._skip_depth > 0:
            return
        if data.strip():
            self._chunks.append(data)

    def text(self) -> str:
        lines = "".join(self._chunks).splitlines()
        cleaned = [" ".join(line.split()) for line in lines]
        return "\n".join(line for line in cleaned if line)


def website_page_text(html: str) -> str:
    """Readable text for one HTML page: title, meta description, then body copy."""

    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    parts: list[str] = []
    title = " ".join(parser.title.split())
    if title:
        parts.append(f"Page title: {title}")
    if parser.meta_description:
        parts.append(f"Meta description: {parser.meta_description}")
    body = parser.text()
    if body:
        parts.append(body)
    return "\n\n".join(parts)


def validate_website_url(url: str) -> str:
    """Normalize a user-supplied site URL and reject anything that is not a public web page."""

    candidate, hostname = _parse_public_web_url(url)
    _reject_literal_or_rebinding_host(hostname)
    return candidate


async def fetch_website_html(client: httpx.AsyncClient, url: str) -> str:
    """Download one page after resolve-then-validate, without leaking upstream status.

    DNS is resolved and checked for private/link-local addresses before each request.
    The HTTP request uses the original hostname so normal TLS/SNI continues to work.
    """

    current = validate_website_url(url)
    for _ in range(WEBSITE_MAX_REDIRECTS + 1):
        hostname = await _assert_public_resolution(current)
        try:
            response = await client.get(
                current,
                follow_redirects=False,
                timeout=WEBSITE_FETCH_TIMEOUT_SECONDS,
                headers={"User-Agent": "LaunchKitBot/1.0 (+website discovery)"},
            )
        except httpx.HTTPError as exc:
            logger.info("website_fetch_failed url_host=%s error=%s", hostname, type(exc).__name__)
            raise DomainError(_GENERIC_FETCH_ERROR) from exc

        if response.is_redirect:
            location = response.headers.get("location")
            if not location:
                raise DomainError(_GENERIC_FETCH_ERROR)
            current = validate_website_url(urljoin(current, location))
            continue

        if response.status_code >= 400:
            logger.info(
                "website_fetch_http_error url_host=%s status=%s",
                hostname,
                response.status_code,
            )
            raise DomainError(_GENERIC_FETCH_ERROR)

        content_type = response.headers.get("content-type", "").lower()
        if content_type and "html" not in content_type and "text" not in content_type:
            raise DomainError("This address is not a web page, so it cannot be scanned.")
        return response.text[:WEBSITE_MAX_BYTES]

    raise DomainError(_GENERIC_FETCH_ERROR)


def _parse_public_web_url(url: str) -> tuple[str, str]:
    candidate = url.strip()
    if not candidate:
        raise DomainError("Enter your website address.")
    if "://" not in candidate:
        candidate = f"https://{candidate}"
    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"}:
        raise DomainError("The website address must start with http:// or https://.")
    hostname = (parsed.hostname or "").lower()
    if not hostname or "." not in hostname:
        # Single-label hosts (intranet names) are not reachable public sites.
        raise DomainError("Enter a full public website address such as https://example.com.")
    if hostname in _BLOCKED_HOSTNAMES:
        raise DomainError(_PRIVATE_NETWORK_ERROR)
    return candidate, hostname


def _reject_literal_or_rebinding_host(hostname: str) -> None:
    if any(hostname == suffix.lstrip(".") or hostname.endswith(suffix) for suffix in _REBINDING_HOST_SUFFIXES):
        raise DomainError(_PRIVATE_NETWORK_ERROR)
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if address is not None:
        if not address.is_global:
            raise DomainError(_PRIVATE_NETWORK_ERROR)
        return
    # OS resolvers accept decimal/hex/odd dotted forms that ipaddress rejects
    # (e.g. 2130706433, 0x7f.0x0.0x0.0x1). Reject those that land on non-global IPs.
    if _looks_like_encoded_ip(hostname):
        for resolved in _sync_resolve_addresses(hostname):
            if not resolved.is_global:
                raise DomainError(_PRIVATE_NETWORK_ERROR)


def _looks_like_encoded_ip(hostname: str) -> bool:
    if not hostname or any(char.isalpha() and char.lower() not in "abcdefx" for char in hostname):
        return False
    return any(char.isdigit() for char in hostname)


def _sync_resolve_addresses(hostname: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    try:
        infos = socket.getaddrinfo(
            hostname,
            None,
            type=socket.SOCK_STREAM,
            proto=socket.IPPROTO_TCP,
        )
    except OSError:
        return []
    addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    seen: set[str] = set()
    for info in infos:
        raw_ip = info[4][0]
        try:
            address = ipaddress.ip_address(raw_ip)
        except ValueError:
            continue
        key = str(address)
        if key not in seen:
            seen.add(key)
            addresses.append(address)
    return addresses


async def _assert_public_resolution(url: str) -> str:
    """Resolve DNS and reject any non-global address before the HTTP fetch runs."""

    candidate, hostname = _parse_public_web_url(url)
    _reject_literal_or_rebinding_host(hostname)
    parsed = urlparse(candidate)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    try:
        address = ipaddress.ip_address(hostname)
        resolved = [address]
    except ValueError:
        resolved = await _resolve_host_addresses(hostname, port)

    if not resolved:
        raise DomainError(_GENERIC_FETCH_ERROR)
    for address in resolved:
        if not address.is_global:
            raise DomainError(_PRIVATE_NETWORK_ERROR)
    return hostname


async def _resolve_host_addresses(hostname: str, port: int) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(
            hostname,
            port,
            type=socket.SOCK_STREAM,
            proto=socket.IPPROTO_TCP,
        )
    except OSError as exc:
        logger.info("website_dns_failed host=%s error=%s", hostname, type(exc).__name__)
        raise DomainError(_GENERIC_FETCH_ERROR) from exc

    addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    seen: set[str] = set()
    for info in infos:
        sockaddr = info[4]
        raw_ip = sockaddr[0]
        try:
            address = ipaddress.ip_address(raw_ip)
        except ValueError:
            continue
        key = str(address)
        if key not in seen:
            seen.add(key)
            addresses.append(address)
    return addresses
