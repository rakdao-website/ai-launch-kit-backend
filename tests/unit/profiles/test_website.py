"""Website discovery: URL validation, SSRF guards, and HTML-to-text reduction."""

import asyncio
from ipaddress import IPv4Address
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from launchkit.core.exceptions import DomainError
from launchkit.profiles.website import fetch_website_html, validate_website_url, website_page_text


class TestValidateWebsiteUrl:
    def test_accepts_public_https_urls(self) -> None:
        assert validate_website_url("https://example.com/menu") == "https://example.com/menu"

    def test_prepends_https_to_bare_domains(self) -> None:
        assert validate_website_url("example.com") == "https://example.com"

    def test_rejects_empty_input(self) -> None:
        with pytest.raises(DomainError):
            validate_website_url("   ")

    def test_rejects_non_web_schemes(self) -> None:
        with pytest.raises(DomainError):
            validate_website_url("ftp://example.com")

    def test_rejects_localhost_and_private_addresses(self) -> None:
        for target in (
            "localhost",
            "http://localhost:8000",
            "http://127.0.0.1",
            "http://10.0.0.5",
            "http://169.254.169.254/",
            "http://0x7f.0x0.0x0.0x1/",
        ):
            with pytest.raises(DomainError):
                validate_website_url(target)

    def test_rejects_dns_rebinding_wildcard_hosts(self) -> None:
        for target in (
            "http://169-254-169-254.sslip.io/",
            "http://169.254.169.254.nip.io/",
            "https://10.0.0.1.xip.io/path",
            "http://localtest.me/",
        ):
            with pytest.raises(DomainError):
                validate_website_url(target)

    def test_rejects_single_label_intranet_hosts(self) -> None:
        with pytest.raises(DomainError):
            validate_website_url("http://intranet")


class TestFetchWebsiteHtml:
    def test_pins_resolved_public_ip_and_hides_status_codes(self) -> None:
        request_url: str | None = None

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal request_url
            request_url = str(request.url)
            assert request.headers["host"] == "example.com"
            return httpx.Response(404, text="missing")

        async def scenario() -> None:
            transport = httpx.MockTransport(handler)
            async with httpx.AsyncClient(transport=transport) as client:
                with patch(
                    "launchkit.profiles.website._resolve_host_addresses",
                    new=AsyncMock(return_value=[IPv4Address("93.184.216.34")]),
                ):
                    await fetch_website_html(client, "https://example.com/page")

        with pytest.raises(DomainError, match="could not be processed"):
            asyncio.run(scenario())
        assert request_url == "https://93.184.216.34/page"

    def test_rejects_hosts_that_resolve_to_private_addresses(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("fetch must not run for private resolutions")

        async def scenario() -> None:
            transport = httpx.MockTransport(handler)
            async with httpx.AsyncClient(transport=transport) as client:
                with patch(
                    "launchkit.profiles.website._resolve_host_addresses",
                    new=AsyncMock(return_value=[IPv4Address("169.254.169.254")]),
                ):
                    await fetch_website_html(client, "https://metadata.example.com/")

        with pytest.raises(DomainError, match="private network"):
            asyncio.run(scenario())

    def test_revalidates_redirect_targets(self) -> None:
        calls: list[str] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            if request.url.host == "93.184.216.34":
                return httpx.Response(302, headers={"location": "http://127.0.0.1/secret"})
            raise AssertionError("redirect into a private address must not be followed")

        async def scenario() -> None:
            transport = httpx.MockTransport(handler)
            async with httpx.AsyncClient(transport=transport) as client:
                with patch(
                    "launchkit.profiles.website._resolve_host_addresses",
                    new=AsyncMock(return_value=[IPv4Address("93.184.216.34")]),
                ):
                    await fetch_website_html(client, "https://example.com/")

        with pytest.raises(DomainError, match="private network"):
            asyncio.run(scenario())
        assert calls == ["https://93.184.216.34/"]


class TestWebsitePageText:
    def test_extracts_title_meta_and_body_copy(self) -> None:
        html = """
        <html><head>
          <title>Lucknow Cuisine</title>
          <meta name="description" content="Authentic Awadhi food in the city centre.">
          <style>body { color: red; }</style>
          <script>console.log("skip me");</script>
        </head><body>
          <h1>Welcome to Lucknow Cuisine</h1>
          <p>Family recipes since 1985.</p>
          <img src="hero.jpg" alt="Chef plating biryani">
        </body></html>
        """
        text = website_page_text(html)
        assert "Page title: Lucknow Cuisine" in text
        assert "Meta description: Authentic Awadhi food in the city centre." in text
        assert "Welcome to Lucknow Cuisine" in text
        assert "Family recipes since 1985." in text
        assert "Chef plating biryani" in text
        assert "skip me" not in text
        assert "color: red" not in text

    def test_returns_empty_string_for_empty_page(self) -> None:
        assert website_page_text("<html><body></body></html>") == ""
