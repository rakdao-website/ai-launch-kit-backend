"""Use operating-system trust stores for outbound TLS (corporate proxies re-sign certs)."""

from __future__ import annotations

import ssl
from typing import Any

import structlog


def outbound_verify() -> Any:
    """Return an httpx ``verify=`` value that trusts the OS certificate store.

    Uses a truststore SSLContext without calling ``inject_into_ssl()``, so it is
    safe alongside boto3/S3 (global injection recurses inside botocore).
    """

    try:
        import truststore
    except ImportError:
        structlog.get_logger(__name__).warning("truststore_not_installed")
        return True
    return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


def use_system_certificates(*, enabled: bool = True) -> None:
    """Make ssl.SSLContext chain to the OS store so proxy CAs are trusted.

    Must stay disabled when boto3/S3 is in use: ``truststore.inject_into_ssl()``
    recurses inside botocore's urllib3 SSLContext setup (RecursionError on startup).
    Prefer ``outbound_verify()`` for httpx clients when S3 is enabled.
    """

    if not enabled:
        structlog.get_logger(__name__).info(
            "truststore_skipped", reason="incompatible_with_boto3_s3"
        )
        return
    try:
        import truststore
    except ImportError:
        structlog.get_logger(__name__).warning("truststore_not_installed")
        return
    truststore.inject_into_ssl()
