"""Use operating-system trust stores for outbound TLS (corporate proxies re-sign certs)."""

import structlog


def use_system_certificates(*, enabled: bool = True) -> None:
    """Make ssl.SSLContext chain to the OS store so proxy CAs are trusted.

    Must stay disabled when boto3/S3 is in use: ``truststore.inject_into_ssl()``
    recurses inside botocore's urllib3 SSLContext setup (RecursionError on startup).
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
