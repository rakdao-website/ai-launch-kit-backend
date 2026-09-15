"""Shared exception hierarchy independent of HTTP transports."""


class LaunchKitError(Exception):
    """Base class for expected backend failures."""


class DomainError(LaunchKitError):
    """Raised when domain input or state violates a business rule."""


class ApplicationError(LaunchKitError):
    """Raised when an application capability cannot complete."""


class ConfigurationError(ApplicationError):
    """Raised when required runtime configuration is invalid or missing."""


class AuthenticationError(ApplicationError):
    """Raised when staging credentials or access tokens are invalid."""


class RateLimitError(ApplicationError):
    """Raised when a caller exceeds an application rate limit."""


class ProviderError(ApplicationError):
    """Normalized external-provider failure without SDK-specific objects."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        provider_name: str | None = None,
        retryable: bool = False,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.provider_name = provider_name
        self.retryable = retryable
        self.retry_after = retry_after
