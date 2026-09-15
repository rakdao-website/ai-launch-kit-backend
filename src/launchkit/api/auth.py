"""Fixed-account OTP authentication for local/test only.

Staging and production must use InnovationCity OAuth (``auth_mode=oauth``).
Fixed OTP never falls back to a hardcoded code and is rejected outside local/test.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime, timedelta
from secrets import compare_digest
from threading import Lock
from time import monotonic
from typing import Any

import jwt
from jwt import InvalidTokenError
from pydantic import Field, SecretStr

from launchkit.core.config import Settings
from launchkit.core.exceptions import AuthenticationError, ConfigurationError, RateLimitError
from launchkit.core.models import AliasedModel

LOCAL_TOKEN_SECRET = "launchkit-local-development-token-secret"
TOKEN_ALGORITHM = "HS256"
TOKEN_AUDIENCE = "ai-launch-kit-api"
TOKEN_ISSUER = "ai-launch-kit"
API_TOKEN_COOKIE = "lk_api_token"

# Never accept these as credentials, even if present in env.
_FORBIDDEN_OTPS = frozenset(
    {
        "123456",
        "000000",
        "111111",
        "654321",
        "999999",
        "121212",
        "123123",
    }
)

_AUTH_RATE_LIMIT_ATTEMPTS = 5
_AUTH_RATE_LIMIT_WINDOW_SECONDS = 300.0


class AccessCodeRequest(AliasedModel):
    email: str = Field(min_length=3, max_length=254)


class AccessCodeResponse(AliasedModel):
    status: str = "accepted"


class VerifyAccessCodeRequest(AccessCodeRequest):
    code: str = Field(pattern=r"^\d{6}$")


class AuthTokenResponse(AliasedModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_seconds: int


class _SlidingWindowLimiter:
    """Process-local attempt throttle for OTP endpoints."""

    def __init__(self, *, max_attempts: int, window_seconds: float) -> None:
        self._max_attempts = max_attempts
        self._window_seconds = window_seconds
        self._hits: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    def check(self, key: str) -> None:
        now = monotonic()
        with self._lock:
            recent = [stamp for stamp in self._hits[key] if now - stamp < self._window_seconds]
            if len(recent) >= self._max_attempts:
                raise RateLimitError("Too many authentication attempts. Try again later.")
            recent.append(now)
            self._hits[key] = recent


_request_limiter = _SlidingWindowLimiter(
    max_attempts=_AUTH_RATE_LIMIT_ATTEMPTS,
    window_seconds=_AUTH_RATE_LIMIT_WINDOW_SECONDS,
)
_verify_limiter = _SlidingWindowLimiter(
    max_attempts=_AUTH_RATE_LIMIT_ATTEMPTS,
    window_seconds=_AUTH_RATE_LIMIT_WINDOW_SECONDS,
)


def request_access_code(settings: Settings, email: str) -> AccessCodeResponse:
    _require_fixed_otp_mode(settings)
    normalized = _verify_email(settings, email)
    _request_limiter.check(f"request:{normalized}")
    # OTP is preconfigured for local/test tooling; never disclose it in the response.
    _configured_otp(settings)
    return AccessCodeResponse()


def verify_access_code(settings: Settings, email: str, code: str) -> AuthTokenResponse:
    _require_fixed_otp_mode(settings)
    normalized_email = _verify_email(settings, email)
    _verify_limiter.check(f"verify:{normalized_email}")
    expected_code = _configured_otp(settings)
    if not compare_digest(code, expected_code):
        raise AuthenticationError("The email or access code is invalid.")
    return mint_token(settings, normalized_email)


def mint_token(settings: Settings, subject: str) -> AuthTokenResponse:
    """Issue a Launch Kit API JWT for an already-authenticated subject."""

    now = datetime.now(UTC)
    expires_at = now + timedelta(seconds=settings.auth_token_ttl_seconds)
    payload: dict[str, Any] = {
        "sub": subject,
        "aud": TOKEN_AUDIENCE,
        "iss": TOKEN_ISSUER,
        "iat": now,
        "exp": expires_at,
    }
    secret = _secret_value(
        settings,
        settings.auth_token_secret,
        LOCAL_TOKEN_SECRET,
        "authentication token secret",
    )
    token = jwt.encode(payload, secret, algorithm=TOKEN_ALGORITHM)
    return AuthTokenResponse(
        access_token=token,
        expires_in_seconds=settings.auth_token_ttl_seconds,
    )


def authenticate_token(settings: Settings, token: str) -> str:
    payload = _decode_token(settings, token)
    subject = payload.get("sub")
    if not isinstance(subject, str) or not subject:
        raise AuthenticationError("The access session is invalid or expired.")
    return subject


def _decode_token(settings: Settings, token: str) -> dict[str, Any]:
    secret = _secret_value(
        settings,
        settings.auth_token_secret,
        LOCAL_TOKEN_SECRET,
        "authentication token secret",
    )
    try:
        payload: dict[str, Any] = jwt.decode(
            token,
            secret,
            algorithms=[TOKEN_ALGORITHM],
            audience=TOKEN_AUDIENCE,
            issuer=TOKEN_ISSUER,
        )
    except InvalidTokenError as exc:
        raise AuthenticationError("The access session is invalid or expired.") from exc
    return payload


def _require_fixed_otp_mode(settings: Settings) -> None:
    if settings.auth_mode != "fixed_otp":
        raise AuthenticationError("Email access-code login is not enabled.")
    if settings.environment not in {"local", "test"}:
        raise ConfigurationError(
            "fixed_otp authentication is only allowed in local and test environments"
        )


def _verify_email(settings: Settings, email: str) -> str:
    normalized = email.strip().lower()
    if not compare_digest(normalized, settings.auth_email.strip().lower()):
        raise AuthenticationError("The email or access code is invalid.")
    return normalized


def _configured_otp(settings: Settings) -> str:
    configured = settings.auth_otp
    if configured is None:
        raise ConfigurationError("The staging OTP is not configured")
    value = configured.get_secret_value().strip()
    if not value:
        raise ConfigurationError("The staging OTP is not configured")
    if value in _FORBIDDEN_OTPS:
        raise ConfigurationError(
            "The configured OTP is a known weak value and cannot be used"
        )
    if not value.isdigit() or len(value) != 6:
        raise ConfigurationError("The staging OTP must be a 6-digit code")
    return value


def _secret_value(
    settings: Settings,
    configured: SecretStr | None,
    local_default: str,
    label: str,
) -> str:
    if configured is not None:
        value = configured.get_secret_value()
        if value:
            return value
    if settings.environment in {"local", "test"}:
        return local_default
    raise ConfigurationError(f"The {label} is not configured")
