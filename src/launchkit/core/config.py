"""Environment-backed application configuration."""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["local", "test", "staging", "production"]
AuthMode = Literal["testing", "fixed_otp", "oauth"]


class Settings(BaseSettings):
    """Runtime settings loaded from ``LAUNCHKIT_`` environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="LAUNCHKIT_",
        extra="ignore",
    )

    app_name: str = "AI Launch Kit Backend"
    environment: Environment = "local"
    debug: bool = False
    log_level: str = Field(default="INFO", pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    log_json: bool = False
    database_url: str = "postgresql+asyncpg://launchkit:launchkit@localhost:5432/launchkit"
    database_echo: bool = False
    testing_user_id: str = "user_testing"
    # oauth: InnovationCity PKCE (required for staging/production).
    # fixed_otp: local/test only — never ship a static OTP to shared environments.
    # testing: bypass auth with testing_user_id (unit/integration harnesses).
    auth_mode: AuthMode = "testing"
    auth_email: str = "test@innovationcity.com"
    auth_otp: SecretStr | None = None
    auth_token_secret: SecretStr | None = None
    auth_token_ttl_seconds: int = Field(default=8 * 60 * 60, ge=300, le=7 * 24 * 60 * 60)
    frontend_origins: str = (
        "http://localhost:5173,"
        "https://ai-launch-kitt-git-codex-aws-dokploy-readiness-innovation-city.vercel.app"
    )
    worker_poll_seconds: float = Field(default=1.0, gt=0)
    worker_lease_seconds: int = Field(default=120, ge=10)
    worker_batch_size: int = Field(default=10, ge=1, le=100)
    upload_max_bytes: int = Field(default=20 * 1024 * 1024, ge=1024)
    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    site_url: str = "http://localhost:8000"
    openrouter_app_title: str = "LaunchKit Generator"
    generation_model: str = "anthropic/claude-sonnet-4.6"
    utility_model: str | None = "openai/gpt-4o-mini"
    image_model: str = "google/gemini-2.5-flash-image"
    pexels_api_key: SecretStr | None = None
    pexels_base_url: str = "https://api.pexels.com/v1"
    openrouter_max_concurrent: int = Field(default=2, ge=1)
    openrouter_min_request_gap_ms: int = Field(default=250, ge=0)
    openrouter_sequential: bool = False
    openrouter_retry_attempts: int = Field(default=6, ge=1)
    v0_api_key: SecretStr | None = None
    v0_base_url: str = "https://api.v0.dev/v1"
    v0_model: str = "v0-max"
    v0_webhook_token: SecretStr | None = None
    v0_webhook_callback_url: str | None = None
    webhook_max_bytes: int = Field(default=1024 * 1024, ge=1024)
    build_reconcile_initial_seconds: int = Field(default=15, ge=1)
    build_reconcile_max_seconds: int = Field(default=900, ge=1)
    build_timeout_seconds: int = Field(default=3600, ge=60)
    sse_poll_seconds: float = Field(default=1.0, gt=0)
    sse_heartbeat_seconds: float = Field(default=15.0, gt=0)
    local_data_dir: Path = Path("local_data")
    s3_bucket: str | None = None
    s3_prefix: str = "submissions/"
    s3_asset_prefix: str = "assets/"
    aws_region: str = "me-central-1"
    vercel_token: SecretStr | None = None
    vercel_team_id: str | None = Field(default=None, pattern=r"^team_[A-Za-z0-9]+$")
    vercel_base_url: str = "https://api.vercel.com"
    vercel_webhook_secret: SecretStr | None = None
    claim_return_url: str = "http://localhost:5173/"
    # Turns the one-website-per-user limit off for everyone (testing environments only).
    disable_generation_quota: bool = False

    @property
    def is_generation_quota_disabled(self) -> bool:
        """Quota is off when explicitly flagged, or on local/test (UAT uses environment=test)."""

        return self.disable_generation_quota or self.environment in {"local", "test"}

    @field_validator(
        "auth_otp",
        "auth_token_secret",
        "openrouter_api_key",
        "pexels_api_key",
        "v0_api_key",
        "v0_webhook_token",
        "vercel_token",
        "vercel_webhook_secret",
        "utility_model",
        "v0_webhook_callback_url",
        "s3_bucket",
        "vercel_team_id",
        mode="before",
    )
    @classmethod
    def empty_values_are_unset(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def reject_fixed_otp_outside_local_test(self) -> Self:
        """Fixed OTP must never be reachable on shared/public deployments."""

        if self.auth_mode != "fixed_otp":
            return self
        if self.environment not in {"local", "test"}:
            raise ValueError(
                "LAUNCHKIT_AUTH_MODE=fixed_otp is only allowed when "
                "LAUNCHKIT_ENVIRONMENT is local or test; use oauth for shared environments"
            )
        site = (self.site_url or "").strip().lower()
        public_https = site.startswith("https://") and not any(
            marker in site for marker in ("localhost", "127.0.0.1", "[::1]")
        )
        if public_https:
            raise ValueError(
                "LAUNCHKIT_AUTH_MODE=fixed_otp cannot be used with a public "
                "LAUNCHKIT_SITE_URL; use oauth instead"
            )
        return self

    # InnovationCity OAuth PKCE (app-auth-service-nodejs).
    # Register client_id + redirect_uri + post_logout_redirect_uri with WeCan.
    auth_base_url: str = "https://app-sandbox.innovationcity.com"
    auth_client_id: str | None = None
    auth_redirect_uri: str = "http://localhost:8000/auth/callback"
    auth_post_logout_redirect_uri: str = "http://localhost:5173/?auth=logged_out"
    auth_frontend_url: str = "http://localhost:5173"
    auth_session_secret: str = "dev-only-change-me"
    auth_cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"


@lru_cache
def get_settings() -> Settings:
    """Return one settings instance per process."""

    return Settings()
