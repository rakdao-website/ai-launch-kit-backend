"""Build creation, reads, and completed archive access."""

import hashlib
import json
from collections.abc import Awaitable, Callable
from urllib.parse import urlparse

import structlog

from launchkit.assets import AssetBlobStore, safe_filename
from launchkit.builds.models import BuildCreate, BuildEventView, BuildPreviewView, BuildView
from launchkit.builds.state import ACTIVE_BUILD_STATUSES
from launchkit.core.config import Settings
from launchkit.core.exceptions import ConfigurationError, DomainError
from launchkit.generation.models import V0GenerationResult
from launchkit.persistence.models import BuildRecord, StatusEventRecord
from launchkit.persistence.repositories import PersistenceRepository

logger = structlog.get_logger(__name__)

V0StatusLookup = Callable[[str], Awaitable[V0GenerationResult]]


class BuildNotFoundError(DomainError):
    """Raised when a build is absent or owned by another testing user."""


class GenerationQuotaExceededError(DomainError):
    """Raised when the owner has already used their single website generation."""

    def __init__(self) -> None:
        super().__init__(
            "You have already generated your website. "
            "You need more credits to generate another one."
        )


class BuildService:
    def __init__(
        self,
        repository: PersistenceRepository,
        owner_id: str,
        settings: Settings,
        asset_store: AssetBlobStore,
        *,
        v0_status: V0StatusLookup | None = None,
    ) -> None:
        self._repository = repository
        self._owner_id = owner_id
        self._settings = settings
        self._asset_store = asset_store
        self._v0_status = v0_status

    async def start(self, project_id: str, request: BuildCreate, idempotency_key: str) -> BuildView:
        project = await self._repository.get_project(project_id, self._owner_id)
        if project is None:
            raise BuildNotFoundError("Project not found")
        if self._settings.v0_api_key is None:
            raise ConfigurationError("v0 is required for final builds")
        if not project.business.get("companyName", "").strip():
            raise DomainError("Add a company name before starting a build.")
        if project.selected_mockup_id is None:
            raise DomainError("Select a mockup before starting a build.")
        if await self._repository.get_mockup(project.selected_mockup_id, project.id) is None:
            raise DomainError("The selected mockup is no longer available.")

        normalized_key = idempotency_key.strip()
        if not normalized_key or len(normalized_key) > 200:
            raise DomainError("A valid Idempotency-Key header is required.")
        scope = f"projects:{project_id}:builds"
        key_hash = hashlib.sha256(normalized_key.encode()).hexdigest()
        request_hash = hashlib.sha256(
            json.dumps(
                [
                    request.model_dump(mode="json"),
                    project.business,
                    project.design,
                    project.page_layout,
                    project.selected_mockup_id,
                ],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        existing = await self._repository.get_idempotency(
            owner_id=self._owner_id, scope=scope, key_hash=key_hash
        )
        if existing is not None:
            if existing.request_hash != request_hash:
                raise DomainError("This idempotency key was already used for a different build.")
            build = await self._repository.get_build(existing.resource_id, self._owner_id)
            if build is None:
                raise BuildNotFoundError("Build not found")
            return build_view(build)

        if await self._repository.find_active_build(project_id) is not None:
            raise DomainError("A build is already active for this project.")
        # Every user gets exactly one website generation (failed builds don't count),
        # unless the generation quota is disabled for this environment.
        unlimited = self._settings.is_generation_quota_disabled
        logger.info(
            "generation_quota_checked",
            owner_id=self._owner_id,
            unlimited=unlimited,
            quota_disabled=unlimited,
        )
        if not unlimited and await self._repository.count_owner_website_builds(self._owner_id) >= 1:
            raise GenerationQuotaExceededError()
        build = await self._repository.add_build(project_id=project.id, provider=request.provider)
        await self._repository.add_status_event(
            resource_type="build",
            resource_id=build.id,
            from_status=None,
            to_status="queued",
            stage="queued",
            message="Build queued",
        )
        await self._repository.enqueue_job("build.submit", {"buildId": build.id})
        await self._repository.add_idempotency(
            owner_id=self._owner_id,
            scope=scope,
            key_hash=key_hash,
            request_hash=request_hash,
            resource_type="build",
            resource_id=build.id,
        )
        project.latest_build_id = build.id
        project.status = "build_queued"
        await self._repository.commit()
        return build_view(build)

    async def get(self, build_id: str) -> BuildView:
        build = await self._repository.get_build(build_id, self._owner_id)
        if build is None:
            raise BuildNotFoundError("Build not found")
        return build_view(build)

    async def download(self, build_id: str) -> tuple[bytes, str]:
        build = await self._repository.get_build(build_id, self._owner_id)
        if build is None:
            raise BuildNotFoundError("Build not found")
        if build.status != "completed" or build.archive_asset_id is None:
            raise DomainError("The build is not ready to download.")
        asset = await self._repository.get_asset(build.archive_asset_id, self._owner_id)
        if asset is None:
            raise BuildNotFoundError("Build archive not found")
        return await self._asset_store.get(asset.storage_key), safe_filename(
            asset.filename, "website.zip"
        )

    async def preview(self, build_id: str) -> BuildPreviewView:
        """Return a fresh v0 demo URL safe to open in a browser or iframe.

        v0 attaches short-lived ``__v0_token`` query params that often 404 when
        embedded or opened directly. The bare ``*.vusercontent.net`` host is the
        stable public preview URL.
        """
        build = await self._repository.get_build(build_id, self._owner_id)
        if build is None:
            raise BuildNotFoundError("Build not found")
        if build.status != "completed":
            raise DomainError("The website preview is not ready yet.")

        chat = await self._repository.get_provider_reference(
            resource_type="build",
            resource_id=build.id,
            provider="v0",
            reference_type="chat_id",
        )
        if chat is None:
            url = public_preview_url(build.preview_url)
            if url is None:
                raise DomainError("No preview is available for this website yet.")
            return BuildPreviewView(url=url)

        if self._v0_status is None:
            raise ConfigurationError("v0 is required to refresh the website preview")

        result = await self._v0_status(chat.reference_value)
        raw = safe_provider_url(result.demo_url) or safe_provider_url(build.preview_url)
        url = public_preview_url(raw)
        if url is None:
            raise DomainError("v0 did not return a live preview URL for this website.")

        if build.preview_url != url:
            build.preview_url = url
            await self._repository.commit()
        return BuildPreviewView(url=url)


def safe_provider_url(value: str | None) -> str | None:
    if value is None:
        return None
    parsed = urlparse(value)
    return value if parsed.scheme == "https" and parsed.netloc else None


def public_preview_url(value: str | None) -> str | None:
    """Strip v0 token query params; bare vusercontent hosts are the stable preview."""

    url = safe_provider_url(value)
    if url is None:
        return None
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host == "vusercontent.net" or host.endswith(".vusercontent.net"):
        return f"{parsed.scheme}://{parsed.netloc}/"
    return url


def build_view(record: BuildRecord) -> BuildView:
    active = record.status in ACTIVE_BUILD_STATUSES
    return BuildView(
        id=record.id,
        project_id=record.project_id,
        provider=record.provider,
        status=record.status,
        stage=record.stage,
        message=record.message,
        warnings=record.warnings,
        preview_url=public_preview_url(record.preview_url),
        web_url=record.web_url,
        download_url=f"/api/v1/builds/{record.id}/download"
        if record.status == "completed"
        else None,
        retry_after_seconds=5 if active else None,
        created_at=record.created_at,
        updated_at=record.updated_at,
        submitted_at=record.submitted_at,
        completed_at=record.completed_at,
    )


def event_view(record: StatusEventRecord) -> BuildEventView:
    return BuildEventView(
        id=record.sequence,
        status=record.to_status,
        stage=record.stage,
        message=record.message,
        created_at=record.created_at,
    )
