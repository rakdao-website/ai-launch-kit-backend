"""Durable worker handlers for profile extraction and mockup generation."""

import base64
import hashlib
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from launchkit.adapters.llm_queue import RequestQueue
from launchkit.adapters.openrouter import OpenRouterAdapter
from launchkit.adapters.pexels import PexelsAdapter
from launchkit.assets import AssetBlobStore, project_asset_key, safe_filename
from launchkit.core.config import Settings
from launchkit.core.exceptions import ConfigurationError, DomainError, ProviderError
from launchkit.generation.briefing import BriefService
from launchkit.generation.html_generation import HtmlGenerationService
from launchkit.generation.mockups import MockupGenerationService
from launchkit.persistence.models import AssetRecord, JobRecord, OperationRecord, ProjectRecord
from launchkit.persistence.repositories import PersistenceRepository
from launchkit.profiles import ExtractedImage, ProfileExtractionService
from launchkit.profiles.website import fetch_website_html, website_page_text
from launchkit.projects.catalogs import BUSINESS_CATEGORIES
from launchkit.projects.grounding import business_form_for_generation, merge_empty
from launchkit.projects.models import DesignDraft
from launchkit.workflows.service import asset_view, mockup_view

CATEGORY_LABELS = {item.id: item.label for item in BUSINESS_CATEGORIES}


class WorkflowJobHandlers:
    def __init__(
        self,
        asset_store: AssetBlobStore,
        *,
        profile_service: ProfileExtractionService | None,
        mockup_service: MockupGenerationService | None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._asset_store = asset_store
        self._profile_service = profile_service
        self._mockup_service = mockup_service
        self._http_client = http_client

    @property
    def handlers(
        self,
    ) -> dict[str, Callable[[JobRecord, AsyncSession], Awaitable[None]]]:
        return {
            "profile.extract": self.profile_extract,
            "website.extract": self.website_extract,
            "mockups.generate": self.generate_mockups,
        }

    async def profile_extract(self, job: JobRecord, session: AsyncSession) -> None:
        operation = await self._start(job, session)
        try:
            if self._profile_service is None:
                raise ConfigurationError("OpenRouter is not configured")
            project = await session.get(ProjectRecord, str(job.payload["projectId"]))
            if project is None:
                raise RuntimeError("Profile workflow resources are missing")
            project.extracted_profile_fields = {}
            repository = PersistenceRepository(session)
            requested_id = str(job.payload.get("assetId") or "")
            assets = [
                asset
                for asset in await repository.list_assets(project.id)
                if asset.kind == "profile_source"
            ]
            # Prefer all brand documents on the project so multi-file uploads feed one brief.
            if not assets and requested_id:
                asset = await session.get(AssetRecord, requested_id)
                if asset is not None and asset.kind == "profile_source":
                    assets = [asset]
            if not assets:
                raise RuntimeError("Profile workflow resources are missing")

            files: list[tuple[bytes, str]] = []
            for asset in assets:
                content = await self._asset_store.get(asset.storage_key)
                files.append((content, asset.filename))

            result = await self._profile_service.extract_many(files)
            image_views: list[dict[str, object]] = []
            for image in result.images:
                image_content, content_type = decode_data_url(image.data_url)
                filename = safe_filename(image.filename, "profile-image")
                storage_key = project_asset_key(
                    project.owner_id,
                    project.id,
                    "profile-images",
                    f"{uuid.uuid4().hex}-{filename}",
                )
                await self._asset_store.put(storage_key, image_content, content_type)
                record = await repository.add_asset(
                    project_id=project.id,
                    kind="profile_image",
                    storage_key=storage_key,
                    filename=filename,
                    label=image.label,
                    content_type=content_type,
                    size=len(image_content),
                    sha256=hashlib.sha256(image_content).hexdigest(),
                )
                image_views.append(asset_view(record).model_dump(by_alias=True))

            fields = result.fields.model_dump(by_alias=True, exclude_none=True)
            hints = result.design_hints.model_dump(by_alias=True, exclude_none=True)
            # Full replace so re-runs are not mixed with a previous document / website extract.
            project.extracted_profile_fields = {
                **fields,
                **{key: value for key, value in hints.items() if value},
            }
            operation.result = {
                "fields": fields,
                "designHints": hints,
                "assets": image_views,
                "sourceFilename": result.source_filename,
                "warnings": result.warnings,
            }
            self._complete(operation)
        except Exception as exc:
            self._fail(operation, exc)
            raise

    async def website_extract(self, job: JobRecord, session: AsyncSession) -> None:
        """Scrape an optional website first, then brand documents, into one AI brief."""

        operation = await self._start(job, session)
        try:
            if self._profile_service is None:
                raise ConfigurationError("OpenRouter is not configured")
            project = await session.get(ProjectRecord, str(job.payload["projectId"]))
            if project is None:
                raise RuntimeError("Website discovery resources are missing")

            url = str(job.payload.get("url") or "").strip()
            # Drop prior extract so a partial failure cannot leave stale document-only brief.
            project.extracted_profile_fields = {}
            website_text: str | None = None
            if url:
                if self._http_client is None:
                    raise ConfigurationError("OpenRouter is not configured")
                html = await fetch_website_html(self._http_client, url)
                scraped = website_page_text(html)
                website_text = scraped if scraped.strip() else None

            repository = PersistenceRepository(session)
            assets = [
                asset
                for asset in await repository.list_assets(project.id)
                if asset.kind == "profile_source"
            ]
            files: list[tuple[bytes, str]] = []
            for asset in assets:
                content = await self._asset_store.get(asset.storage_key)
                files.append((content, asset.filename))

            if not url and not files:
                raise DomainError("Upload brand documents or enter a website address.")

            result = await self._profile_service.extract_many(
                files,
                website_text=website_text,
                website_url=url or None,
            )
            image_views: list[dict[str, object]] = []
            for image in result.images:
                image_content, content_type = decode_data_url(image.data_url)
                filename = safe_filename(image.filename, "profile-image")
                storage_key = project_asset_key(
                    project.owner_id,
                    project.id,
                    "profile-images",
                    f"{uuid.uuid4().hex}-{filename}",
                )
                await self._asset_store.put(storage_key, image_content, content_type)
                record = await repository.add_asset(
                    project_id=project.id,
                    kind="profile_image",
                    storage_key=storage_key,
                    filename=filename,
                    label=image.label,
                    content_type=content_type,
                    size=len(image_content),
                    sha256=hashlib.sha256(image_content).hexdigest(),
                )
                image_views.append(asset_view(record).model_dump(by_alias=True))

            fields = result.fields.model_dump(by_alias=True, exclude_none=True)
            hints = result.design_hints.model_dump(by_alias=True, exclude_none=True)
            # Full replace — each AI Summary run reflects only the current URL + documents.
            # Do not merge into business/design here: that bled prior extracts into later runs.
            project.extracted_profile_fields = {
                **fields,
                **{key: value for key, value in hints.items() if value},
            }
            operation.result = {
                "fields": fields,
                "designHints": hints,
                "assets": image_views,
                "sourceUrl": url or None,
                "sourceFilename": result.source_filename,
                "warnings": result.warnings,
            }
            self._complete(operation)
        except Exception as exc:
            self._fail(operation, exc)
            raise

    async def generate_mockups(self, job: JobRecord, session: AsyncSession) -> None:
        operation = await self._start(job, session)
        try:
            if self._mockup_service is None:
                raise ConfigurationError("OpenRouter is not configured")
            project = await session.get(ProjectRecord, str(job.payload["projectId"]))
            if project is None:
                raise RuntimeError("Mockup project is missing")
            repository = PersistenceRepository(session)
            form = business_form_for_generation(
                project.business,
                project.extracted_profile_fields,
                category_fallback_industry=CATEGORY_LABELS.get(
                    str(project.business.get("categoryId") or "tech-saas"),
                    str(project.business.get("categoryId") or "tech-saas"),
                ),
            )
            design = DesignDraft.model_validate(project.design).to_preferences()
            uploaded = await self._uploaded_images(repository, project.id)
            generated = await self._mockup_service.generate(form, design, uploaded)
            generation = await repository.next_mockup_generation(project.id)
            views: list[dict[str, object]] = []
            for ordinal, mockup in enumerate(generated.mockups, start=1):
                content = mockup.html.encode("utf-8")
                filename = f"mockup-{generation}-{ordinal}.html"
                storage_key = project_asset_key(
                    project.owner_id,
                    project.id,
                    "mockups",
                    f"{uuid.uuid4().hex}-{filename}",
                )
                await self._asset_store.put(storage_key, content, "text/html; charset=utf-8")
                asset = await repository.add_asset(
                    project_id=project.id,
                    kind="mockup_html",
                    storage_key=storage_key,
                    filename=filename,
                    label=mockup.label,
                    content_type="text/html; charset=utf-8",
                    size=len(content),
                    sha256=hashlib.sha256(content).hexdigest(),
                )
                record = await repository.add_mockup(
                    project_id=project.id,
                    generation=generation,
                    ordinal=ordinal,
                    label=mockup.label,
                    direction=mockup.direction,
                    artifact_asset_id=asset.id,
                )
                views.append(mockup_view(record).model_dump(by_alias=True, mode="json"))
            project.status = "mockups_ready"
            operation.result = {"mockups": views, "brief": generated.brief}
            self._complete(operation)
        except Exception as exc:
            self._fail(operation, exc)
            raise

    async def _uploaded_images(
        self, repository: PersistenceRepository, project_id: str
    ) -> list[ExtractedImage]:
        images: list[ExtractedImage] = []
        for asset in await repository.list_assets(project_id):
            if asset.kind != "profile_image":
                continue
            content = await self._asset_store.get(asset.storage_key)
            encoded = base64.b64encode(content).decode("ascii")
            images.append(
                ExtractedImage(
                    filename=asset.filename,
                    label=asset.label,
                    data_url=f"data:{asset.content_type};base64,{encoded}",
                )
            )
        return images

    @staticmethod
    async def _start(job: JobRecord, session: AsyncSession) -> OperationRecord:
        if job.operation_id is None:
            raise RuntimeError("Workflow job has no operation")
        operation = await session.get(OperationRecord, job.operation_id)
        if operation is None:
            raise RuntimeError("Workflow operation is missing")
        operation.status = "running"
        operation.error_code = None
        operation.error_message = None
        return operation

    @staticmethod
    def _complete(operation: OperationRecord) -> None:
        operation.status = "completed"
        operation.completed_at = datetime.now(UTC)

    @staticmethod
    def _fail(operation: OperationRecord, exc: Exception) -> None:
        operation.status = "failed"
        operation.completed_at = datetime.now(UTC)
        if isinstance(exc, ConfigurationError):
            operation.error_code = "provider_configuration_missing"
            operation.error_message = "The generation service is not configured."
        elif isinstance(exc, DomainError):
            # Domain messages are written for end users (e.g. unreachable website).
            operation.error_code = "invalid_request"
            operation.error_message = str(exc)
        elif isinstance(exc, ProviderError):
            operation.error_code = "provider_unavailable"
            operation.error_message = "The generation service could not complete this operation."
        else:
            operation.error_code = "operation_failed"
            operation.error_message = "The operation could not be completed."


def create_workflow_job_handlers(
    settings: Settings, client: httpx.AsyncClient, asset_store: AssetBlobStore
) -> WorkflowJobHandlers:
    api_key = settings.openrouter_api_key.get_secret_value() if settings.openrouter_api_key else ""
    if not api_key:
        return WorkflowJobHandlers(
            asset_store, profile_service=None, mockup_service=None, http_client=client
        )
    queue = RequestQueue(
        max_concurrent=settings.openrouter_max_concurrent,
        min_gap_seconds=settings.openrouter_min_request_gap_ms / 1000,
    )
    openrouter = OpenRouterAdapter(
        client,
        queue,
        api_key=api_key,
        base_url=settings.openrouter_base_url,
        generation_model=settings.generation_model,
        utility_model=settings.utility_model,
        image_model=settings.image_model,
        site_url=settings.site_url,
        app_title=settings.openrouter_app_title,
        max_attempts=settings.openrouter_retry_attempts,
    )
    pexels_key = settings.pexels_api_key.get_secret_value() if settings.pexels_api_key else None
    pexels = PexelsAdapter(client, api_key=pexels_key, base_url=settings.pexels_base_url)
    profile = ProfileExtractionService(openrouter, openrouter, openrouter)
    mockups = MockupGenerationService(
        BriefService(openrouter),
        HtmlGenerationService(openrouter),
        image_generator=openrouter,
        image_search=pexels,
    )
    return WorkflowJobHandlers(
        asset_store, profile_service=profile, mockup_service=mockups, http_client=client
    )


def decode_data_url(value: str) -> tuple[bytes, str]:
    header, encoded = value.split(",", maxsplit=1)
    if not header.startswith("data:") or ";base64" not in header:
        raise ValueError("Invalid extracted image data URL")
    content_type = header[5:].split(";", maxsplit=1)[0]
    return base64.b64decode(encoded, validate=True), content_type


__all__ = ("WorkflowJobHandlers", "create_workflow_job_handlers", "decode_data_url", "merge_empty")
