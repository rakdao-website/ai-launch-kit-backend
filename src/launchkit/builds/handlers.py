"""Durable one-shot submission and repeatable v0 reconciliation handlers."""

import base64
import hashlib
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Protocol

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from launchkit.adapters.llm_queue import RequestQueue
from launchkit.adapters.openrouter import OpenRouterAdapter
from launchkit.adapters.pexels import PexelsAdapter
from launchkit.adapters.v0 import V0Adapter
from launchkit.assets import AssetBlobStore, project_asset_key, safe_filename
from launchkit.builds.service import safe_provider_url
from launchkit.builds.state import TERMINAL_BUILD_STATUSES, transition_build
from launchkit.core.config import Settings
from launchkit.core.exceptions import ConfigurationError, ProviderError
from launchkit.design.models import DesignPreferences
from launchkit.generation.briefing import BriefService
from launchkit.generation.models import ArchiveDownload, PipelineStatus, V0GenerationResult
from launchkit.generation.prompts import build_v0_multi_page_brief
from launchkit.images.catalogs import ImageCatalogService
from launchkit.images.registry import ImageRegistry
from launchkit.persistence.models import (
    AssetRecord,
    BuildRecord,
    JobRecord,
    MockupRecord,
    ProjectRecord,
)
from launchkit.persistence.repositories import PersistenceRepository
from launchkit.planning.models import PlannedPage, SitePlan
from launchkit.planning.text import render_plan_text
from launchkit.profiles import ExtractedImage
from launchkit.projects.catalogs import BUSINESS_CATEGORIES
from launchkit.projects.grounding import business_form_for_generation
from launchkit.projects.models import BusinessDraft, DesignDraft, PageLayout

CATEGORY_LABELS = {item.id: item.label for item in BUSINESS_CATEGORIES}


class V0BuildGateway(Protocol):
    async def create_chat(self, prompt: str) -> V0GenerationResult:
        """Submit one paid asynchronous build."""

    async def get_status(self, chat_id: str) -> V0GenerationResult:
        """Read authoritative provider state."""

    async def download_zip(self, chat_id: str) -> ArchiveDownload:
        """Download a completed provider archive."""


class BriefPreparer(Protocol):
    async def prepare(self, form: BusinessDraft, design: DesignPreferences) -> str:
        """Prepare the canonical grounded brief."""


class CatalogPreparer(Protocol):
    async def build(
        self,
        form: BusinessDraft,
        design: DesignPreferences,
        plan: SitePlan,
        uploaded_images: Sequence[ExtractedImage],
        registry: ImageRegistry | None = None,
    ) -> Mapping[str, str]:
        """Prepare page image instructions."""


class BuildJobHandlers:
    def __init__(
        self,
        settings: Settings,
        asset_store: AssetBlobStore,
        *,
        v0: V0BuildGateway | None,
        brief_service: BriefPreparer | None,
        catalogs: CatalogPreparer | None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        self._asset_store = asset_store
        self._v0 = v0
        self._brief_service = brief_service
        self._catalogs = catalogs
        self._http_client = http_client

    @property
    def handlers(self) -> dict[str, Callable[[JobRecord, AsyncSession], Awaitable[None]]]:
        return {"build.submit": self.submit, "build.reconcile": self.reconcile}

    async def submit(self, job: JobRecord, session: AsyncSession) -> None:
        build = await self._build(job, session)
        if build.status != "queued":
            return
        repository = PersistenceRepository(session)
        try:
            self._require_services()
            project = await session.get(ProjectRecord, build.project_id)
            if project is None:
                raise RuntimeError("Build project is missing")
            prompt = await self._prompt(project, session)
            build.submitted_at = datetime.now(UTC)
            await transition_build(
                repository,
                build,
                "submitting",
                stage="submitting",
                message="Submitting the website to v0",
            )
            await session.commit()

            result = await self._v0.create_chat(prompt)  # type: ignore[union-attr]
            await repository.add_provider_reference(
                resource_type="build",
                resource_id=build.id,
                provider="v0",
                reference_type="chat_id",
                reference_value=result.chat_id,
            )
            await transition_build(
                repository,
                build,
                "running" if result.status is PipelineStatus.PENDING else "processing_result",
                stage="generating"
                if result.status is PipelineStatus.PENDING
                else "processing_result",
                message="v0 is generating the website"
                if result.status is PipelineStatus.PENDING
                else "Processing the generated website",
            )
            await session.commit()
            if result.status is PipelineStatus.COMPLETED:
                demo = safe_provider_url(result.demo_url)
                if demo and not await demo_url_ready(self._http_client, demo):
                    await transition_build(
                        repository,
                        build,
                        "running",
                        stage="awaiting_preview",
                        message="Website code is ready; waiting for the live preview to respond",
                    )
                    await self._schedule(build, repository)
                else:
                    await self._complete(build, result, result.chat_id, session)
            elif result.status is PipelineStatus.FAILED:
                await transition_build(
                    repository,
                    build,
                    "failed",
                    stage="failed",
                    message="v0 could not generate the website",
                )
            else:
                await self._schedule(build, repository)
        except Exception as exc:
            if build.status not in TERMINAL_BUILD_STATUSES:
                warning = (
                    "Submission outcome may be unknown; this build will not be submitted again "
                    "automatically."
                    if isinstance(exc, ProviderError) and exc.retryable
                    else "The website could not be submitted to v0."
                )
                build.warnings = [*build.warnings, warning]
                await transition_build(repository, build, "failed", stage="failed", message=warning)
            raise

    async def reconcile(self, job: JobRecord, session: AsyncSession) -> None:
        build = await self._build(job, session)
        if build.status in TERMINAL_BUILD_STATUSES:
            return
        repository = PersistenceRepository(session)
        reference = await repository.get_provider_reference(
            resource_type="build",
            resource_id=build.id,
            provider="v0",
            reference_type="chat_id",
        )
        if reference is None:
            await transition_build(
                repository,
                build,
                "failed",
                stage="failed",
                message="The provider reference for this build is missing",
            )
            return
        if self._timed_out(build):
            await transition_build(
                repository,
                build,
                "timed_out",
                stage="timed_out",
                message="The website generation timed out",
            )
            build.completed_at = datetime.now(UTC)
            return
        if self._v0 is None:
            raise ConfigurationError("v0 is not configured")
        try:
            result = await self._v0.get_status(reference.reference_value)
            if result.status is PipelineStatus.COMPLETED:
                demo = safe_provider_url(result.demo_url)
                waiting = bool(demo) and not await demo_url_ready(self._http_client, demo)
                # Don't block forever if the preview host stays on a Next 404 shell.
                force_complete = build.reconcile_attempts >= 12
                if waiting and not force_complete:
                    await transition_build(
                        repository,
                        build,
                        "running",
                        stage="awaiting_preview",
                        message="Website code is ready; waiting for the live preview to respond",
                    )
                    await self._schedule(build, repository)
                    return
                await transition_build(
                    repository,
                    build,
                    "processing_result",
                    stage="processing_result",
                    message="Preparing the completed website",
                )
                await session.commit()
                await self._complete(build, result, reference.reference_value, session)
            elif result.status is PipelineStatus.FAILED:
                await transition_build(
                    repository,
                    build,
                    "failed",
                    stage="failed",
                    message="v0 could not generate the website",
                )
                build.completed_at = datetime.now(UTC)
            else:
                await transition_build(
                    repository,
                    build,
                    "running",
                    stage="generating",
                    message="v0 is generating the website",
                )
                await self._schedule(build, repository)
        except ProviderError:
            await self._schedule(build, repository)

    async def _complete(
        self,
        build: BuildRecord,
        result: V0GenerationResult,
        chat_id: str,
        session: AsyncSession,
    ) -> None:
        if self._v0 is None:
            raise ConfigurationError("v0 is not configured")
        repository = PersistenceRepository(session)
        archive = await self._v0.download_zip(chat_id)
        filename = safe_filename(archive.filename, "website.zip")
        project = await session.get(ProjectRecord, build.project_id)
        owner_id = project.owner_id if project is not None else "owner"
        storage_key = project_asset_key(
            owner_id,
            build.project_id,
            "builds",
            f"{uuid.uuid4().hex}-{filename}",
        )
        await self._asset_store.put(storage_key, archive.content, "application/zip")
        asset = await repository.add_asset(
            project_id=build.project_id,
            kind="build_archive",
            storage_key=storage_key,
            filename=filename,
            label="Generated website",
            content_type="application/zip",
            size=len(archive.content),
            sha256=hashlib.sha256(archive.content).hexdigest(),
        )
        if (
            result.version_id
            and await repository.get_provider_reference(
                resource_type="build",
                resource_id=build.id,
                provider="v0",
                reference_type="version_id",
            )
            is None
        ):
            await repository.add_provider_reference(
                resource_type="build",
                resource_id=build.id,
                provider="v0",
                reference_type="version_id",
                reference_value=result.version_id,
            )
        build.archive_asset_id = asset.id
        # Re-poll once so preview_url gets a fresh demo host/token after ZIP download.
        try:
            refreshed = await self._v0.get_status(chat_id)
            demo = safe_provider_url(refreshed.demo_url) or safe_provider_url(result.demo_url)
            web = safe_provider_url(refreshed.web_url) or safe_provider_url(result.web_url)
        except ProviderError:
            demo = safe_provider_url(result.demo_url)
            web = safe_provider_url(result.web_url)
        build.preview_url = demo
        build.web_url = web
        build.file_manifest = [{"name": name} for name in result.files]
        build.next_reconcile_at = None
        build.completed_at = datetime.now(UTC)
        project = await session.get(ProjectRecord, build.project_id)
        if project is not None:
            project.status = "build_completed"
        await transition_build(
            repository,
            build,
            "completed",
            stage="completed",
            message="Website generation completed",
        )

    async def _prompt(self, project: ProjectRecord, session: AsyncSession) -> str:
        if self._brief_service is None or self._catalogs is None:
            raise ConfigurationError("Build prompt services are not configured")
        form = business_form_for_generation(
            project.business,
            project.extracted_profile_fields,
            category_fallback_industry=CATEGORY_LABELS.get(
                str(project.business.get("categoryId") or "tech-saas"),
                str(project.business.get("categoryId") or "tech-saas"),
            ),
        )
        design = DesignDraft.model_validate(project.design).to_preferences()
        plan = page_layout_to_plan(PageLayout.model_validate(project.page_layout))
        mockup = await session.get(MockupRecord, project.selected_mockup_id)
        if mockup is None:
            raise RuntimeError("Selected mockup is missing")
        artifact = await session.get(AssetRecord, mockup.artifact_asset_id)
        if artifact is None:
            raise RuntimeError("Selected mockup artifact is missing")
        mockup_html = (await self._asset_store.get(artifact.storage_key)).decode("utf-8")
        repository = PersistenceRepository(session)
        uploaded = await uploaded_images(repository, project.id, self._asset_store)
        brief = await self._brief_service.prepare(form, design)
        catalogs = await self._catalogs.build(form, design, plan, uploaded)
        return build_v0_multi_page_brief(brief, mockup_html, plan.pages, catalogs)

    async def _schedule(self, build: BuildRecord, repository: PersistenceRepository) -> None:
        build.reconcile_attempts += 1
        delay = min(
            self._settings.build_reconcile_initial_seconds
            * 2 ** max(0, build.reconcile_attempts - 1),
            self._settings.build_reconcile_max_seconds,
        )
        available_at = datetime.now(UTC) + timedelta(seconds=delay)
        build.next_reconcile_at = available_at
        await repository.enqueue_job(
            "build.reconcile", {"buildId": build.id}, available_at=available_at
        )

    async def _build(self, job: JobRecord, session: AsyncSession) -> BuildRecord:
        build = await session.get(BuildRecord, str(job.payload.get("buildId", "")))
        if build is None:
            raise RuntimeError("Build job has no build")
        return build

    def _require_services(self) -> None:
        if self._v0 is None or self._brief_service is None or self._catalogs is None:
            raise ConfigurationError("Final build providers are not configured")

    def _timed_out(self, build: BuildRecord) -> bool:
        started = build.submitted_at or build.created_at
        if started.tzinfo is None:
            started = started.replace(tzinfo=UTC)
        return datetime.now(UTC) - started >= timedelta(
            seconds=self._settings.build_timeout_seconds
        )


def create_build_job_handlers(
    settings: Settings, client: httpx.AsyncClient, asset_store: AssetBlobStore
) -> BuildJobHandlers:
    v0_key = settings.v0_api_key.get_secret_value() if settings.v0_api_key else ""
    openrouter_key = (
        settings.openrouter_api_key.get_secret_value() if settings.openrouter_api_key else ""
    )
    if not v0_key or not openrouter_key:
        return BuildJobHandlers(
            settings, asset_store, v0=None, brief_service=None, catalogs=None, http_client=client
        )
    queue = RequestQueue(
        max_concurrent=settings.openrouter_max_concurrent,
        min_gap_seconds=settings.openrouter_min_request_gap_ms / 1000,
    )
    openrouter = OpenRouterAdapter(
        client,
        queue,
        api_key=openrouter_key,
        base_url=settings.openrouter_base_url,
        generation_model=settings.generation_model,
        utility_model=settings.utility_model,
        image_model=settings.image_model,
        site_url=settings.site_url,
        app_title=settings.openrouter_app_title,
        max_attempts=settings.openrouter_retry_attempts,
    )
    pexels_key = settings.pexels_api_key.get_secret_value() if settings.pexels_api_key else None
    return BuildJobHandlers(
        settings,
        asset_store,
        v0=V0Adapter(
            client,
            api_key=v0_key,
            base_url=settings.v0_base_url,
            model_id=settings.v0_model,
        ),
        brief_service=BriefService(openrouter),
        catalogs=ImageCatalogService(
            image_generator=openrouter,
            image_search=PexelsAdapter(
                client, api_key=pexels_key, base_url=settings.pexels_base_url
            ),
        ),
        http_client=client,
    )


def page_layout_to_plan(layout: PageLayout) -> SitePlan:
    has_home = any(page.template_id == "home" for page in layout.pages)
    pages = [
        PlannedPage(
            name=page.name,
            slug=page.slug,
            is_home=page.template_id == "home" if has_home else index == 0,
            purpose=f"The {page.name} page for this website.",
            sections=[section.name for section in page.sections if not section.locked],
            images=[],
        )
        for index, page in enumerate(layout.pages)
    ]
    return SitePlan(pages=pages, raw=render_plan_text(pages))


async def demo_url_ready(client: httpx.AsyncClient | None, demo_url: str) -> bool:
    """Return True when the v0 demo host looks like a live site.

    Rejects connection failures, non-200 responses, and the common Next.js
    ``404 | This page could not be found.`` shell that still returns HTTP 200.
    """

    if client is None:
        return True
    try:
        response = await client.get(
            demo_url,
            timeout=httpx.Timeout(5.0, read=8.0),
            headers={"User-Agent": "launchkit-readiness-probe"},
            follow_redirects=True,
        )
        if response.status_code != 200:
            return False
        body = response.text[:4_000].lower()
        if "this page could not be found" in body or "404 |" in body:
            return False
        return True
    except httpx.HTTPError:
        return False


async def uploaded_images(
    repository: PersistenceRepository, project_id: str, store: AssetBlobStore
) -> list[ExtractedImage]:
    images: list[ExtractedImage] = []
    for asset in await repository.list_assets(project_id):
        if asset.kind != "profile_image":
            continue
        content = await store.get(asset.storage_key)
        encoded = base64.b64encode(content).decode("ascii")
        images.append(
            ExtractedImage(
                filename=asset.filename,
                label=asset.label,
                data_url=f"data:{asset.content_type};base64,{encoded}",
            )
        )
    return images
