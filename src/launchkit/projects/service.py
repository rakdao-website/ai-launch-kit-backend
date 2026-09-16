"""Project application service, independent of HTTP transport."""

from typing import Any

import structlog
from pydantic import BaseModel

from launchkit.builds.service import GenerationQuotaExceededError, public_preview_url
from launchkit.core.config import Settings
from launchkit.core.exceptions import DomainError
from launchkit.persistence.models import BuildRecord, ProjectRecord
from launchkit.persistence.repositories import PersistenceRepository
from launchkit.projects.models import ProjectDraft, ProjectPatch, ProjectSummaryView, ProjectView
from launchkit.workflows.service import assets_view, mockups_view

logger = structlog.get_logger(__name__)


class ProjectNotFoundError(DomainError):
    """Raised when a project is missing or belongs to another user."""


class ProjectService:
    def __init__(
        self,
        repository: PersistenceRepository,
        owner_id: str,
        settings: Settings | None = None,
    ) -> None:
        self._repository = repository
        self._owner_id = owner_id
        self._settings = settings or Settings()

    async def create(self, draft: ProjectDraft) -> ProjectView:
        unlimited = self._settings.is_generation_quota_disabled
        logger.info(
            "generation_quota_checked",
            owner_id=self._owner_id,
            unlimited=unlimited,
            quota_disabled=unlimited,
        )

        # One website per user: soft-reset an unfinished draft, or block after a generation.
        # When the quota is disabled, additional projects may be created for retesting.
        if not unlimited:
            if await self._repository.count_owner_website_builds(self._owner_id) >= 1:
                raise GenerationQuotaExceededError()
            existing = list(await self._repository.list_projects(self._owner_id))
            if existing:
                record = existing[0]
                # "Create new website" reuses the single allowed project but starts clean.
                record.business = draft.business.model_dump(by_alias=True)
                record.design = draft.design.model_dump(by_alias=True)
                record.page_layout = draft.page_layout.model_dump(by_alias=True)
                record.extracted_profile_fields = {}
                record.selected_mockup_id = None
                record.status = "draft"
                for asset in list(await self._repository.list_assets(record.id)):
                    if asset.kind in {"profile_source", "profile_image"}:
                        await self._repository.delete_asset(asset)
                await self._repository.commit()
                await self._repository.refresh(record)
                return await self._view(record)

        record = await self._repository.add_project(
            owner_id=self._owner_id,
            business=draft.business.model_dump(by_alias=True),
            design=draft.design.model_dump(by_alias=True),
            page_layout=draft.page_layout.model_dump(by_alias=True),
        )
        await self._repository.commit()
        return await self._view(record)

    async def list(self) -> list[ProjectSummaryView]:
        records = list(await self._repository.list_projects(self._owner_id))
        build_ids = [record.latest_build_id for record in records if record.latest_build_id]
        builds = {
            build.id: build
            for build in await self._repository.get_builds_by_ids(build_ids, self._owner_id)
        }
        return [
            self._summary(record, builds.get(record.latest_build_id or ""))
            for record in records
        ]

    async def get(self, project_id: str) -> ProjectView:
        return await self._view(await self._record(project_id))

    async def patch(self, project_id: str, patch: ProjectPatch) -> ProjectView:
        record = await self._record(project_id)
        business = self._merge(record.business, patch.business)
        design = self._merge(record.design, patch.design)
        page_layout = (
            patch.page_layout.model_dump(by_alias=True)
            if patch.page_layout is not None
            else record.page_layout
        )
        validated = ProjectDraft.model_validate(
            {"business": business, "design": design, "pageLayout": page_layout}
        )
        record.business = validated.business.model_dump(by_alias=True)
        record.design = validated.design.model_dump(by_alias=True)
        record.page_layout = validated.page_layout.model_dump(by_alias=True)
        await self._repository.commit()
        await self._repository.refresh(record)
        return await self._view(record)

    async def _record(self, project_id: str) -> ProjectRecord:
        record = await self._repository.get_project(project_id, self._owner_id)
        if record is None:
            raise ProjectNotFoundError("Project not found")
        return record

    @staticmethod
    def _merge(current: dict[str, Any], patch: BaseModel | None) -> dict[str, Any]:
        if patch is None:
            return current
        values = patch.model_dump(by_alias=True, exclude_unset=True)
        return {**current, **values}

    async def _view(self, record: ProjectRecord) -> ProjectView:
        assets = [
            asset
            for asset in await self._repository.list_assets(record.id)
            if asset.kind in {"profile_source", "profile_image"}
        ]
        mockups = await self._repository.list_mockups(record.id)
        return ProjectView.model_validate(
            {
                "id": record.id,
                "status": record.status,
                "business": record.business,
                "design": record.design,
                "pageLayout": record.page_layout,
                "extractedProfileFields": record.extracted_profile_fields,
                "uploadedAssets": assets_view(assets),
                "mockups": mockups_view(mockups),
                "selectedMockupId": record.selected_mockup_id,
                "latestBuildId": record.latest_build_id,
                "latestDeploymentId": record.latest_deployment_id,
                "createdAt": record.created_at,
                "updatedAt": record.updated_at,
            }
        )

    @staticmethod
    def _summary(record: ProjectRecord, build: BuildRecord | None) -> ProjectSummaryView:
        company_name = str(record.business.get("companyName") or "").strip()
        download_url = (
            f"/api/v1/builds/{build.id}/download"
            if build is not None and build.status == "completed"
            else None
        )
        return ProjectSummaryView(
            id=record.id,
            status=record.status,
            company_name=company_name,
            latest_build_id=record.latest_build_id,
            latest_build_status=build.status if build is not None else None,
            preview_url=(
                public_preview_url(build.preview_url) if build is not None else None
            ),
            download_url=download_url,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )
