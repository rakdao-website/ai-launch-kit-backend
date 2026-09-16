"""Build service readiness, idempotency, ownership, and download behavior."""

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast

import pytest

from launchkit.assets import AssetBlobStore
from launchkit.builds.models import BuildCreate, BuildView
from launchkit.builds.service import (
    BuildNotFoundError,
    BuildService,
    GenerationQuotaExceededError,
)
from launchkit.core.config import Settings
from launchkit.core.exceptions import ConfigurationError, DomainError
from launchkit.persistence.models import AssetRecord, BuildRecord, ProjectRecord
from launchkit.persistence.repositories import PersistenceRepository


class RepositoryStub:
    def __init__(self) -> None:
        now = datetime.now(UTC)
        self.project: ProjectRecord | None = ProjectRecord(
            id="project-1",
            owner_id="owner-1",
            status="draft",
            business={"companyName": "Northstar"},
            design={},
            page_layout={},
            selected_mockup_id="mockup-1",
            created_at=now,
            updated_at=now,
        )
        self.build: BuildRecord | None = None
        self.mockup_exists = True
        self.active = False
        self.owner_build_count = 0
        self.idempotency: Any | None = None
        self.asset: AssetRecord | None = None
        self.provider_ref: Any | None = None
        self.user: Any | None = None
        self.events: list[dict[str, Any]] = []
        self.jobs: list[tuple[str, dict[str, str]]] = []
        self.commits = 0

    async def get_project(self, project_id: str, owner_id: str) -> ProjectRecord | None:
        del project_id, owner_id
        return self.project

    async def get_user(self, user_id: str) -> Any | None:
        del user_id
        return self.user

    async def get_mockup(self, mockup_id: str, project_id: str) -> object | None:
        del mockup_id, project_id
        return object() if self.mockup_exists else None

    async def get_idempotency(self, **kwargs: str) -> Any | None:
        del kwargs
        return self.idempotency

    async def find_active_build(self, project_id: str) -> BuildRecord | None:
        del project_id
        return self.build if self.active else None

    async def count_owner_website_builds(self, owner_id: str) -> int:
        del owner_id
        return self.owner_build_count

    async def add_build(self, *, project_id: str, provider: str) -> BuildRecord:
        now = datetime.now(UTC)
        self.build = BuildRecord(
            id="build-1",
            project_id=project_id,
            provider=provider,
            status="queued",
            stage="queued",
            message="Build queued",
            warnings=[],
            created_at=now,
            updated_at=now,
        )
        return self.build

    async def add_status_event(self, **event: Any) -> None:
        self.events.append(event)

    async def enqueue_job(self, kind: str, payload: dict[str, str]) -> None:
        self.jobs.append((kind, payload))

    async def add_idempotency(self, **values: str) -> None:
        self.idempotency = SimpleNamespace(
            request_hash=values["request_hash"], resource_id=values["resource_id"]
        )

    async def commit(self) -> None:
        self.commits += 1

    async def get_build(self, build_id: str, owner_id: str) -> BuildRecord | None:
        del build_id, owner_id
        return self.build

    async def get_asset(self, asset_id: str, owner_id: str) -> AssetRecord | None:
        del asset_id, owner_id
        return self.asset

    async def get_provider_reference(self, **kwargs: str) -> Any | None:
        del kwargs
        return self.provider_ref


class BlobStoreStub:
    async def get(self, storage_key: str) -> bytes:
        assert storage_key == "archives/site.zip"
        return b"PK-archive"


def service(
    repository: RepositoryStub,
    *,
    configured: bool = True,
    v0_status: Any | None = None,
) -> BuildService:
    return BuildService(
        cast(PersistenceRepository, repository),
        "owner-1",
        Settings(
            environment="staging",
            v0_api_key="v0" if configured else None,
            # Pinned: load_dotenv_file leaks a developer's .env into os.environ.
            disable_generation_quota=False,
            _env_file=None,
        ),
        cast(AssetBlobStore, BlobStoreStub()),
        v0_status=v0_status,
    )


def start(repository: RepositoryStub, key: str = "build-key") -> BuildView:
    return asyncio.run(service(repository).start("project-1", BuildCreate(), key))


def test_start_enforces_project_configuration_and_readiness() -> None:
    repository = RepositoryStub()
    repository.project = None
    with pytest.raises(BuildNotFoundError, match="Project not found"):
        start(repository)

    repository = RepositoryStub()
    with pytest.raises(ConfigurationError, match="v0"):
        asyncio.run(service(repository, configured=False).start("project-1", BuildCreate(), "key"))

    assert repository.project is not None
    repository.project.business["companyName"] = " "
    with pytest.raises(DomainError, match="company name"):
        start(repository)

    repository = RepositoryStub()
    assert repository.project is not None
    repository.project.selected_mockup_id = None
    with pytest.raises(DomainError, match="Select a mockup"):
        start(repository)

    repository = RepositoryStub()
    repository.mockup_exists = False
    with pytest.raises(DomainError, match="no longer available"):
        start(repository)

    with pytest.raises(DomainError, match="Idempotency-Key"):
        start(RepositoryStub(), " ")


def test_start_is_idempotent_and_prevents_conflicts() -> None:
    repository = RepositoryStub()
    created = start(repository)
    repeated = start(repository)

    assert created.id == repeated.id == "build-1"
    assert repository.commits == 1
    assert repository.jobs == [("build.submit", {"buildId": "build-1"})]
    assert repository.project is not None
    assert repository.project.latest_build_id == "build-1"
    assert repository.project.status == "build_queued"

    assert repository.idempotency is not None
    repository.idempotency.request_hash = "different"
    with pytest.raises(DomainError, match="different build"):
        start(repository)

    repository.idempotency = None
    repository.active = True
    with pytest.raises(DomainError, match="already active"):
        start(repository, "new-key")


def test_start_enforces_one_website_per_user() -> None:
    repository = RepositoryStub()
    repository.owner_build_count = 1
    with pytest.raises(GenerationQuotaExceededError, match="need more credits"):
        start(repository)

    # Idempotent replays of the already-created build still succeed.
    repository = RepositoryStub()
    created = start(repository)
    repository.owner_build_count = 1
    repeated = start(repository)
    assert created.id == repeated.id


def test_start_allows_extra_builds_when_quota_disabled() -> None:
    repository = RepositoryStub()
    repository.owner_build_count = 1
    created = asyncio.run(
        BuildService(
            cast(PersistenceRepository, repository),
            "owner-1",
            Settings(
                environment="staging",
                v0_api_key="v0",
                disable_generation_quota=True,
                _env_file=None,
            ),
            cast(AssetBlobStore, BlobStoreStub()),
        ).start("project-1", BuildCreate(), "extra-key")
    )
    assert created.id == "build-1"


def test_start_rejects_a_dangling_idempotency_resource() -> None:
    repository = RepositoryStub()
    start(repository)
    repository.build = None

    with pytest.raises(BuildNotFoundError, match="Build not found"):
        start(repository)


def test_get_and_download_enforce_ownership_and_archive_readiness() -> None:
    repository = RepositoryStub()
    build_service = service(repository)
    with pytest.raises(BuildNotFoundError, match="Build not found"):
        asyncio.run(build_service.get("missing"))
    with pytest.raises(BuildNotFoundError, match="Build not found"):
        asyncio.run(build_service.download("missing"))

    start(repository)
    assert asyncio.run(build_service.get("build-1")).id == "build-1"
    with pytest.raises(DomainError, match="not ready"):
        asyncio.run(build_service.download("build-1"))

    assert repository.build is not None
    repository.build.status = "completed"
    repository.build.archive_asset_id = "asset-1"
    with pytest.raises(BuildNotFoundError, match="archive not found"):
        asyncio.run(build_service.download("build-1"))

    now = datetime.now(UTC)
    repository.asset = AssetRecord(
        id="asset-1",
        project_id="project-1",
        kind="build_archive",
        storage_key="archives/site.zip",
        filename="Northstar site.zip",
        label="Build archive",
        content_type="application/zip",
        size=10,
        sha256="a" * 64,
        created_at=now,
        updated_at=now,
    )
    content, filename = asyncio.run(build_service.download("build-1"))
    assert content == b"PK-archive"
    assert filename == "Northstar site.zip"


def test_preview_refreshes_stale_demo_url_from_v0() -> None:
    from launchkit.generation.models import PipelineStatus, V0GenerationResult

    repository = RepositoryStub()
    start(repository)
    assert repository.build is not None
    repository.build.status = "completed"
    repository.build.preview_url = "https://demo-old.vusercontent.net/"
    repository.provider_ref = SimpleNamespace(reference_value="chat-private")

    async def v0_status(chat_id: str) -> V0GenerationResult:
        assert chat_id == "chat-private"
        return V0GenerationResult(
            chat_id=chat_id,
            web_url="https://v0.app/chat/chat-private",
            demo_url="https://demo-fresh.vusercontent.net/?__v0_token=abc",
            status=PipelineStatus.COMPLETED,
            file_count=1,
        )

    preview = asyncio.run(service(repository, v0_status=v0_status).preview("build-1"))
    assert preview.url == "https://demo-fresh.vusercontent.net/"
    assert repository.build.preview_url == preview.url
    assert repository.commits == 2  # start + preview refresh


def test_preview_falls_back_to_stored_url_without_chat_reference() -> None:
    repository = RepositoryStub()
    start(repository)
    assert repository.build is not None
    repository.build.status = "completed"
    repository.build.preview_url = "https://demo-stored.vusercontent.net/?__v0_token=old"
    repository.provider_ref = None

    preview = asyncio.run(service(repository).preview("build-1"))
    assert preview.url == "https://demo-stored.vusercontent.net/"


def test_public_preview_url_strips_v0_token_only_for_demo_hosts() -> None:
    from launchkit.builds.service import public_preview_url

    assert (
        public_preview_url("https://demo-abc.vusercontent.net/?__v0_token=secret&x=1")
        == "https://demo-abc.vusercontent.net/"
    )
    assert public_preview_url("https://northstar.vercel.app/site") == "https://northstar.vercel.app/site"
    assert public_preview_url(None) is None
    assert public_preview_url("not-a-url") is None
