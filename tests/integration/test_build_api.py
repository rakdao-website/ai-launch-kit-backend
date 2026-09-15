"""End-to-end asynchronous build lifecycle with an offline v0 gateway."""

import asyncio
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from fastapi.testclient import TestClient
from sqlalchemy import select

from launchkit.assets.storage import LocalAssetBlobStore
from launchkit.builds.handlers import BuildJobHandlers
from launchkit.core.config import Settings
from launchkit.core.exceptions import ProviderError
from launchkit.design.models import DesignPreferences
from launchkit.generation.models import ArchiveDownload, PipelineStatus, V0GenerationResult
from launchkit.main import create_app
from launchkit.persistence import PersistenceRepository, create_database
from launchkit.persistence.base import Base
from launchkit.persistence.models import BuildRecord, JobRecord
from launchkit.planning.models import SitePlan
from launchkit.profiles import ExtractedImage
from launchkit.projects.models import BusinessDraft
from launchkit.worker import Worker


class BriefStub:
    async def prepare(self, form: BusinessDraft, design: DesignPreferences) -> str:
        del design
        return f"Brief for {form.company_name}"


class CatalogStub:
    async def build(
        self,
        form: BusinessDraft,
        design: DesignPreferences,
        plan: SitePlan,
        uploaded_images: Sequence[ExtractedImage],
        registry: object | None = None,
    ) -> Mapping[str, str]:
        del form, design, uploaded_images, registry
        return {page.name: "No images" for page in plan.pages}


class V0Stub:
    def __init__(
        self,
        statuses: list[PipelineStatus] | None = None,
        *,
        submit_error: ProviderError | None = None,
    ) -> None:
        self.statuses = statuses or []
        self.submit_error = submit_error
        self.submissions = 0
        self.status_calls = 0
        self.downloads = 0

    async def create_chat(self, prompt: str) -> V0GenerationResult:
        assert "PAGES TO BUILD" in prompt
        self.submissions += 1
        if self.submit_error:
            raise self.submit_error
        return result(PipelineStatus.PENDING)

    async def get_status(self, chat_id: str) -> V0GenerationResult:
        assert chat_id == "chat-private"
        self.status_calls += 1
        return result(self.statuses.pop(0) if self.statuses else PipelineStatus.PENDING)

    async def download_zip(self, chat_id: str) -> ArchiveDownload:
        assert chat_id == "chat-private"
        self.downloads += 1
        return ArchiveDownload(content=b"PK-test-archive", filename="Northstar website.zip")


def result(status: PipelineStatus) -> V0GenerationResult:
    return V0GenerationResult(
        chat_id="chat-private",
        web_url="https://v0.dev/chat/private",
        demo_url="https://demo.v0.dev/site",
        status=status,
        file_count=2,
        version_id="ver-private" if status is PipelineStatus.COMPLETED else None,
        files=["app/page.tsx", "package.json"] if status is PipelineStatus.COMPLETED else [],
    )


@contextmanager
def build_client(
    tmp_path: Path, *, configured: bool = True
) -> Iterator[tuple[TestClient, Settings, LocalAssetBlobStore]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    settings = Settings(
        environment="test",
        auth_mode="testing",
        database_url=f"sqlite+aiosqlite:///{(tmp_path / 'build.sqlite3').as_posix()}",
        local_data_dir=tmp_path / "data",
        openrouter_api_key="openrouter" if configured else None,
        v0_api_key="v0" if configured else None,
        v0_webhook_token="hook-secret",
        build_reconcile_initial_seconds=1,
        build_reconcile_max_seconds=2,
        build_timeout_seconds=60,
        sse_poll_seconds=0.01,
        sse_heartbeat_seconds=0.05,
    )
    database = create_database(settings)
    store = LocalAssetBlobStore(settings.local_data_dir / "assets")

    async def create_schema() -> None:
        async with database.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    asyncio.run(create_schema())
    with TestClient(create_app(settings, database, store)) as client:
        yield client, settings, store


def create_project(client: TestClient) -> dict[str, Any]:
    response = client.post("/api/v1/projects", json={"business": {"companyName": "Northstar"}})
    assert response.status_code == 201
    return cast(dict[str, Any], response.json())


def prepare_mockup(settings: Settings, store: LocalAssetBlobStore, project_id: str) -> str:
    async def scenario() -> str:
        database = create_database(settings)
        key = f"projects/{project_id}/mockups/chosen.html"
        content = b"<!DOCTYPE html><html><h1>Chosen direction</h1></html>"
        await store.put(key, content, "text/html")
        async with database.session() as session:
            repository = PersistenceRepository(session)
            asset = await repository.add_asset(
                project_id=project_id,
                kind="mockup_html",
                storage_key=key,
                filename="chosen.html",
                label="Chosen",
                content_type="text/html",
                size=len(content),
                sha256="a" * 64,
            )
            mockup = await repository.add_mockup(
                project_id=project_id,
                generation=1,
                ordinal=1,
                label="Chosen",
                direction="Editorial",
                artifact_asset_id=asset.id,
            )
            project = await repository.get_project(project_id, settings.testing_user_id)
            assert project is not None
            project.selected_mockup_id = mockup.id
            await repository.commit()
        await database.close()
        return mockup.id

    return asyncio.run(scenario())


def run_worker(settings: Settings, handlers: BuildJobHandlers) -> int:
    async def scenario() -> int:
        worker = Worker(settings, handlers=handlers.handlers)
        count = await worker.run_once()
        await worker.close()
        return count

    return asyncio.run(scenario())


def release_reconciliation(settings: Settings, build_id: str, *, timed_out: bool = False) -> None:
    async def scenario() -> None:
        database = create_database(settings)
        async with database.session() as session:
            jobs = await session.scalars(
                select(JobRecord).where(
                    JobRecord.kind == "build.reconcile", JobRecord.status == "queued"
                )
            )
            for job in jobs:
                job.available_at = datetime.now(UTC) - timedelta(seconds=1)
            if timed_out:
                build = await session.get(BuildRecord, build_id)
                assert build is not None
                build.submitted_at = datetime.now(UTC) - timedelta(seconds=61)
            await session.commit()
        await database.close()

    asyncio.run(scenario())


def test_webhook_completes_build_and_enables_sse_and_download(tmp_path: Path) -> None:
    with build_client(tmp_path) as (client, settings, store):
        project = create_project(client)
        prepare_mockup(settings, store, project["id"])
        headers = {"Idempotency-Key": "final-build-1"}
        queued = client.post(f"/api/v1/projects/{project['id']}/builds", json={}, headers=headers)
        duplicate = client.post(
            f"/api/v1/projects/{project['id']}/builds", json={}, headers=headers
        )
        before_download = client.get(f"/api/v1/builds/{queued.json()['id']}/download")
        v0 = V0Stub([PipelineStatus.COMPLETED])
        handlers = BuildJobHandlers(
            settings,
            store,
            v0=v0,
            brief_service=BriefStub(),
            catalogs=CatalogStub(),
        )

        assert queued.status_code == 202
        assert duplicate.json()["id"] == queued.json()["id"]
        assert before_download.status_code == 409
        assert run_worker(settings, handlers) == 1
        running = client.get(f"/api/v1/builds/{queued.json()['id']}")
        assert running.json()["status"] == "running"
        assert "chat-private" not in running.text

        payload = {
            "id": "delivery-1",
            "type": "message.finished",
            "data": {"chatId": "chat-private"},
        }
        hook = client.post("/api/v1/webhooks/v0/hook-secret", json=payload)
        hook_duplicate = client.post("/api/v1/webhooks/v0/hook-secret", json=payload)
        assert hook.json()["correlated"] is True
        assert hook_duplicate.json()["duplicate"] is True
        assert run_worker(settings, handlers) == 1

        completed = client.get(f"/api/v1/builds/{queued.json()['id']}")
        download = client.get(completed.json()["downloadUrl"])
        events = client.get(f"/api/v1/builds/{queued.json()['id']}/events")
        replay = client.get(
            f"/api/v1/builds/{queued.json()['id']}/events",
            headers={"Last-Event-ID": "3"},
        )
        terminal_repeat = client.post(
            "/api/v1/webhooks/v0/hook-secret",
            json={**payload, "id": "delivery-2"},
        )

    assert completed.json()["status"] == "completed"
    assert completed.json()["previewUrl"] == "https://demo.v0.dev/site"
    assert download.content == b"PK-test-archive"
    assert "attachment;" in download.headers["content-disposition"]
    assert "event: status" in events.text
    assert "id: 1" in events.text and "id: 5" in events.text
    assert "id: 1" not in replay.text and "id: 4" in replay.text
    assert terminal_repeat.json()["correlated"] is True
    assert v0.submissions == v0.downloads == 1


def test_reconciliation_completes_when_webhook_is_missing(tmp_path: Path) -> None:
    with build_client(tmp_path) as (client, settings, store):
        project = create_project(client)
        prepare_mockup(settings, store, project["id"])
        queued = client.post(
            f"/api/v1/projects/{project['id']}/builds",
            json={},
            headers={"Idempotency-Key": "missing-hook"},
        )
        v0 = V0Stub([PipelineStatus.COMPLETED])
        handlers = BuildJobHandlers(
            settings, store, v0=v0, brief_service=BriefStub(), catalogs=CatalogStub()
        )
        assert run_worker(settings, handlers) == 1
        release_reconciliation(settings, queued.json()["id"])
        assert run_worker(settings, handlers) == 1
        completed = client.get(f"/api/v1/builds/{queued.json()['id']}")

    assert completed.json()["status"] == "completed"
    # reconcile get_status + post-download refresh inside _complete
    assert v0.status_calls == 2


def test_reconciliation_times_out_without_querying_provider(tmp_path: Path) -> None:
    with build_client(tmp_path) as (client, settings, store):
        project = create_project(client)
        prepare_mockup(settings, store, project["id"])
        queued = client.post(
            f"/api/v1/projects/{project['id']}/builds",
            json={},
            headers={"Idempotency-Key": "timeout"},
        )
        v0 = V0Stub()
        handlers = BuildJobHandlers(
            settings, store, v0=v0, brief_service=BriefStub(), catalogs=CatalogStub()
        )
        assert run_worker(settings, handlers) == 1
        release_reconciliation(settings, queued.json()["id"], timed_out=True)
        assert run_worker(settings, handlers) == 1
        timed_out = client.get(f"/api/v1/builds/{queued.json()['id']}")

    assert timed_out.json()["status"] == "timed_out"
    assert v0.status_calls == 0


def test_uncertain_submission_is_not_automatically_retried(tmp_path: Path) -> None:
    with build_client(tmp_path) as (client, settings, store):
        project = create_project(client)
        prepare_mockup(settings, store, project["id"])
        queued = client.post(
            f"/api/v1/projects/{project['id']}/builds",
            json={},
            headers={"Idempotency-Key": "uncertain"},
        )
        v0 = V0Stub(submit_error=ProviderError("network", retryable=True))
        handlers = BuildJobHandlers(
            settings, store, v0=v0, brief_service=BriefStub(), catalogs=CatalogStub()
        )
        assert run_worker(settings, handlers) == 1
        assert run_worker(settings, handlers) == 0
        failed = client.get(f"/api/v1/builds/{queued.json()['id']}")

    assert failed.json()["status"] == "failed"
    assert "will not be submitted again" in failed.json()["message"]
    assert v0.submissions == 1


def test_webhook_reconciliation_persists_authoritative_failure(tmp_path: Path) -> None:
    with build_client(tmp_path) as (client, settings, store):
        project = create_project(client)
        prepare_mockup(settings, store, project["id"])
        queued = client.post(
            f"/api/v1/projects/{project['id']}/builds",
            json={},
            headers={"Idempotency-Key": "provider-failure"},
        )
        v0 = V0Stub([PipelineStatus.FAILED])
        handlers = BuildJobHandlers(
            settings, store, v0=v0, brief_service=BriefStub(), catalogs=CatalogStub()
        )
        assert run_worker(settings, handlers) == 1
        hook = client.post(
            "/api/v1/webhooks/v0/hook-secret",
            json={
                "id": "failure-event",
                "type": "message.finished",
                "data": {"message": {"chatId": "chat-private"}},
            },
        )
        assert hook.status_code == 202
        assert run_worker(settings, handlers) == 1
        failed = client.get(f"/api/v1/builds/{queued.json()['id']}")

    assert failed.json()["status"] == "failed"
    assert v0.downloads == 0


def test_build_readiness_configuration_and_unknown_webhooks(tmp_path: Path) -> None:
    with build_client(tmp_path) as (client, settings, store):
        project = create_project(client)
        not_ready = client.post(
            f"/api/v1/projects/{project['id']}/builds",
            json={},
            headers={"Idempotency-Key": "not-ready"},
        )
        prepare_mockup(settings, store, project["id"])
        unknown = client.post(
            "/api/v1/webhooks/v0/hook-secret",
            json={
                "id": "unknown",
                "type": "message.finished",
                "data": {"chatId": "chat-unknown"},
            },
        )
        invalid_token = client.post(
            "/api/v1/webhooks/v0/wrong",
            json={"type": "message.finished", "data": {"chatId": "chat-unknown"}},
        )

    assert not_ready.status_code == 409
    assert unknown.json() == {"status": "ignored", "duplicate": False, "correlated": False}
    assert invalid_token.status_code == 404

    with build_client(tmp_path / "unconfigured", configured=False) as (client, settings, store):
        project = create_project(client)
        prepare_mockup(settings, store, project["id"])
        response = client.post(
            f"/api/v1/projects/{project['id']}/builds",
            json={},
            headers={"Idempotency-Key": "no-provider"},
        )
    assert response.status_code == 503


def test_build_creation_guards_and_owned_reads(tmp_path: Path) -> None:
    with build_client(tmp_path) as (client, settings, store):
        missing_project = client.post(
            "/api/v1/projects/missing/builds",
            json={},
            headers={"Idempotency-Key": "missing-project"},
        )
        missing_build = client.get("/api/v1/builds/missing")
        missing_download = client.get("/api/v1/builds/missing/download")

        project = create_project(client)
        prepare_mockup(settings, store, project["id"])
        invalid_key = client.post(
            f"/api/v1/projects/{project['id']}/builds",
            json={},
            headers={"Idempotency-Key": "x" * 201},
        )
        first = client.post(
            f"/api/v1/projects/{project['id']}/builds",
            json={},
            headers={"Idempotency-Key": "guarded-build"},
        )
        active = client.post(
            f"/api/v1/projects/{project['id']}/builds",
            json={},
            headers={"Idempotency-Key": "another-build"},
        )
        changed = client.patch(
            f"/api/v1/projects/{project['id']}",
            json={"business": {"companyName": "Changed"}},
        )
        conflict = client.post(
            f"/api/v1/projects/{project['id']}/builds",
            json={},
            headers={"Idempotency-Key": "guarded-build"},
        )

    assert missing_project.status_code == 404
    assert missing_build.status_code == 404
    assert missing_download.status_code == 404
    assert invalid_key.status_code == 409
    assert first.status_code == 202
    assert active.status_code == 409
    assert changed.status_code == 200
    assert conflict.status_code == 409
    assert "different build" in conflict.json()["error"]["message"]


def test_webhook_rejects_bad_payloads_and_ignores_other_events(tmp_path: Path) -> None:
    with build_client(tmp_path) as (client, settings, store):
        del settings, store
        malformed = client.post(
            "/api/v1/webhooks/v0/hook-secret",
            content=b"{",
            headers={"Content-Type": "application/json"},
        )
        non_object = client.post("/api/v1/webhooks/v0/hook-secret", json=["event"])
        missing_chat = client.post(
            "/api/v1/webhooks/v0/hook-secret",
            json={"id": "missing-chat", "type": "message.finished"},
        )
        ignored = client.post(
            "/api/v1/webhooks/v0/hook-secret",
            json={"type": "message.created", "data": {"chatId": "chat-private"}},
        )
        ignored_duplicate = client.post(
            "/api/v1/webhooks/v0/hook-secret",
            json={"type": "message.created", "data": {"chatId": "chat-private"}},
        )

    assert malformed.status_code == 400
    assert non_object.status_code == 400
    assert missing_chat.status_code == 400
    assert ignored.json()["status"] == "ignored"
    assert ignored_duplicate.json()["duplicate"] is True


def test_webhook_configuration_and_payload_size_are_enforced(tmp_path: Path) -> None:
    with build_client(tmp_path / "large") as (client, settings, store):
        del store
        settings.webhook_max_bytes = 1024
        oversized = client.post(
            "/api/v1/webhooks/v0/hook-secret",
            content=b"x" * 1025,
            headers={"Content-Type": "application/json"},
        )
    assert oversized.status_code == 400

    with build_client(tmp_path / "missing-token") as (client, settings, store):
        del store
        settings.v0_webhook_token = None
        unconfigured = client.post("/api/v1/webhooks/v0/anything", json={})
    assert unconfigured.status_code == 503
