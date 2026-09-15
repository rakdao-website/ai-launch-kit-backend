"""Durable database-backed worker process."""

import argparse
import asyncio
import socket
import uuid
from collections.abc import Awaitable, Callable

import httpx
import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from launchkit.assets import create_asset_store, load_dotenv_file
from launchkit.builds.handlers import create_build_job_handlers
from launchkit.core.config import Settings, get_settings
from launchkit.core.logging import configure_logging
from launchkit.core.tls import use_system_certificates
from launchkit.deployment.handlers import create_deployment_job_handlers
from launchkit.persistence import PersistenceRepository, create_database
from launchkit.persistence.models import JobRecord
from launchkit.workflows.handlers import create_workflow_job_handlers

JobHandler = Callable[[JobRecord, AsyncSession], Awaitable[None]]


class Worker:
    def __init__(self, settings: Settings, handlers: dict[str, JobHandler] | None = None) -> None:
        self._settings = settings
        self._database = create_database(settings)
        self._handlers = handlers or {}
        self._worker_id = f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
        self._logger = structlog.get_logger(__name__)

    async def run_once(self) -> int:
        async with self._database.sessions() as session:
            repository = PersistenceRepository(session)
            jobs = await repository.lease_jobs(
                worker_id=self._worker_id,
                limit=self._settings.worker_batch_size,
                lease_seconds=self._settings.worker_lease_seconds,
            )
            await repository.commit()

        for job in jobs:
            await self._run_job(job.id)
        return len(jobs)

    async def run_forever(self) -> None:
        self._logger.info("worker_started", worker_id=self._worker_id)
        try:
            while True:
                count = await self.run_once()
                if count == 0:
                    await asyncio.sleep(self._settings.worker_poll_seconds)
        finally:
            await self._database.close()

    async def close(self) -> None:
        await self._database.close()

    async def _run_job(self, job_id: str) -> None:
        async with self._database.sessions() as session:
            job = await session.get(JobRecord, job_id)
            if job is None or job.lease_owner != self._worker_id:
                return
            handler = self._handlers.get(job.kind)
            if handler is None:
                job.status = "failed"
                job.last_error = f"No handler registered for job kind: {job.kind}"
            else:
                try:
                    await handler(job, session)
                    job.status = "completed"
                    job.last_error = None
                except Exception as exc:
                    job.status = "failed"
                    job.last_error = str(exc)[:1000]
                    # Always include the message in structured fields — UAT JSON logs
                    # previously only showed exc_info=true with no usable text.
                    self._logger.exception(
                        "job_failed",
                        job_id=job.id,
                        kind=job.kind,
                        error=job.last_error,
                    )
            job.lease_owner = None
            job.leased_until = None
            await session.commit()


async def _main(once: bool) -> None:
    load_dotenv_file()
    settings = get_settings()
    configure_logging(settings)
    use_system_certificates(enabled=not bool(settings.s3_bucket))
    asset_store = create_asset_store(settings)
    timeout = httpx.Timeout(120, connect=10)
    async with httpx.AsyncClient(timeout=timeout) as client:
        runtime = create_workflow_job_handlers(settings, client, asset_store)
        builds = create_build_job_handlers(settings, client, asset_store)
        deployments = create_deployment_job_handlers(settings, client, asset_store)
        worker = Worker(
            settings,
            handlers={**runtime.handlers, **builds.handlers, **deployments.handlers},
        )
        if once:
            await worker.run_once()
            await worker.close()
        else:
            await worker.run_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AI Launch Kit durable worker")
    parser.add_argument("--once", action="store_true", help="Lease one batch and exit")
    args = parser.parse_args()
    asyncio.run(_main(args.once))


if __name__ == "__main__":
    main()
