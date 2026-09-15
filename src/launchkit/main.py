"""FastAPI application composition: V1 API plus InnovationCity OAuth PKCE auth."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from launchkit.api import api_router
from launchkit.api.errors import register_error_handlers
from launchkit.api.middleware import RequestIdMiddleware
from launchkit.assets import AssetBlobStore, create_asset_store, load_dotenv_file
from launchkit.auth import auth_router
from launchkit.builds.handlers import create_build_job_handlers
from launchkit.core.config import Settings, get_settings
from launchkit.core.logging import configure_logging
from launchkit.deployment.handlers import create_deployment_job_handlers
from launchkit.core.tls import outbound_verify, use_system_certificates
from launchkit.persistence import Database, create_database
from launchkit.worker import Worker
from launchkit.workflows.handlers import create_workflow_job_handlers


def _cors_origins(settings: Settings) -> list[str]:
    merged: list[str] = []
    for raw in (settings.frontend_origins, settings.auth_cors_origins):
        for origin in raw.split(","):
            origin = origin.strip()
            if origin and origin not in merged:
                merged.append(origin)
    return merged


async def _run_embedded_worker(settings: Settings, asset_store: AssetBlobStore) -> None:
    """Process durable jobs inside the API process so local dev does not require a second CLI."""

    logger = structlog.get_logger(__name__)
    timeout = httpx.Timeout(120, connect=10)
    async with httpx.AsyncClient(timeout=timeout, verify=outbound_verify()) as client:
        runtime = create_workflow_job_handlers(settings, client, asset_store)
        builds = create_build_job_handlers(settings, client, asset_store)
        deployments = create_deployment_job_handlers(settings, client, asset_store)
        worker = Worker(
            settings,
            handlers={**runtime.handlers, **builds.handlers, **deployments.handlers},
        )
        logger.info("embedded_worker_started")
        try:
            await worker.run_forever()
        except asyncio.CancelledError:
            logger.info("embedded_worker_stopped")
            raise
        finally:
            await worker.close()


def create_app(
    settings: Settings | None = None,
    database: Database | None = None,
    asset_store: AssetBlobStore | None = None,
) -> FastAPI:
    """Compose transport dependencies without requiring optional provider credentials."""

    # AWS_* from .env for boto3. Never inject truststore when S3 is on — that
    # combination RecursionErrors inside botocore SSLContext on startup.
    load_dotenv_file()
    resolved_settings = settings or get_settings()
    configure_logging(resolved_settings)
    use_system_certificates(enabled=not bool(resolved_settings.s3_bucket))

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        resolved_database = database or create_database(resolved_settings)
        resolved_asset_store = asset_store or create_asset_store(resolved_settings)
        application.state.settings = resolved_settings
        application.state.database = resolved_database
        application.state.asset_store = resolved_asset_store
        worker_task: asyncio.Task[None] | None = None
        # Tests exercise the worker explicitly; avoid background leasing races.
        if resolved_settings.environment != "test":
            worker_task = asyncio.create_task(
                _run_embedded_worker(resolved_settings, resolved_asset_store)
            )
        try:
            yield
        finally:
            if worker_task is not None:
                worker_task.cancel()
                try:
                    await worker_task
                except asyncio.CancelledError:
                    pass
            await resolved_database.close()

    application = FastAPI(
        title=resolved_settings.app_name,
        debug=resolved_settings.debug,
        version="1.0.0",
        lifespan=lifespan,
    )
    application.add_middleware(RequestIdMiddleware)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(resolved_settings),
        # IC OAuth session cookies require credentialed CORS.
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=[
            "Accept",
            "Authorization",
            "Content-Type",
            "Idempotency-Key",
            "Last-Event-ID",
        ],
    )
    register_error_handlers(application)
    application.include_router(api_router)
    application.include_router(auth_router)
    application.dependency_overrides[get_settings] = lambda: resolved_settings
    return application


app = create_app()
