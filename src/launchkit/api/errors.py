"""Uniform V1 API error translation and safe response envelopes."""

from collections.abc import Mapping, Sequence
from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from launchkit.assets import UploadTooLargeError, UploadValidationError
from launchkit.builds import BuildNotFoundError, GenerationQuotaExceededError
from launchkit.builds.webhooks import WebhookAccessError, WebhookPayloadError
from launchkit.core.exceptions import (
    AuthenticationError,
    ConfigurationError,
    DomainError,
    ProviderError,
    RateLimitError,
)
from launchkit.deployment.service import DeploymentNotFoundError
from launchkit.deployment.webhooks import (
    VercelWebhookAccessError,
    VercelWebhookPayloadError,
)
from launchkit.projects import ProjectNotFoundError
from launchkit.workflows import WorkflowNotFoundError


def request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "req_unknown")


def error_response(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
    details: list[dict[str, Any]] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "details": details or [],
                "requestId": request_id(request),
            }
        },
    )


def validation_details(errors: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "field": ".".join(str(part) for part in error.get("loc", ())),
            "message": str(error.get("msg", "Invalid value")),
            "type": str(error.get("type", "validation_error")),
        }
        for error in errors
    ]


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AuthenticationError)
    async def handle_authentication(request: Request, exc: AuthenticationError) -> JSONResponse:
        return error_response(
            request,
            status_code=401,
            code="authentication_required",
            message=str(exc),
        )

    @app.exception_handler(RateLimitError)
    async def handle_rate_limit(request: Request, exc: RateLimitError) -> JSONResponse:
        return error_response(
            request,
            status_code=429,
            code="rate_limited",
            message=str(exc) or "Too many requests. Try again later.",
        )

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return error_response(
            request,
            status_code=422,
            code="invalid_input",
            message="The request contains invalid values.",
            details=validation_details(exc.errors()),
        )

    @app.exception_handler(ValidationError)
    async def handle_domain_validation(request: Request, exc: ValidationError) -> JSONResponse:
        return error_response(
            request,
            status_code=422,
            code="invalid_input",
            message="The project draft contains invalid values.",
            details=validation_details(exc.errors()),
        )

    @app.exception_handler(ProjectNotFoundError)
    async def handle_not_found(request: Request, _: ProjectNotFoundError) -> JSONResponse:
        return error_response(
            request, status_code=404, code="project_not_found", message="Project not found."
        )

    @app.exception_handler(WorkflowNotFoundError)
    async def handle_workflow_not_found(request: Request, _: WorkflowNotFoundError) -> JSONResponse:
        return error_response(
            request, status_code=404, code="resource_not_found", message="Resource not found."
        )

    @app.exception_handler(GenerationQuotaExceededError)
    async def handle_generation_quota(
        request: Request, exc: GenerationQuotaExceededError
    ) -> JSONResponse:
        return error_response(
            request,
            status_code=402,
            code="generation_quota_exceeded",
            message=str(exc),
        )

    @app.exception_handler(BuildNotFoundError)
    async def handle_build_not_found(request: Request, _: BuildNotFoundError) -> JSONResponse:
        return error_response(
            request, status_code=404, code="build_not_found", message="Build not found."
        )

    @app.exception_handler(DeploymentNotFoundError)
    async def handle_deployment_not_found(
        request: Request, _: DeploymentNotFoundError
    ) -> JSONResponse:
        return error_response(
            request,
            status_code=404,
            code="deployment_not_found",
            message="Deployment not found.",
        )

    @app.exception_handler(WebhookAccessError)
    async def handle_webhook_access(request: Request, _: WebhookAccessError) -> JSONResponse:
        return error_response(
            request, status_code=404, code="not_found", message="Resource not found."
        )

    @app.exception_handler(WebhookPayloadError)
    async def handle_webhook_payload(request: Request, exc: WebhookPayloadError) -> JSONResponse:
        return error_response(request, status_code=400, code="invalid_webhook", message=str(exc))

    @app.exception_handler(VercelWebhookAccessError)
    async def handle_vercel_webhook_access(
        request: Request, _: VercelWebhookAccessError
    ) -> JSONResponse:
        return error_response(
            request,
            status_code=401,
            code="invalid_webhook_signature",
            message="Invalid webhook signature.",
        )

    @app.exception_handler(VercelWebhookPayloadError)
    async def handle_vercel_webhook_payload(
        request: Request, exc: VercelWebhookPayloadError
    ) -> JSONResponse:
        return error_response(request, status_code=400, code="invalid_webhook", message=str(exc))

    @app.exception_handler(ConfigurationError)
    async def handle_configuration(request: Request, _: ConfigurationError) -> JSONResponse:
        return error_response(
            request,
            status_code=503,
            code="provider_configuration_missing",
            message="This service is not configured.",
        )

    @app.exception_handler(UploadTooLargeError)
    async def handle_upload_too_large(request: Request, exc: UploadTooLargeError) -> JSONResponse:
        return error_response(request, status_code=413, code="upload_too_large", message=str(exc))

    @app.exception_handler(UploadValidationError)
    async def handle_upload_validation(
        request: Request, exc: UploadValidationError
    ) -> JSONResponse:
        return error_response(request, status_code=422, code="invalid_upload", message=str(exc))

    @app.exception_handler(ProviderError)
    async def handle_provider(request: Request, exc: ProviderError) -> JSONResponse:
        status = 429 if exc.status_code == 429 else 503
        code = "rate_limited" if status == 429 else "provider_unavailable"
        return error_response(
            request,
            status_code=status,
            code=code,
            message="The external generation service is temporarily unavailable.",
        )

    @app.exception_handler(DomainError)
    async def handle_domain_error(request: Request, exc: DomainError) -> JSONResponse:
        return error_response(
            request,
            status_code=409,
            code="invalid_workflow_state",
            message=str(exc) or "The requested action is not valid right now.",
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = "not_found" if exc.status_code == 404 else "http_error"
        message = "Resource not found." if exc.status_code == 404 else "The request failed."
        return error_response(request, status_code=exc.status_code, code=code, message=message)

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        structlog.get_logger(__name__).exception(
            "unhandled_api_error", request_id=request_id(request), error_type=type(exc).__name__
        )
        return error_response(
            request,
            status_code=500,
            code="internal_error",
            message="An unexpected error occurred.",
        )
