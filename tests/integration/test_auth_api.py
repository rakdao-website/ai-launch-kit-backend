"""Restricted local/test authentication and readiness contracts."""

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from launchkit.core.config import Settings
from launchkit.main import create_app
from launchkit.persistence import create_database
from launchkit.persistence.base import Base


@contextmanager
def auth_client(tmp_path: Path) -> Iterator[TestClient]:
    settings = Settings(
        environment="test",
        auth_mode="fixed_otp",
        auth_email="test@innovationcity.com",
        auth_otp="847291",
        auth_token_secret="a-long-test-token-secret-that-is-not-used-in-production",
        site_url="http://localhost:8000",
        database_url=f"sqlite+aiosqlite:///{(tmp_path / 'auth.sqlite3').as_posix()}",
        frontend_origins="https://app.example",
        _env_file=None,
    )
    database = create_database(settings)

    async def create_schema() -> None:
        async with database.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    asyncio.run(create_schema())
    with TestClient(create_app(settings, database)) as client:
        yield client


def test_fixed_account_login_protects_projects_and_allows_readiness(tmp_path: Path) -> None:
    with auth_client(tmp_path) as client:
        health = client.get("/api/v1/health")
        ready = client.get("/api/v1/ready")
        unauthorized = client.post("/api/v1/projects", json={})
        requested = client.post(
            "/api/v1/auth/request-code",
            json={"email": "test@innovationcity.com"},
        )
        verified = client.post(
            "/api/v1/auth/verify",
            json={"email": "test@innovationcity.com", "code": "847291"},
        )
        token = verified.json()["accessToken"]
        created = client.post(
            "/api/v1/projects",
            json={},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert health.status_code == 200
    assert ready.json()["status"] == "ready"
    assert unauthorized.status_code == 401
    assert unauthorized.json()["error"]["code"] == "authentication_required"
    assert requested.status_code == 202
    assert verified.status_code == 200
    assert verified.json()["expiresInSeconds"] == 8 * 60 * 60
    assert created.status_code == 201


def test_fixed_account_rejects_wrong_email_code_and_token(tmp_path: Path) -> None:
    with auth_client(tmp_path) as client:
        wrong_email = client.post(
            "/api/v1/auth/request-code",
            json={"email": "someone@example.com"},
        )
        wrong_code = client.post(
            "/api/v1/auth/verify",
            json={"email": "test@innovationcity.com", "code": "654321"},
        )
        wrong_token = client.post(
            "/api/v1/projects",
            json={},
            headers={"Authorization": "Bearer invalid"},
        )

    assert wrong_email.status_code == 401
    assert wrong_code.status_code == 401
    assert wrong_token.status_code == 401


def test_fixed_otp_rejected_for_public_deployments() -> None:
    with pytest.raises(ValidationError, match="public"):
        Settings(
            environment="test",
            auth_mode="fixed_otp",
            auth_otp="847291",
            site_url="https://launchkit-api-uat.innovationcity.com",
            _env_file=None,
        )


def test_fixed_otp_rejected_outside_local_test() -> None:
    with pytest.raises(ValidationError, match="fixed_otp"):
        Settings(
            environment="staging",
            auth_mode="fixed_otp",
            auth_otp="847291",
            site_url="http://localhost:8000",
            _env_file=None,
        )
