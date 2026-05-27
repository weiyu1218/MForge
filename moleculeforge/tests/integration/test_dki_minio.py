"""Integration tests for DKI MinIO object storage."""

from __future__ import annotations

import os
from uuid import uuid4

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest.fixture
async def minio_client():
    required = (
        "MINIO_ENDPOINT_URL",
        "MINIO_ACCESS_KEY",
        "MINIO_SECRET_KEY",
        "MINIO_BUCKET",
    )
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        pytest.skip(", ".join(missing) + " required for MinIO integration tests")
    from mf_core.db.minio_client import MinIOStorageClient

    return MinIOStorageClient(
        endpoint_url=os.environ["MINIO_ENDPOINT_URL"],
        access_key=os.environ["MINIO_ACCESS_KEY"],
        secret_key=os.environ["MINIO_SECRET_KEY"],
        bucket=os.environ["MINIO_BUCKET"],
    )


async def test_put_get_and_hash(minio_client) -> None:
    object_name = f"integration/{uuid4().hex}.bin"
    payload = b"moleculeforge-dki-minio"

    await minio_client.put_object(object_name, payload, "application/octet-stream")

    assert await minio_client.get_object(object_name) == payload
    assert await minio_client.object_exists(object_name) is True
    assert len(await minio_client.object_sha256(object_name)) == 64
