"""Unit tests for MinIOStorageClient and path helpers (Mock aiobotocore)."""

from __future__ import annotations

import hashlib

import pytest
from botocore.exceptions import ClientError
from mf_core.db.object_store.paths import (
    BUCKET,
    audit_log_path,
    conformer_path,
    fep_result_path,
    md_trajectory_path,
    model_weights_path,
    xdl_protocol_path,
)


@pytest.mark.unit
def test_conformer_path() -> None:
    assert conformer_path("mol-001", 0) == "conformers/mol-001/0000.sdf"
    assert conformer_path("mol-001", 42) == "conformers/mol-001/0042.sdf"


@pytest.mark.unit
def test_md_trajectory_path() -> None:
    assert md_trajectory_path("run-1", "sim_0") == "md_trajectories/run-1/sim_0.xtc"


@pytest.mark.unit
def test_fep_result_path() -> None:
    assert fep_result_path("run-1", "pair_0") == "fep_results/run-1/pair_0.json"


@pytest.mark.unit
def test_model_weights_path() -> None:
    assert model_weights_path("humu-encoder", "v2") == "models/humu-encoder/v2/weights/"


@pytest.mark.unit
def test_audit_log_path() -> None:
    assert audit_log_path("2026-05-04", "run-001") == "audit_logs/2026-05-04/run-001/signed.jsonl"


@pytest.mark.unit
def test_xdl_protocol_path() -> None:
    assert xdl_protocol_path("mol-001", "route-1") == "xdl_protocols/mol-001/route-1.xdl"


@pytest.mark.unit
def test_bucket_name() -> None:
    assert BUCKET == "mf-data"


@pytest.mark.unit
def test_minio_client_init() -> None:
    from mf_core.db.minio_client import MinIOStorageClient

    client = MinIOStorageClient()
    assert client.endpoint_url == "http://localhost:9000"
    assert client.bucket == "mf-data"
    assert client._session is not None


class _Body:
    def __init__(self, data: bytes) -> None:
        self.data = data

    async def read(self) -> bytes:
        return self.data


class _FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    async def put_object(self, Bucket: str, Key: str, Body: bytes, ContentType: str):
        self.objects[(Bucket, Key)] = Body
        return {"ETag": "etag"}

    async def get_object(self, Bucket: str, Key: str):
        return {"Body": _Body(self.objects[(Bucket, Key)])}

    async def head_object(self, Bucket: str, Key: str):
        if (Bucket, Key) not in self.objects:
            raise KeyError(Key)
        return {"ContentLength": len(self.objects[(Bucket, Key)])}


@pytest.mark.asyncio
async def test_minio_client_put_get_and_hash() -> None:
    from mf_core.db.minio_client import MinIOStorageClient

    s3_client = _FakeS3Client()
    client = MinIOStorageClient(s3_client=s3_client)
    payload = b"molecule-artifact"

    await client.put_object("artifacts/mol.sdf", payload, "chemical/x-mdl-sdfile")

    assert await client.get_object("artifacts/mol.sdf") == payload
    assert await client.object_exists("artifacts/mol.sdf") is True
    assert await client.object_sha256("artifacts/mol.sdf") == hashlib.sha256(payload).hexdigest()


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["404", "NoSuchKey", "NotFound"])
async def test_minio_object_exists_returns_false_only_for_missing_codes(code: str) -> None:
    from mf_core.db.minio_client import MinIOStorageClient

    class MissingClient:
        async def head_object(self, **kwargs):
            raise ClientError({"Error": {"Code": code}}, "HeadObject")

    client = MinIOStorageClient(s3_client=MissingClient())

    assert await client.object_exists("missing") is False


@pytest.mark.asyncio
async def test_minio_object_exists_propagates_non_missing_errors() -> None:
    from mf_core.db.minio_client import MinIOStorageClient

    class FailingClient:
        async def head_object(self, **kwargs):
            raise ClientError({"Error": {"Code": "AccessDenied"}}, "HeadObject")

    client = MinIOStorageClient(s3_client=FailingClient())

    with pytest.raises(ClientError):
        await client.object_exists("forbidden")


@pytest.mark.asyncio
async def test_minio_put_object_if_absent_uses_s3_precondition() -> None:
    from mf_core.db.minio_client import MinIOStorageClient

    calls: list[dict] = []

    class ConditionalClient:
        async def put_object(self, **kwargs):
            calls.append(kwargs)

    client = MinIOStorageClient(s3_client=ConditionalClient())

    created = await client.put_object_if_absent(
        "provenance/artifact.json",
        b"record",
        "application/json",
    )

    assert created is True
    assert calls == [
        {
            "Bucket": "mf-data",
            "Key": "provenance/artifact.json",
            "Body": b"record",
            "ContentType": "application/json",
            "IfNoneMatch": "*",
        }
    ]


@pytest.mark.asyncio
async def test_minio_put_object_if_absent_reports_precondition_race() -> None:
    from mf_core.db.minio_client import MinIOStorageClient

    class RacingClient:
        async def put_object(self, **kwargs):
            raise ClientError(
                {
                    "Error": {"Code": "PreconditionFailed"},
                    "ResponseMetadata": {"HTTPStatusCode": 412},
                },
                "PutObject",
            )

    client = MinIOStorageClient(s3_client=RacingClient())

    assert (
        await client.put_object_if_absent(
            "provenance/artifact.json",
            b"record",
            "application/json",
        )
        is False
    )


@pytest.mark.asyncio
async def test_minio_ensure_bucket_creates_missing_bucket() -> None:
    from mf_core.db.minio_client import MinIOStorageClient

    calls: list[tuple[str, str]] = []

    class BucketClient:
        async def head_bucket(self, **kwargs):
            calls.append(("head", kwargs["Bucket"]))
            raise ClientError({"Error": {"Code": "404"}}, "HeadBucket")

        async def create_bucket(self, **kwargs):
            calls.append(("create", kwargs["Bucket"]))

    client = MinIOStorageClient(s3_client=BucketClient())

    await client.ensure_bucket()

    assert calls == [("head", "mf-data"), ("create", "mf-data")]
