"""MinIO storage client for molecule artifacts and checkpoints."""

from __future__ import annotations

import hashlib
from typing import Any

from botocore.exceptions import ClientError


class MinIOStorageClient:
    """Wrapper around MinIO S3-compatible object store."""

    def __init__(
        self,
        endpoint_url: str = "http://localhost:9000",
        access_key: str = "minioadmin",
        secret_key: str = "minioadmin",
        bucket: str = "mf-data",
        s3_client: Any | None = None,
    ):
        self.endpoint_url = endpoint_url
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket = bucket
        self._client = s3_client
        self._session: Any = self._create_session()

    def _create_session(self) -> Any:
        if self._client is not None:
            return object()
        try:
            from aiobotocore.session import get_session

            return get_session()
        except ImportError as exc:
            raise RuntimeError("aiobotocore is required for MinIOStorageClient") from exc

    async def put_object(
        self,
        object_name: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> None:
        client = self._client
        if client is not None:
            await client.put_object(
                Bucket=self.bucket,
                Key=object_name,
                Body=data,
                ContentType=content_type,
            )
            return
        async with self._session.create_client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        ) as created_client:
            await created_client.put_object(
                Bucket=self.bucket,
                Key=object_name,
                Body=data,
                ContentType=content_type,
            )

    async def put_object_if_absent(
        self,
        object_name: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> bool:
        request = {
            "Bucket": self.bucket,
            "Key": object_name,
            "Body": data,
            "ContentType": content_type,
            "IfNoneMatch": "*",
        }
        try:
            client = self._client
            if client is not None:
                await client.put_object(**request)
                return True
            async with self._session.create_client(
                "s3",
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
            ) as created_client:
                await created_client.put_object(**request)
                return True
        except ClientError as exc:
            if _is_precondition_failed(exc):
                return False
            raise

    async def get_object(self, object_name: str) -> bytes:
        client = self._client
        if client is not None:
            response = await client.get_object(Bucket=self.bucket, Key=object_name)
            return await response["Body"].read()
        async with self._session.create_client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        ) as created_client:
            response = await created_client.get_object(
                Bucket=self.bucket,
                Key=object_name,
            )
            return await response["Body"].read()

    async def object_exists(self, object_name: str) -> bool:
        client = self._client
        try:
            if client is not None:
                await client.head_object(Bucket=self.bucket, Key=object_name)
                return True
            async with self._session.create_client(
                "s3",
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
            ) as created_client:
                await created_client.head_object(Bucket=self.bucket, Key=object_name)
                return True
        except ClientError as exc:
            if _is_not_found(exc):
                return False
            raise

    async def ensure_bucket(self) -> None:
        client = self._client
        if client is not None:
            await self._ensure_bucket_with_client(client)
            return
        async with self._session.create_client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        ) as created_client:
            await self._ensure_bucket_with_client(created_client)

    async def _ensure_bucket_with_client(self, client: Any) -> None:
        try:
            await client.head_bucket(Bucket=self.bucket)
            return
        except ClientError as exc:
            if not _is_not_found(exc):
                raise
        try:
            await client.create_bucket(Bucket=self.bucket)
        except ClientError as exc:
            if _error_code(exc) != "BucketAlreadyOwnedByYou":
                raise

    async def object_sha256(self, object_name: str) -> str:
        return hashlib.sha256(await self.get_object(object_name)).hexdigest()


def _error_code(exc: ClientError) -> str:
    return str(exc.response.get("Error", {}).get("Code", ""))


def _http_status(exc: ClientError) -> int | None:
    status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return status if isinstance(status, int) else None


def _is_not_found(exc: ClientError) -> bool:
    return _error_code(exc) in {"404", "NoSuchKey", "NotFound"} or _http_status(exc) == 404


def _is_precondition_failed(exc: ClientError) -> bool:
    return _error_code(exc) in {"412", "PreconditionFailed"} or _http_status(exc) == 412
