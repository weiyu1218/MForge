"""MinIO storage client for molecule artifacts and checkpoints."""
from __future__ import annotations

import hashlib
from typing import Any


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
        except Exception:
            return False

    async def object_sha256(self, object_name: str) -> str:
        return hashlib.sha256(await self.get_object(object_name)).hexdigest()
