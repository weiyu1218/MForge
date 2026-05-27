"""MMPT-RAG Generator Service - gRPC server for retrieval-augmented generation."""
import asyncio
import inspect
import json
import os
import time
from concurrent import futures
from urllib.parse import unquote, urlparse

import grpc
from mf_core.artifacts import (
    ArtifactRequirement,
    RequirementStatus,
    check_artifact,
    require_available,
)
from mf_core.proto_gen.moleculeforge.v1.generator import generator_pb2_grpc

_REQUIREMENTS = (ArtifactRequirement("mmpt_index", "MMPT_INDEX_URI", kind="uri"),)
_GENERATOR_NAME = "mmpt_rag"


def _require_runtime() -> list[RequirementStatus]:
    statuses = [check_artifact(requirement) for requirement in _REQUIREMENTS]
    require_available(statuses)
    return statuses


def runtime_status() -> list[dict]:
    return [check_artifact(requirement).to_dict() for requirement in _REQUIREMENTS]


async def _abort_unavailable(context):
    statuses = [check_artifact(requirement) for requirement in _REQUIREMENTS]
    try:
        require_available(statuses)
    except RuntimeError as exc:
        message = str(exc)
    else:
        message = "MMPT-RAG generator runner is not configured"
    if context is not None and hasattr(context, "abort"):
        await context.abort(grpc.StatusCode.FAILED_PRECONDITION, message)
    raise RuntimeError(message)


async def _abort_invalid_argument(context, message: str):
    if context is not None and hasattr(context, "abort"):
        await context.abort(grpc.StatusCode.INVALID_ARGUMENT, message)
    raise ValueError(message)


def _batch_size(request) -> int:
    value = int(getattr(request, "batch_size", 0))
    if value <= 0:
        raise ValueError("batch_size must be positive")
    return value


def _serialize_molecule(molecule) -> bytes:
    if hasattr(molecule, "model_dump_json"):
        return molecule.model_dump_json().encode("utf-8")
    if isinstance(molecule, dict):
        return json.dumps(molecule, sort_keys=True).encode("utf-8")
    raise TypeError(f"Unsupported molecule payload: {type(molecule)!r}")


def _index_path_from_uri(index_uri: str) -> str:
    parsed = urlparse(index_uri)
    if parsed.scheme != "file":
        raise ValueError("Only file:// MMPT index URIs are supported by local service mode")
    if parsed.netloc not in {"", "localhost"}:
        raise ValueError("Only local file:// MMPT index URIs are supported")
    return unquote(parsed.path)


def _build_generator():
    from mf_generators.mmpt_rag.generator import MMPTRAGGenerator

    return MMPTRAGGenerator(index_path=_index_path_from_uri(os.environ["MMPT_INDEX_URI"]))


async def _collect_molecules(result) -> list:
    if hasattr(result, "__aiter__"):
        return [item async for item in result]
    if inspect.isawaitable(result):
        return await result
    return list(result)


def _seed(params: dict) -> int | None:
    raw_seed = params.get("seed")
    if raw_seed in {None, ""}:
        return None
    return int(raw_seed)


class MMPTGeneratorServicer:
    def __init__(self, generator=None):
        self.generator = generator if generator is not None else _build_generator()

    async def Generate(self, request, context):  # noqa: N802
        """Generate molecules via MMPT-RAG (matched molecular pair transform + RAG)."""
        try:
            _require_runtime()
        except RuntimeError:
            return await _abort_unavailable(context)
        if self.generator is None:
            return await _abort_unavailable(context)
        try:
            batch_size = _batch_size(request)
            params = dict(getattr(request, "generator_params", {}) or {})
            seed = _seed(params)
        except ValueError as exc:
            return await _abort_invalid_argument(context, str(exc))
        start = time.perf_counter()
        molecules = await _collect_molecules(
            self.generator.generate(
                None,
                None,
                None,
                n_samples=batch_size,
                seed=seed,
            )
        )
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return type(
            "GenerateResponse",
            (),
            {
                "generator_name": _GENERATOR_NAME,
                "generation_id": getattr(request, "project_id", ""),
                "molecules": [_serialize_molecule(mol) for mol in molecules],
                "humu_embeddings": [],
                "aggregate_stats": {},
                "elapsed_ms": elapsed_ms,
            },
        )()

    async def GenerateStream(self, request_iterator, context):  # noqa: N802
        async for request in request_iterator:
            yield await self.Generate(request, context)

    async def BatchGenerate(self, request_iterator, context):  # noqa: N802
        async for request in request_iterator:
            yield await self.Generate(request, context)

    async def Info(self, request, context):  # noqa: N802
        return request


async def serve():
    _require_runtime()
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=8))
    generator_pb2_grpc.add_GeneratorServiceServicer_to_server(MMPTGeneratorServicer(), server)
    server.add_insecure_port("[::]:50069")
    await server.start()
    print("MMPT-RAG Generator Service running on :50069")
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
