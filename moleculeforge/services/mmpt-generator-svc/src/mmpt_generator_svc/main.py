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
    CommandRequirement,
    RequirementStatus,
    check_artifact,
    check_command,
    require_available,
)
from mf_core.proto_gen.moleculeforge.v1.generator import generator_pb2, generator_pb2_grpc
from mf_core.types.humu import IntentCone

_REQUIREMENTS = (ArtifactRequirement("mmpt_index", "MMPT_INDEX_URI", kind="uri"),)
_COMMAND_REQUIREMENTS = (
    CommandRequirement("mmpt_patent_rag_command", "MMPT_PATENT_RAG_COMMAND"),
    CommandRequirement("mmpt_seq2seq_decoder_command", "MMPT_SEQ2SEQ_DECODER_COMMAND"),
)
_GENERATOR_NAME = "mmpt_rag"


def _require_runtime() -> list[RequirementStatus]:
    statuses = _runtime_statuses()
    require_available(statuses)
    return statuses


def runtime_status() -> list[dict]:
    return [status.to_dict() for status in _runtime_statuses()]


def _runtime_statuses() -> list[RequirementStatus]:
    statuses = [_index_status()]
    for requirement in _COMMAND_REQUIREMENTS:
        if os.environ.get(requirement.env_var, "").strip():
            statuses.append(check_command(requirement))
    return statuses


def _index_status() -> RequirementStatus:
    index_uri = os.environ.get("MMPT_INDEX_URI", "").strip()
    if not index_uri:
        return check_artifact(_REQUIREMENTS[0])
    parsed = urlparse(index_uri)
    if parsed.scheme != "file":
        return RequirementStatus(
            name="mmpt_index",
            configured=True,
            available=False,
            required=True,
            path=index_uri,
            source="MMPT_INDEX_URI",
            message="MMPT_INDEX_URI must use file:// for local MMPT index mode",
        )
    if parsed.netloc not in {"", "localhost"}:
        return RequirementStatus(
            name="mmpt_index",
            configured=True,
            available=False,
            required=True,
            path=index_uri,
            source="MMPT_INDEX_URI",
            message="MMPT_INDEX_URI must reference a local file:// URI",
        )
    return check_artifact(_REQUIREMENTS[0])


async def _abort_unavailable(context):
    statuses = _runtime_statuses()
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


def _intent_cone_from_request(request) -> IntentCone | None:
    raw = getattr(request, "intent_cone", None)
    if raw in (None, "", b"", {}):
        return None
    if isinstance(raw, IntentCone):
        return raw
    if isinstance(raw, bytes):
        raw = json.loads(raw.decode("utf-8"))
    elif isinstance(raw, str):
        raw = json.loads(raw)
    elif hasattr(raw, "model_dump"):
        raw = raw.model_dump(mode="json")
    if isinstance(raw, dict):
        return IntentCone.model_validate(raw)
    raise TypeError(f"Unsupported intent_cone payload: {type(raw)!r}")


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
                _intent_cone_from_request(request),
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
        return generator_pb2.GeneratorInfo(
            name=_GENERATOR_NAME,
            version="0.1.0",
            description="MMPT-RAG matched molecular pair generator",
            supported_properties=["qed", "sa_score", "novelty"],
            max_batch_size=256,
            supports_streaming=True,
            requires_gpu=False,
        )


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
