"""MMPT-RAG Generator Service - gRPC server for retrieval-augmented generation."""

import asyncio
import inspect
import os
import sys
import time
from concurrent import futures
from urllib.parse import unquote, urlparse

import grpc
from mf_chem.molecule.parsing import canonicalize
from mf_core.artifacts import (
    ArtifactRequirement,
    CommandRequirement,
    RequirementStatus,
    check_artifact,
    check_command,
    require_available,
)
from mf_core.plugins.generator import (
    GeneratorRequestError,
    GeneratorResultError,
    build_generate_response,
    build_generator_info,
    validate_generate_request,
)
from mf_core.proto_gen.moleculeforge.v1.generator import generator_pb2_grpc

_REQUIREMENTS = (ArtifactRequirement("mmpt_index", "MMPT_INDEX_URI", kind="uri"),)
_COMMAND_REQUIREMENTS = (
    CommandRequirement("mmpt_patent_rag_command", "MMPT_PATENT_RAG_COMMAND"),
    CommandRequirement("mmpt_seq2seq_decoder_command", "MMPT_SEQ2SEQ_DECODER_COMMAND"),
)
_GENERATOR_NAME = "mmpt_rag"
_MAX_BATCH_SIZE = 256
_ALLOW_VALIDATION_ARTIFACT_ENV = "MMPT_ALLOW_VALIDATION_ARTIFACT"


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
    validation_status = _validation_artifact_opt_in_status()
    if validation_status is not None:
        statuses.append(validation_status)
    return statuses


def _validation_artifact_opt_in_status() -> RequirementStatus | None:
    index_uri = os.environ.get("MMPT_INDEX_URI", "").strip()
    if not index_uri:
        return None
    try:
        index_path = _index_path_from_uri(index_uri)
        from mf_generators.mmpt_rag.generator import (
            load_validation_artifact_metadata,
        )

        metadata = load_validation_artifact_metadata(index_path)
    except Exception as exc:
        return RequirementStatus(
            name="mmpt_validation_artifact_opt_in",
            configured=False,
            available=False,
            required=True,
            path=index_uri,
            source=_ALLOW_VALIDATION_ARTIFACT_ENV,
            message=f"MMPT validation artifact metadata is invalid: {exc}",
        )
    if metadata is None:
        return None
    opted_in = os.environ.get(_ALLOW_VALIDATION_ARTIFACT_ENV, "").strip() == "true"
    return RequirementStatus(
        name="mmpt_validation_artifact_opt_in",
        configured=opted_in,
        available=opted_in,
        required=True,
        path=index_uri,
        source=_ALLOW_VALIDATION_ARTIFACT_ENV,
        message=(
            "MMPT validation artifact is explicitly enabled"
            if opted_in
            else f"{_ALLOW_VALIDATION_ARTIFACT_ENV}=true is required"
        ),
    )


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


async def _abort_internal(context, message: str):
    if context is not None and hasattr(context, "abort"):
        await context.abort(grpc.StatusCode.INTERNAL, message)
    raise RuntimeError(message)


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
            statuses = _require_runtime()
        except RuntimeError:
            return await _abort_unavailable(context)
        if self.generator is None:
            return await _abort_unavailable(context)
        try:
            request_context = validate_generate_request(
                request,
                max_batch_size=_MAX_BATCH_SIZE,
            )
            params = dict(getattr(request, "generator_params", {}) or {})
            seed = _seed(params)
        except (GeneratorRequestError, ValueError) as exc:
            return await _abort_invalid_argument(context, str(exc))
        start = time.perf_counter()
        try:
            molecules = await _collect_molecules(
                self.generator.generate(
                    request_context.hciv,
                    request_context.intent_cone,
                    request_context.cig,
                    n_samples=request_context.batch_size,
                    seed=seed,
                )
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return await _abort_internal(context, str(exc))
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        try:
            return build_generate_response(
                generator_name=_GENERATOR_NAME,
                request=request,
                molecules=molecules,
                statuses=statuses,
                elapsed_ms=elapsed_ms,
                canonicalize_smiles=canonicalize,
            )
        except GeneratorResultError as exc:
            return await _abort_internal(context, str(exc))

    async def GenerateStream(self, request_iterator, context):  # noqa: N802
        async for request in request_iterator:
            yield await self.Generate(request, context)

    async def BatchGenerate(self, request_iterator, context):  # noqa: N802
        async for request in request_iterator:
            yield await self.Generate(request, context)

    async def Info(self, request, context):  # noqa: N802
        return await build_generator_info(
            generator_name=_GENERATOR_NAME,
            generator=self.generator,
            statuses=_runtime_statuses(),
            fallback={
                "version": "0.1.0",
                "description": "MMPT-RAG matched molecular pair generator",
                "supported_properties": ["qed", "sa_score", "novelty"],
                "max_batch_size": _MAX_BATCH_SIZE,
                "supports_streaming": True,
                "requires_gpu": False,
            },
        )


async def serve():
    _require_runtime()
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=8))
    generator_pb2_grpc.add_GeneratorServiceServicer_to_server(MMPTGeneratorServicer(), server)
    server.add_insecure_port("[::]:50069")
    await server.start()
    print("MMPT-RAG Generator Service running on :50069")
    await server.wait_for_termination()


def _main(argv: list[str]) -> None:
    if not argv:
        asyncio.run(serve())
        return
    if len(argv) != 2 or argv[0] != "--bootstrap-validation-artifacts":
        raise ValueError(
            "usage: mmpt_generator_svc.main "
            "--bootstrap-validation-artifacts <directory>"
        )
    from mf_generators.mmpt_rag.generator import bootstrap_validation_artifacts

    paths = asyncio.run(bootstrap_validation_artifacts(argv[1]))
    sys.stdout.write(f"{paths['metadata'].parent}\n")


if __name__ == "__main__":
    _main(sys.argv[1:])
