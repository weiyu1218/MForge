"""HFM Molecule Generator Service - gRPC server for Hyperbolic Flow Matching generation."""

import asyncio
import os
import sys
import time
from concurrent import futures

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

_CHECKPOINT_REQUIREMENT = ArtifactRequirement("hfm_checkpoint", "HFM_CHECKPOINT_PATH")
_DECODER_REQUIREMENT = ArtifactRequirement("hfm_decoder", "HFM_DECODER_PATH")
_MOLECULAR_DECODER_COMMAND = CommandRequirement(
    "hfm_molecular_decoder_command",
    "HFM_MOLECULAR_DECODER_COMMAND",
)
_GENERATOR_NAME = "hfm_3d"
_MAX_BATCH_SIZE = 1024
_ALLOW_VALIDATION_ARTIFACT_ENV = "HFM_ALLOW_VALIDATION_ARTIFACT"


def _require_runtime() -> list[RequirementStatus]:
    statuses = _runtime_statuses()
    require_available(statuses)
    return statuses


def runtime_status() -> list[dict]:
    return [status.to_dict() for status in _runtime_statuses()]


def _runtime_statuses() -> list[RequirementStatus]:
    statuses = [check_artifact(_CHECKPOINT_REQUIREMENT)]
    decoder_command = os.environ.get("HFM_MOLECULAR_DECODER_COMMAND", "").strip()
    if decoder_command:
        statuses.append(check_command(_MOLECULAR_DECODER_COMMAND))
    else:
        statuses.append(check_artifact(_DECODER_REQUIREMENT))
    validation_status = _validation_artifact_opt_in_status()
    if validation_status is not None:
        statuses.append(validation_status)
    return statuses


def _validation_artifact_opt_in_status() -> RequirementStatus | None:
    checkpoint_path = os.environ.get("HFM_CHECKPOINT_PATH", "").strip()
    if not checkpoint_path:
        return None
    artifact_paths = [checkpoint_path]
    if not os.environ.get("HFM_MOLECULAR_DECODER_COMMAND", "").strip():
        decoder_path = os.environ.get("HFM_DECODER_PATH", "").strip()
        if decoder_path:
            artifact_paths.append(decoder_path)
    try:
        from mf_generators.hfm_3d.generator import (
            load_validation_artifact_metadata,
        )

        validation_path = ""
        for artifact_path in artifact_paths:
            metadata = load_validation_artifact_metadata(artifact_path)
            if metadata is not None and not validation_path:
                validation_path = artifact_path
    except Exception as exc:
        return RequirementStatus(
            name="hfm_validation_artifact_opt_in",
            configured=False,
            available=False,
            required=True,
            path=artifact_path,
            source=_ALLOW_VALIDATION_ARTIFACT_ENV,
            message=f"HFM validation artifact metadata is invalid: {exc}",
        )
    if not validation_path:
        return None
    opted_in = os.environ.get(_ALLOW_VALIDATION_ARTIFACT_ENV, "").strip() == "true"
    return RequirementStatus(
        name="hfm_validation_artifact_opt_in",
        configured=opted_in,
        available=opted_in,
        required=True,
        path=validation_path,
        source=_ALLOW_VALIDATION_ARTIFACT_ENV,
        message=(
            "HFM validation artifact is explicitly enabled"
            if opted_in
            else f"{_ALLOW_VALIDATION_ARTIFACT_ENV}=true is required"
        ),
    )


async def _abort_unavailable(context):
    statuses = _runtime_statuses()
    try:
        require_available(statuses)
    except RuntimeError as exc:
        message = str(exc)
    else:
        message = "HFM generator runner is not configured"
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


def _build_generator():
    from mf_generators.hfm_3d.generator import HFM3DGenerator

    return HFM3DGenerator(
        checkpoint_path=os.environ["HFM_CHECKPOINT_PATH"],
        decoder_path=os.environ.get("HFM_DECODER_PATH", ""),
    )


class HFMGeneratorServicer:
    def __init__(self, generator=None):
        self.generator = generator if generator is not None else _build_generator()

    async def Generate(self, request, context):
        """Generate molecules via Hyperbolic Flow Matching in Lorentz manifold."""
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
        except GeneratorRequestError as exc:
            return await _abort_invalid_argument(context, str(exc))
        params = dict(getattr(request, "generator_params", {}) or {})
        start = time.perf_counter()
        try:
            molecules = await self.generator.generate(
                batch_size=request_context.batch_size,
                intent_cone=request_context.intent_cone,
                **params,
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

    async def GenerateStream(self, request_iterator, context):
        async for request in request_iterator:
            yield await self.Generate(request, context)

    async def BatchGenerate(self, request_iterator, context):
        async for request in request_iterator:
            yield await self.Generate(request, context)

    async def Info(self, request, context):
        return await build_generator_info(
            generator_name=_GENERATOR_NAME,
            generator=self.generator,
            statuses=_runtime_statuses(),
            fallback={
                "version": "0.1.0",
                "description": "HFM-3D Lorentz flow generator",
                "supported_properties": ["qed", "sa_score", "mw", "logp"],
                "max_batch_size": _MAX_BATCH_SIZE,
                "supports_streaming": True,
                "requires_gpu": True,
            },
        )


async def serve():
    _require_runtime()
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=6))
    generator_pb2_grpc.add_GeneratorServiceServicer_to_server(HFMGeneratorServicer(), server)
    server.add_insecure_port("[::]:50066")
    await server.start()
    print("HFM Generator Service running on :50066")
    await server.wait_for_termination()


def _main(argv: list[str]) -> None:
    if not argv:
        asyncio.run(serve())
        return
    if len(argv) != 2 or argv[0] != "--bootstrap-validation-artifacts":
        raise ValueError(
            "usage: hfm_generator_svc.main "
            "--bootstrap-validation-artifacts <directory>"
        )
    from mf_generators.hfm_3d.generator import bootstrap_validation_artifacts

    paths = asyncio.run(bootstrap_validation_artifacts(argv[1]))
    sys.stdout.write(f"{paths['metadata'].parent}\n")


if __name__ == "__main__":
    _main(sys.argv[1:])
