"""CReM-pharm-3D Molecule Generator Service - gRPC server."""

import asyncio
import json
import os
import shlex
import subprocess
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
from mf_core.types.humu import IntentCone

_REQUIREMENTS = (ArtifactRequirement("crem_mmp_database", "CREM_MMP_DB_PATH"),)
_SCORER_COMMAND_REQUIREMENTS = (
    CommandRequirement(
        "crem_pharmacophore_scorer_command",
        "CREM_PHARMACOPHORE_SCORER_COMMAND",
    ),
    CommandRequirement("crem_humu_scorer_command", "CREM_HUMU_SCORER_COMMAND"),
)
_SCORER_COMMAND_REQUIREMENTS_BY_ENV = {
    requirement.env_var: requirement for requirement in _SCORER_COMMAND_REQUIREMENTS
}
_SCORER_COMMAND_REQUIREMENTS_BY_SOURCE = {
    "pharmacophore": _SCORER_COMMAND_REQUIREMENTS_BY_ENV["CREM_PHARMACOPHORE_SCORER_COMMAND"],
    "humu_embedding": _SCORER_COMMAND_REQUIREMENTS_BY_ENV["CREM_HUMU_SCORER_COMMAND"],
}
_GENERATOR_NAME = "crem_3d"
_MAX_BATCH_SIZE = 256
_SCORER_TIMEOUT_ENV = "CREM_SCORER_COMMAND_TIMEOUT_SECONDS"
_ALLOW_VALIDATION_ARTIFACT_ENV = "CREM_ALLOW_VALIDATION_ARTIFACT"


def _require_runtime() -> list[RequirementStatus]:
    statuses = _runtime_statuses()
    require_available(statuses)
    return statuses


def runtime_status() -> list[dict]:
    return [status.to_dict() for status in _runtime_statuses()]


def _runtime_statuses() -> list[RequirementStatus]:
    statuses = [check_artifact(requirement) for requirement in _REQUIREMENTS]
    for requirement in _SCORER_COMMAND_REQUIREMENTS:
        if os.environ.get(requirement.env_var, "").strip():
            statuses.append(check_command(requirement))
    validation_status = _validation_artifact_opt_in_status()
    if validation_status is not None:
        statuses.append(validation_status)
    return statuses


def _validation_artifact_opt_in_status() -> RequirementStatus | None:
    database_path = os.environ.get("CREM_MMP_DB_PATH", "").strip()
    if not database_path:
        return None
    try:
        from mf_generators.crem_3d.generator import (
            load_validation_artifact_metadata,
        )

        metadata = load_validation_artifact_metadata(database_path)
    except Exception as exc:
        return RequirementStatus(
            name="crem_validation_artifact_opt_in",
            configured=False,
            available=False,
            required=True,
            path=database_path,
            source=_ALLOW_VALIDATION_ARTIFACT_ENV,
            message=f"CReM validation artifact metadata is invalid: {exc}",
        )
    if metadata is None:
        return None
    opted_in = os.environ.get(_ALLOW_VALIDATION_ARTIFACT_ENV, "").strip() == "true"
    return RequirementStatus(
        name="crem_validation_artifact_opt_in",
        configured=opted_in,
        available=opted_in,
        required=True,
        path=database_path,
        source=_ALLOW_VALIDATION_ARTIFACT_ENV,
        message=(
            "CReM validation artifact is explicitly enabled"
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
        message = "CReM generator runner is not configured"
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
    from mf_generators.crem_3d.generator import CReM3DGenerator, DockOracleGrpcScorer

    return CReM3DGenerator(
        mmp_db_path=os.environ["CREM_MMP_DB_PATH"],
        docking_scorer=_dock_scorer_from_env(DockOracleGrpcScorer),
        pharmacophore_scorer=_json_score_provider_from_env(
            "CREM_PHARMACOPHORE_SCORER_COMMAND",
            source="pharmacophore",
        ),
        humu_embedding_scorer=_json_score_provider_from_env(
            "CREM_HUMU_SCORER_COMMAND",
            source="humu_embedding",
        ),
    )


def _dock_scorer_from_env(scorer_cls):
    target = os.getenv("CREM_DOCK_ORACLE_TARGET", "").strip()
    if not target:
        return None
    return scorer_cls(target=target)


def _json_score_provider_from_env(env_name: str, *, source: str):
    command = os.getenv(env_name, "").strip()
    if not command:
        return None
    return ExternalJSONScoreProvider(
        command=command,
        source=source,
        command_requirement=_SCORER_COMMAND_REQUIREMENTS_BY_ENV[env_name],
    )


class ExternalJSONScoreProvider:
    def __init__(
        self,
        command: str,
        source: str,
        command_requirement: CommandRequirement | None = None,
    ) -> None:
        self.command = command
        self.source = source
        self.command_requirement = (
            command_requirement or _SCORER_COMMAND_REQUIREMENTS_BY_SOURCE[source]
        )
        self.timeout = float(os.getenv(_SCORER_TIMEOUT_ENV, "120"))

    async def score_batch(
        self,
        smiles_list: list[str],
        *,
        intent_cone: IntentCone | None = None,
    ) -> dict[str, dict]:
        if not smiles_list:
            raise ValueError("CReM scorer smiles_list must not be empty")
        return await asyncio.to_thread(
            self._score_batch_sync,
            list(smiles_list),
            intent_cone,
        )

    async def score(
        self,
        smiles: str,
        *,
        intent_cone: IntentCone | None = None,
    ) -> dict:
        records = await self.score_batch([smiles], intent_cone=intent_cone)
        return records[smiles]

    def _score_batch_sync(
        self,
        smiles_list: list[str],
        intent_cone: IntentCone | None,
    ) -> dict[str, dict]:
        payload: dict[str, object] = {
            "smiles": smiles_list,
            "source": self.source,
        }
        if intent_cone is not None:
            payload["intent_cone"] = intent_cone.model_dump(mode="json")
        _require_command_available(self.command_requirement, self.command)
        completed = subprocess.run(
            shlex.split(self.command),
            input=json.dumps(payload, sort_keys=True),
            capture_output=True,
            text=True,
            timeout=self.timeout,
            check=False,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            raise RuntimeError(f"CReM scorer command failed: {stderr}")
        try:
            response = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("CReM scorer command returned invalid JSON") from exc
        return _score_records_from_response(response, smiles_list)


def _score_records_from_response(response: object, smiles_list: list[str]) -> dict[str, dict]:
    if not isinstance(response, dict):
        raise RuntimeError("CReM scorer command must return a JSON object")
    raw_records = response.get("records", response)
    if isinstance(raw_records, list):
        records = _score_record_list_to_mapping(raw_records)
    elif isinstance(raw_records, dict):
        records = {
            str(smiles): dict(record)
            for smiles, record in raw_records.items()
            if isinstance(record, dict)
        }
    else:
        raise RuntimeError("CReM scorer command records must be an object or list")
    for smiles in smiles_list:
        if smiles not in records:
            raise RuntimeError(f"CReM scorer command returned no record for {smiles}")
    return {smiles: records[smiles] for smiles in smiles_list}


def _score_record_list_to_mapping(raw_records: list[object]) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for item in raw_records:
        if not isinstance(item, dict):
            raise RuntimeError("CReM scorer command record rows must be JSON objects")
        smiles = item.get("smiles")
        if not isinstance(smiles, str) or not smiles:
            raise RuntimeError("CReM scorer command record rows require smiles")
        record = dict(item)
        records[smiles] = record
    return records


def _require_command_available(
    requirement: CommandRequirement,
    command: str,
) -> None:
    required_requirement = CommandRequirement(
        requirement.name,
        requirement.env_var,
        required=True,
    )
    env = {**os.environ, requirement.env_var: command}
    require_available([check_command(required_requirement, env=env)])


class CReMGeneratorServicer:
    def __init__(self, generator=None):
        self.generator = generator if generator is not None else _build_generator()

    async def Generate(self, request, context):
        """Generate molecules via CReM-pharm-3D fragment replacement."""
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
                "description": "CReM-pharm-3D fragment replacement generator",
                "supported_properties": ["qed", "sa_score", "docking_score"],
                "max_batch_size": _MAX_BATCH_SIZE,
                "supports_streaming": True,
                "requires_gpu": False,
            },
        )


async def serve():
    _require_runtime()
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=8))
    generator_pb2_grpc.add_GeneratorServiceServicer_to_server(CReMGeneratorServicer(), server)
    server.add_insecure_port("[::]:50062")
    await server.start()
    print("CReM Generator Service running on :50062")
    await server.wait_for_termination()


def _main(argv: list[str]) -> None:
    if not argv:
        asyncio.run(serve())
        return
    if len(argv) != 2 or argv[0] != "--bootstrap-validation-artifacts":
        raise ValueError(
            "usage: crem_generator_svc.main "
            "--bootstrap-validation-artifacts <directory>"
        )
    from mf_generators.crem_3d.generator import bootstrap_validation_artifacts

    paths = asyncio.run(bootstrap_validation_artifacts(argv[1]))
    sys.stdout.write(f"{paths['metadata'].parent}\n")


if __name__ == "__main__":
    _main(sys.argv[1:])
