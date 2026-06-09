"""CReM-pharm-3D Molecule Generator Service - gRPC server."""
import asyncio
import json
import os
import shlex
import subprocess
import time
from concurrent import futures

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

_REQUIREMENTS = (ArtifactRequirement("crem_mmp_database", "CREM_MMP_DB_PATH"),)
_SCORER_COMMAND_REQUIREMENTS = (
    CommandRequirement(
        "crem_pharmacophore_scorer_command",
        "CREM_PHARMACOPHORE_SCORER_COMMAND",
    ),
    CommandRequirement("crem_humu_scorer_command", "CREM_HUMU_SCORER_COMMAND"),
)
_SCORER_COMMAND_REQUIREMENTS_BY_ENV = {
    requirement.env_var: requirement
    for requirement in _SCORER_COMMAND_REQUIREMENTS
}
_SCORER_COMMAND_REQUIREMENTS_BY_SOURCE = {
    "pharmacophore": _SCORER_COMMAND_REQUIREMENTS_BY_ENV[
        "CREM_PHARMACOPHORE_SCORER_COMMAND"
    ],
    "humu_embedding": _SCORER_COMMAND_REQUIREMENTS_BY_ENV["CREM_HUMU_SCORER_COMMAND"],
}
_GENERATOR_NAME = "crem_3d"
_SCORER_TIMEOUT_ENV = "CREM_SCORER_COMMAND_TIMEOUT_SECONDS"


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
    return statuses


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
        self.command_requirement = command_requirement or _SCORER_COMMAND_REQUIREMENTS_BY_SOURCE[
            source
        ]
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
            _require_runtime()
        except RuntimeError:
            return await _abort_unavailable(context)
        if self.generator is None:
            return await _abort_unavailable(context)
        try:
            batch_size = _batch_size(request)
        except ValueError as exc:
            return await _abort_invalid_argument(context, str(exc))
        params = dict(getattr(request, "generator_params", {}) or {})
        start = time.perf_counter()
        molecules = await self.generator.generate(
            batch_size=batch_size,
            intent_cone=_intent_cone_from_request(request),
            **params,
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

    async def GenerateStream(self, request_iterator, context):
        async for request in request_iterator:
            yield await self.Generate(request, context)

    async def BatchGenerate(self, request_iterator, context):
        async for request in request_iterator:
            yield await self.Generate(request, context)

    async def Info(self, request, context):
        return generator_pb2.GeneratorInfo(
            name=_GENERATOR_NAME,
            version="0.1.0",
            description="CReM-pharm-3D fragment replacement generator",
            supported_properties=["qed", "sa_score", "docking_score"],
            max_batch_size=256,
            supports_streaming=True,
            requires_gpu=False,
        )


async def serve():
    _require_runtime()
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=8))
    generator_pb2_grpc.add_GeneratorServiceServicer_to_server(CReMGeneratorServicer(), server)
    server.add_insecure_port("[::]:50062")
    await server.start()
    print("CReM Generator Service running on :50062")
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
