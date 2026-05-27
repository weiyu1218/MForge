"""ADMET Prediction Service.

gRPC server for ADMET property prediction and L0 drug-likeness filtering.
"""
import asyncio
import time
from concurrent import futures
from types import SimpleNamespace

import grpc
from mf_core.artifacts import (
    ArtifactRequirement,
    PythonPackageRequirement,
    RequirementStatus,
    check_artifact,
    check_python_package,
    require_available,
)

_REQUIREMENTS = (ArtifactRequirement("admet_model", "ADMET_MODEL_PATH", kind="directory"),)
_PACKAGES = (PythonPackageRequirement("rdkit", module="rdkit"),)


def _require_runtime() -> list[RequirementStatus]:
    statuses = [
        *(check_artifact(requirement) for requirement in _REQUIREMENTS),
        *(check_python_package(requirement) for requirement in _PACKAGES),
    ]
    require_available(statuses)
    return statuses


def runtime_status() -> list[dict]:
    return [status.to_dict() for status in _status_objects()]


async def _abort_unavailable(context):
    statuses = _status_objects()
    try:
        require_available(statuses)
    except RuntimeError as exc:
        message = str(exc)
    else:
        message = "ADMET model runner is not configured"
    if context is not None and hasattr(context, "abort"):
        await context.abort(grpc.StatusCode.FAILED_PRECONDITION, message)
    raise RuntimeError(message)


def _status_objects() -> list[RequirementStatus]:
    return [
        *(check_artifact(requirement) for requirement in _REQUIREMENTS),
        *(check_python_package(requirement) for requirement in _PACKAGES),
    ]


class ADMETServicer:
    def __init__(self, runner=None):
        self.runner = runner

    def _runner(self):
        if self.runner is not None:
            return self.runner
        from mf_oracles.admet_ai.oracle import ADMETHTTPRunner

        self.runner = ADMETHTTPRunner.from_env()
        return self.runner

    async def Predict(self, request, context):
        """Predict ADMET properties for a molecule."""
        try:
            _require_runtime()
        except RuntimeError:
            return await _abort_unavailable(context)
        smiles = _request_smiles(request)
        properties = _request_properties(request)
        start = time.perf_counter()
        from mf_oracles.admet_ai.oracle import ADMETAIOracle

        result = await ADMETAIOracle(runner=self._runner()).evaluate([smiles], properties)
        predictions = result[smiles]
        return SimpleNamespace(
            smiles=smiles,
            predictions=predictions,
            properties=predictions,
            elapsed_ms=int((time.perf_counter() - start) * 1000),
        )

    async def BatchPredict(self, request, context):
        """Batch ADMET prediction."""
        results = []
        for req in getattr(request, "requests", []):
            results.append(await self.Predict(req, context))
        return type(
            "BatchADMETResponse",
            (),
            {"results": results, "total_elapsed_ms": 200},
        )()

    async def Screen(self, request, context):
        """Quick L0 screen: returns pass/fail with filter reasons."""
        try:
            _require_runtime()
        except RuntimeError:
            return await _abort_unavailable(context)
        smiles = _request_smiles(request)
        properties = _request_properties(request)
        from mf_oracles.admet_ai.oracle import ADMETAIOracle

        result = await ADMETAIOracle(runner=self._runner()).predict_with_uncertainty(
            [smiles],
            properties,
        )
        return SimpleNamespace(smiles=smiles, result=result)


def _request_smiles(request) -> str:
    smiles = (
        getattr(request, "smiles", None)
        or getattr(request, "molecule_smiles", None)
        or getattr(request, "canonical_smiles", None)
    )
    if not smiles:
        raise ValueError("request.smiles is required")
    return str(smiles)


def _request_properties(request) -> list[str]:
    properties = (
        getattr(request, "properties", None)
        or getattr(request, "requested_properties", None)
        or []
    )
    if isinstance(properties, str):
        properties = [properties]
    properties = [str(item) for item in properties if str(item)]
    if properties:
        return properties
    import os

    return [item.strip() for item in os.environ.get("ADMET_TARGETS", "").split(",") if item.strip()]


async def serve():
    _require_runtime()
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=16))
    server.add_insecure_port("[::]:50056")
    await server.start()
    print("ADMET Prediction Service running on :50056")
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
