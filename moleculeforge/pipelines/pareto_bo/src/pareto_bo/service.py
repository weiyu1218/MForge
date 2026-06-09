"""Service wrapper for Pareto constrained Bayesian optimization."""
from __future__ import annotations

import asyncio
import importlib
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Callable, Mapping
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from mf_core.artifacts import CommandRequirement, check_command, require_available
from mf_eval.hv_evaluator import PCBOOptimizationScheduler

_CANDIDATE_PROVIDER_ENV = "PARETO_BO_CANDIDATE_PROVIDER"
_CANDIDATE_PROVIDER_COMMAND_ENV = "PARETO_BO_CANDIDATE_PROVIDER_COMMAND"
_ORACLE_EVALUATE_ENV = "PARETO_BO_ORACLE_EVALUATE"
_ORACLE_EVALUATE_COMMAND_ENV = "PARETO_BO_ORACLE_EVALUATE_COMMAND"
_COMMAND_TIMEOUT_ENV = "PARETO_BO_COMMAND_TIMEOUT_SECONDS"

rest_app = FastAPI(title="ParetoBO Service", version="0.1.0")


@rest_app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy", "service": "pareto_bo"}


@rest_app.post("/v1/pareto-bo/optimize")
async def optimize_endpoint(request: dict) -> dict[str, Any]:
    try:
        return await ParetoBOService.from_env().optimize(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


class ParetoBOService:
    def __init__(
        self,
        *,
        candidate_provider: Callable[[dict], object] | object,
        oracle_evaluate: Callable[[dict], object],
    ) -> None:
        if candidate_provider is None:
            raise ValueError("candidate_provider is required")
        if oracle_evaluate is None:
            raise ValueError("oracle_evaluate is required")
        self.candidate_provider = candidate_provider
        self.oracle_evaluate = oracle_evaluate

    @classmethod
    def from_env(cls) -> ParetoBOService:
        return cls(
            candidate_provider=_candidate_provider_from_env(),
            oracle_evaluate=_oracle_evaluate_from_env(),
        )

    async def optimize(self, request: Mapping[str, object]) -> dict[str, Any]:
        scheduler = PCBOOptimizationScheduler(
            candidate_provider=self.candidate_provider,
            oracle_evaluate=self.oracle_evaluate,
            reference=_required(request, "reference"),
            lower_bounds=_required(request, "lower_bounds"),
            upper_bounds=_required(request, "upper_bounds"),
            batch_size=int(_required(request, "batch_size")),
            n_rounds=int(_required(request, "n_rounds")),
            lengthscale=float(request.get("lengthscale", 1.0)),
            noise=float(request.get("noise", 1e-4)),
            maximize=_bool_request(request.get("maximize", True)),
        )
        result = await scheduler.run(
            observed_embeddings=_required(request, "observed_embeddings"),
            observed_objectives=_required(request, "observed_objectives"),
            observed_constraints=_required(request, "observed_constraints"),
        )
        return _json_ready(result)


def _required(request: Mapping[str, object], name: str) -> object:
    if name not in request:
        raise ValueError(f"ParetoBO optimize request requires {name}")
    return request[name]


def _bool_request(value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError("maximize must be a boolean")
    return value


def _json_ready(value: object) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    return value


class _CommandCandidateProvider:
    def __init__(self, command: str) -> None:
        self.command = command

    def propose(self, state: dict) -> object:
        response = _run_json_command(
            self.command,
            state,
            source_env=_CANDIDATE_PROVIDER_COMMAND_ENV,
        )
        if isinstance(response, Mapping):
            if "candidate_embeddings" not in response:
                raise RuntimeError(
                    f"{_CANDIDATE_PROVIDER_COMMAND_ENV} response requires candidate_embeddings"
                )
            return response["candidate_embeddings"]
        if isinstance(response, list):
            return response
        raise RuntimeError(
            f"{_CANDIDATE_PROVIDER_COMMAND_ENV} must return a JSON object or list"
        )


class _CommandOracleEvaluator:
    def __init__(self, command: str) -> None:
        self.command = command

    def __call__(self, request: dict) -> dict:
        response = _run_json_command(
            self.command,
            request,
            source_env=_ORACLE_EVALUATE_COMMAND_ENV,
        )
        if not isinstance(response, Mapping):
            raise RuntimeError(f"{_ORACLE_EVALUATE_COMMAND_ENV} must return a JSON object")
        return dict(response)


def _candidate_provider_from_env() -> Callable[[dict], object] | object:
    return _runtime_from_env(
        path_env=_CANDIDATE_PROVIDER_ENV,
        command_env=_CANDIDATE_PROVIDER_COMMAND_ENV,
        command_factory=_CommandCandidateProvider,
        default_factory=_default_candidate_provider,
    )


def _oracle_evaluate_from_env() -> Callable[[dict], object] | object:
    return _runtime_from_env(
        path_env=_ORACLE_EVALUATE_ENV,
        command_env=_ORACLE_EVALUATE_COMMAND_ENV,
        command_factory=_CommandOracleEvaluator,
        default_factory=_default_oracle_evaluator,
    )


def _runtime_from_env(
    *,
    path_env: str,
    command_env: str,
    command_factory: Callable[[str], object],
    default_factory: Callable[[], object] | None = None,
) -> Callable[[dict], object] | object:
    path = os.environ.get(path_env, "").strip()
    command = os.environ.get(command_env, "").strip()
    if path and command:
        raise RuntimeError(f"{path_env} and {command_env} are mutually exclusive")
    if path:
        return _load_callable(path)
    if command:
        return command_factory(command)
    if default_factory is not None:
        return default_factory()
    raise RuntimeError(f"{path_env} or {command_env} is required")


def _default_candidate_provider() -> object:
    from pareto_bo.providers import TangentSpaceNoiseCandidateProvider

    return TangentSpaceNoiseCandidateProvider.from_env()


def _default_oracle_evaluator() -> object:
    from pareto_bo.providers import LocalOracleEvaluator

    return LocalOracleEvaluator.from_env()


def _run_json_command(command: str, payload: object, *, source_env: str) -> object:
    _require_command_available(command, source_env)
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        raise RuntimeError(f"{source_env} is not a valid shell command") from exc
    if not argv:
        raise RuntimeError(f"{source_env} is empty")
    completed = subprocess.run(
        argv,
        input=json.dumps(_json_ready(payload)),
        capture_output=True,
        check=False,
        text=True,
        timeout=float(os.environ.get(_COMMAND_TIMEOUT_ENV, "300")),
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(f"{source_env} failed: {stderr}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{source_env} returned invalid JSON") from exc


def _require_command_available(command: str, source_env: str) -> None:
    env = {**os.environ, source_env: command}
    require_available(
        [
            check_command(
                CommandRequirement(_command_requirement_name(source_env), source_env),
                env=env,
            )
        ]
    )


def _command_requirement_name(source_env: str) -> str:
    return source_env.lower()


def _load_callable(path: str) -> Callable[[dict], object] | object:
    if ":" not in path:
        raise ValueError("callable path must use module:attribute format")
    module_name, attribute_name = path.split(":", 1)
    if not module_name or not attribute_name:
        raise ValueError("callable path must use module:attribute format")
    module = importlib.import_module(module_name)
    target = module
    for part in attribute_name.split("."):
        target = getattr(target, part)
    return target


async def _main_async() -> None:
    request = json.load(sys.stdin)
    if not isinstance(request, Mapping):
        raise ValueError("ParetoBO CLI request must be a JSON object")
    service = ParetoBOService.from_env()
    result = await service.optimize(request)
    sys.stdout.write(json.dumps(result, sort_keys=True))
    sys.stdout.write("\n")


def main() -> None:
    asyncio.run(_main_async())
