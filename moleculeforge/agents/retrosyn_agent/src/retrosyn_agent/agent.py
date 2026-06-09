"""RetroSyn Agent - 3-layer retrosynthesis planning agent (Agent-3)."""
import asyncio
import inspect
import json
import os
import shlex
import struct
import subprocess
from typing import Any

from mf_agents.base.agent import BaseAgent, ensure_default_event_loop
from mf_agents.crg.graph import ChemicalReasoningGraph
from mf_core.artifacts import CommandRequirement, check_command, require_available
from mf_core.db.repositories import build_shared_crg_repository_from_env
from mf_core.proto_gen.moleculeforge.v1.humu import encoder_pb2, encoder_pb2_grpc

_PLANNER_COMMAND = CommandRequirement(
    "retrosyn_planner_command",
    "RETROSYN_PLANNER_COMMAND",
    required=False,
)
_NAMED_PLANNER_COMMAND_ENVS = (
    ("rascore", "RASCORE_PLANNER_COMMAND"),
    ("rsgpt", "RSGPT_PLANNER_COMMAND"),
    ("ualign", "UALIGN_PLANNER_COMMAND"),
    ("aizynth", "AIZYNTH_PLANNER_COMMAND"),
)
_NAMED_PLANNER_COMMAND_REQUIREMENTS = {
    engine: CommandRequirement(
        f"retrosyn_{engine}_planner_command",
        env_name,
        required=False,
    )
    for engine, env_name in _NAMED_PLANNER_COMMAND_ENVS
}


class HUMURouteEncoderGrpcClient:
    def __init__(self, target: str):
        import grpc

        ensure_default_event_loop()
        self.channel = grpc.aio.insecure_channel(target)
        self.stub = encoder_pb2_grpc.HUMUEncoderServiceStub(self.channel)

    async def encode_route(self, route: dict) -> dict:
        response = await self.stub.Encode(
            encoder_pb2.EncodeRequest(
                entity_type="route",
                input_data=json.dumps(_route_encoder_payload(route)).encode(),
            )
        )
        return {
            "humu_embedding": _float32_embedding_from_bytes(response.humu_embedding),
            "curvature": float(response.curvature),
        }


class ExternalCommandRetrosynPlanner:
    def __init__(
        self,
        command: str,
        engine: str = "external_command",
        command_requirement: CommandRequirement = _PLANNER_COMMAND,
    ):
        self.command = command
        self.engine = engine or "external_command"
        self.command_requirement = command_requirement
        self.timeout = float(os.getenv("RETROSYN_PLANNER_COMMAND_TIMEOUT_SECONDS", "300"))

    async def find_routes(self, smiles: str, max_routes: int = 10) -> list[dict]:
        payload = {
            "smiles": smiles,
            "max_routes": max_routes,
            "engine": self.engine,
        }
        result = await asyncio.to_thread(self._run, payload)
        routes = result.get("routes", result)
        if not isinstance(routes, list):
            raise RuntimeError("RETROSYN_PLANNER_COMMAND must return routes as a list")
        for route in routes:
            if not isinstance(route, dict):
                raise RuntimeError("RETROSYN_PLANNER_COMMAND routes must be JSON objects")
            route.setdefault("source_engine", self.engine)
        return routes[:max_routes]

    def _run(self, payload: dict) -> dict | list:
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
            raise RuntimeError(f"RETROSYN_PLANNER_COMMAND failed: {stderr}")
        try:
            parsed = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("RETROSYN_PLANNER_COMMAND returned invalid JSON") from exc
        if not isinstance(parsed, dict | list):
            raise RuntimeError("RETROSYN_PLANNER_COMMAND must return a JSON object or list")
        return parsed


class RetroSynAgent(BaseAgent):
    def __init__(
        self,
        message_bus=None,
        planner=None,
        route_planners: dict[str, Any] | None = None,
        route_encoder_client=None,
        route_encoder_target: str | None = None,
        crg_repository: Any = None,
    ):
        super().__init__("retrosyn_agent", message_bus)
        self._subscription_subjects = ["agent.retrosyn.request", "orchestrator.retrosyn.plan"]
        self.crg = ChemicalReasoningGraph()
        self.planner = planner
        self.route_planners = route_planners or _route_planners_from_env()
        self.planner_command = os.getenv("RETROSYN_PLANNER_COMMAND", "").strip()
        self.route_encoder_client = route_encoder_client or _build_route_encoder_client(
            route_encoder_target
        )
        self.crg_repository = (
            crg_repository
            if crg_repository is not None
            else build_shared_crg_repository_from_env()
        )

    async def handle_message(self, subject, payload, reply_to=""):
        data = json.loads(payload) if isinstance(payload, bytes) else {"raw": payload}
        result = await self.process(data)
        if reply_to:
            await self.publish(reply_to, json.dumps(result).encode())

    async def process(self, data):
        """Plan 3-layer retrosynthesis: strategy -> pathway -> reaction.

        Layer 1: Strategic disconnections and route planning
        Layer 2: Pathway enumeration and ranking
        Layer 3: Individual reaction feasibility and condition selection
        """
        target_smiles = data.get("smiles", "")
        if not isinstance(target_smiles, str) or not target_smiles:
            raise ValueError("smiles is required for retrosynthesis planning")
        max_routes = int(data.get("max_routes", 10) or 10)
        run_id = str(data.get("run_id") or data.get("request_id") or "")
        if await self._has_failed_validation_belief(target_smiles, run_id):
            belief = self.crg.add_belief(
                subject=target_smiles,
                predicate="retrosyn_routes",
                obj="0",
                confidence=1.0,
                source_agent=self.name,
                evidence_ids=["crg_validation_status"],
            )
            await self._persist_belief(
                belief,
                project_id=str(data.get("project_id") or ""),
                run_id=run_id,
            )
            return {
                "agent": self.name,
                "status": "skipped",
                "target_smiles": target_smiles,
                "routes": [],
                "skip_reason": "shared CRG contains failed validation_status",
                "layers": {
                    "strategy": {
                        "route_count": 0,
                        "engine": "shared_crg",
                    },
                    "pathways": [],
                    "reactions": [],
                },
            }
        if await self._has_zero_routes_belief(target_smiles, run_id):
            return {
                "agent": self.name,
                "status": "skipped",
                "target_smiles": target_smiles,
                "routes": [],
                "cache_source": "shared_crg",
                "skip_reason": "shared CRG contains zero retrosyn_routes",
                "layers": {
                    "strategy": {
                        "route_count": 0,
                        "engine": "shared_crg",
                    },
                    "pathways": [],
                    "reactions": [],
                },
            }
        routes, strategy = await self._find_routes(target_smiles, max_routes)
        routes = await self._attach_route_embeddings(routes)
        belief = self.crg.add_belief(
            subject=target_smiles,
            predicate="retrosyn_routes",
            obj=str(len(routes)),
            confidence=1.0,
            source_agent=self.name,
            evidence_ids=[
                str(route["route_id"])
                for route in routes
                if isinstance(route, dict) and route.get("route_id")
            ],
        )
        beliefs_to_persist = [belief]
        for route in routes:
            if not isinstance(route, dict) or "humu_embedding" not in route:
                continue
            route_id = str(route.get("route_id") or "")
            route_belief = self.crg.add_belief(
                subject=target_smiles,
                predicate="route_humu_embedding",
                obj=json.dumps(
                    {
                        "curvature": float(route.get("humu_curvature", 1.0)),
                        "humu_embedding": route["humu_embedding"],
                        "route_id": route_id,
                    },
                    sort_keys=True,
                ),
                confidence=1.0,
                source_agent=self.name,
                evidence_ids=[route_id] if route_id else [],
            )
            beliefs_to_persist.append(route_belief)
        project_id = str(data.get("project_id") or "")
        for belief in beliefs_to_persist:
            await self._persist_belief(
                belief,
                project_id=project_id,
                run_id=run_id,
            )
        return {
            "agent": self.name,
            "status": "planned",
            "target_smiles": target_smiles,
            "routes": routes,
            "layers": {
                "strategy": {
                    **strategy,
                    "route_count": len(routes),
                },
                "pathways": routes,
                "reactions": _route_reactions(routes),
            },
        }

    async def _find_routes(
        self,
        target_smiles: str,
        max_routes: int,
    ) -> tuple[list[dict], dict]:
        if self.route_planners:
            routes: list[dict] = []
            for engine, planner in self.route_planners.items():
                engine_routes = await _find_routes_with_planner(
                    planner,
                    target_smiles,
                    max_routes,
                )
                for route in engine_routes:
                    route_with_engine = dict(route)
                    route_with_engine.setdefault("source_engine", engine)
                    routes.append(route_with_engine)
            ranked = _rank_routes(_dedupe_routes(routes))[:max_routes]
            return ranked, {
                "engine": "ensemble",
                "engines": list(self.route_planners.keys()),
            }
        planner = self._planner()
        return await _find_routes_with_planner(
            planner,
            target_smiles,
            max_routes,
        ), {"engine": planner.__class__.__name__}

    def _planner(self):
        if self.planner is None:
            if self.planner_command:
                self.planner = ExternalCommandRetrosynPlanner(self.planner_command)
                return self.planner
            from mf_retrosyn.aizynth.retrosyn import AiZynthRetrosyn

            self.planner = AiZynthRetrosyn.from_env()
        return self.planner

    async def _attach_route_embeddings(self, routes: list[dict]) -> list[dict]:
        if self.route_encoder_client is None:
            return routes
        encoded_routes = []
        for route in routes:
            if not isinstance(route, dict):
                encoded_routes.append(route)
                continue
            embedding_record = await _encode_route(self.route_encoder_client, route)
            route_with_embedding = dict(route)
            route_with_embedding["humu_embedding"] = embedding_record["humu_embedding"]
            route_with_embedding["humu_curvature"] = embedding_record["curvature"]
            encoded_routes.append(route_with_embedding)
        return encoded_routes

    async def _has_failed_validation_belief(self, target_smiles: str, run_id: str) -> bool:
        if not run_id or self.crg_repository is None:
            return False
        if not callable(getattr(self.crg_repository, "get_run_crg", None)):
            return False
        crg = await self.read_shared_crg(run_id)
        for belief in crg.get("beliefs", []):
            if not isinstance(belief, dict):
                continue
            if str(belief.get("subject") or "") != target_smiles:
                continue
            predicate = str(belief.get("predicate") or "")
            object_value = str(
                belief.get("object_value", belief.get("object", ""))
            )
            if predicate == "validation_status" and object_value == "failed":
                return True
        return False

    async def _has_zero_routes_belief(self, target_smiles: str, run_id: str) -> bool:
        if not run_id or self.crg_repository is None:
            return False
        if not callable(getattr(self.crg_repository, "get_run_crg", None)):
            return False
        crg = await self.read_shared_crg(run_id)
        for belief in crg.get("beliefs", []):
            if not isinstance(belief, dict):
                continue
            if str(belief.get("subject") or "") != target_smiles:
                continue
            predicate = str(belief.get("predicate") or "")
            object_value = str(
                belief.get("object_value", belief.get("object", ""))
            )
            if predicate == "retrosyn_routes" and object_value == "0":
                return True
        return False

    async def _persist_belief(self, belief, project_id: str, run_id: str) -> None:
        if self.crg_repository is None:
            return
        write_belief = getattr(self.crg_repository, "write_workflow_belief", None)
        if not callable(write_belief):
            raise TypeError("crg_repository must expose write_workflow_belief(**kwargs)")
        result = write_belief(
            project_id=project_id,
            run_id=run_id or belief.subject,
            belief_id=belief.id,
            subject=belief.subject,
            predicate=belief.predicate,
            object_value=belief.object,
            confidence=belief.confidence,
            source_agent=belief.source_agent,
            timestamp_ns=belief.timestamp_ns,
            evidence_ids=list(belief.evidence_ids),
        )
        if inspect.isawaitable(result):
            await result


def _route_reactions(routes: list[dict]) -> list[str]:
    reactions: list[str] = []
    for route in routes:
        for step in route.get("steps") or []:
            reaction = step.get("reaction") if isinstance(step, dict) else None
            if isinstance(reaction, str) and reaction:
                reactions.append(reaction)
    return reactions


def _route_planners_from_env() -> dict[str, ExternalCommandRetrosynPlanner]:
    raw = os.getenv("RETROSYN_PLANNER_COMMANDS_JSON", "").strip()
    if not raw:
        return _named_route_planners_from_env()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("RETROSYN_PLANNER_COMMANDS_JSON must be a JSON object")
    planners = {}
    for engine, command in payload.items():
        engine_name = str(engine).strip()
        command_value = str(command).strip()
        if not engine_name:
            raise ValueError("RETROSYN_PLANNER_COMMANDS_JSON engine names must be non-empty")
        if not command_value:
            raise ValueError("RETROSYN_PLANNER_COMMANDS_JSON commands must be non-empty")
        planners[engine_name] = ExternalCommandRetrosynPlanner(
            command_value,
            engine=engine_name,
            command_requirement=CommandRequirement(
                f"retrosyn_{engine_name}_planner_command",
                "RETROSYN_PLANNER_COMMANDS_JSON",
                required=False,
            ),
        )
    return planners


def _named_route_planners_from_env() -> dict[str, ExternalCommandRetrosynPlanner]:
    planners = {}
    for engine_name, env_name in _NAMED_PLANNER_COMMAND_ENVS:
        command_value = os.getenv(env_name, "").strip()
        if not command_value:
            continue
        planners[engine_name] = ExternalCommandRetrosynPlanner(
            command_value,
            engine=engine_name,
            command_requirement=_NAMED_PLANNER_COMMAND_REQUIREMENTS[engine_name],
        )
    return planners


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


async def _find_routes_with_planner(
    planner: Any,
    target_smiles: str,
    max_routes: int,
) -> list[dict]:
    result = planner.find_routes(target_smiles, max_routes=max_routes)
    if inspect.isawaitable(result):
        result = await result
    if not isinstance(result, list):
        raise TypeError("retrosynthesis planner must return a list of route dicts")
    for route in result:
        if not isinstance(route, dict):
            raise TypeError("retrosynthesis planner routes must be dictionaries")
    return result


def _dedupe_routes(routes: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique: list[dict] = []
    for route in routes:
        key = _route_key(route)
        if key in seen:
            continue
        seen.add(key)
        unique.append(route)
    return unique


def _route_key(route: dict) -> str:
    route_id = route.get("route_id")
    if isinstance(route_id, str) and route_id:
        return f"id:{route_id}"
    reactions = _route_reactions([route])
    if reactions:
        return "reactions:" + "|".join(reactions)
    return json.dumps(route, sort_keys=True, default=str)


def _rank_routes(routes: list[dict]) -> list[dict]:
    indexed = list(enumerate(routes))
    indexed.sort(key=lambda item: (_route_priority(item[1]), -_route_score(item[1]), item[0]))
    return [route for _, route in indexed]


def _route_priority(route: dict) -> int:
    if str(route.get("route_type") or "") == "retrosynthetic_accessibility_score":
        return 1
    return 0


def _route_score(route: dict) -> float:
    for key in (
        "score",
        "predicted_score",
        "route_score",
        "estimated_yield",
        "predicted_yield",
    ):
        value = route.get(key)
        if isinstance(value, int | float):
            return float(value)
    return 0.0


def _build_route_encoder_client(route_encoder_target: str | None):
    target = route_encoder_target or os.environ.get("HUMU_ENCODER_TARGET", "")
    return HUMURouteEncoderGrpcClient(target) if target else None


async def _encode_route(client: Any, route: dict) -> dict:
    if hasattr(client, "encode_route"):
        result = client.encode_route(route)
    elif callable(client):
        result = client(route)
    else:
        raise TypeError("route encoder client must expose encode_route(route) or be callable")
    if inspect.isawaitable(result):
        result = await result
    return _normalise_route_embedding(result)


def _normalise_route_embedding(result: object) -> dict:
    if isinstance(result, dict):
        embedding = result.get("humu_embedding", result.get("embedding"))
        curvature = float(result.get("curvature", 1.0))
    else:
        embedding = result
        curvature = 1.0
    if isinstance(embedding, bytes):
        embedding = _float32_embedding_from_bytes(embedding)
    if not isinstance(embedding, list) or not embedding:
        raise ValueError("route encoder result requires a non-empty humu_embedding")
    return {
        "humu_embedding": [float(value) for value in embedding],
        "curvature": curvature,
    }


def _route_encoder_payload(route: dict) -> dict:
    payload = dict(route)
    reactions = payload.get("reactions")
    if not isinstance(reactions, list) or not reactions:
        payload["reactions"] = _route_reactions([route])
    if not payload["reactions"]:
        raise ValueError("route encoder requires at least one reaction")
    return payload


def _float32_embedding_from_bytes(payload: bytes) -> list[float]:
    if len(payload) % 4 != 0:
        raise ValueError("route HUMU embedding bytes must contain float32 values")
    return [float(item[0]) for item in struct.iter_unpack("<f", payload)]
