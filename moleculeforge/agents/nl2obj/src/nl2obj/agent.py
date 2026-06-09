"""NL2Obj Agent - Natural language to objective specification agent (Agent-1)."""
import inspect
import json
import os
from typing import Any

from mf_agents.base.agent import BaseAgent, ensure_default_event_loop
from mf_agents.crg.graph import ChemicalReasoningGraph
from mf_core.db.repositories import build_shared_crg_repository_from_env
from mf_core.proto_gen.moleculeforge.v1.core import cig_pb2, cig_pb2_grpc

from nl2obj.parser import parse as parse_intent


class CIGCompilerGrpcClient:
    def __init__(self, target: str):
        import grpc

        ensure_default_event_loop()
        self.channel = grpc.aio.insecure_channel(target)
        self.stub = cig_pb2_grpc.CIGCompilerServiceStub(self.channel)

    async def compile_intent(self, request: dict) -> dict:
        payload = {
            "project_id": str(request.get("project_id", "")),
            "nl_query": str(request["nl_query"]),
        }
        if request.get("seed") is not None:
            payload["seed"] = int(request["seed"])
        response = await self.stub.Compile(cig_pb2.CIGCompileRequest(**payload))
        return _compiled_intent_from_proto(response)


class NL2ObjAgent(BaseAgent):
    def __init__(
        self,
        message_bus=None,
        cig_compiler_client=None,
        cig_compiler_target: str | None = None,
        crg_repository: Any = None,
    ):
        super().__init__("nl2obj", message_bus)
        self._subscription_subjects = ["agent.nl2obj.request", "orchestrator.nl2obj.resolve"]
        self.crg = ChemicalReasoningGraph()
        self.cig_compiler_client = cig_compiler_client or _build_cig_compiler_client(
            cig_compiler_target
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
        """Parse natural language intent into structured objective specification (CIG).

        Extracts target properties, constraints, and objectives from user intent text
        and compiles them into a Compliant Intent Graph (CIG).
        """
        intent_text = (
            data.get("intent")
            or data.get("text")
            or data.get("natural_language_prompt")
            or data.get("nl_query")
            or ""
        )
        if not intent_text:
            raise ValueError("intent text is required")

        project_id = str(data.get("project_id") or "")
        run_id = str(data.get("run_id") or data.get("request_id") or "")
        parsed = parse_intent(str(intent_text))
        constraints = dict(parsed["constraints"])
        constraints.update(data.get("constraints", {}))
        confidence = _confidence(parsed)
        parsed_belief = self.crg.add_belief(
            subject=str(intent_text),
            predicate="parsed_intent",
            obj=parsed["intent_summary"],
            confidence=confidence,
            source_agent=self.name,
            evidence_ids=list(parsed.get("tokens", [])),
        )
        result = {
            "agent": self.name,
            "status": "resolved",
            "intent": intent_text,
            "parsed_intent": parsed["intent_summary"],
            "confidence": confidence,
            "objectives": {
                "target_properties": data.get("target_properties", {}),
                "targets": parsed["target_details"],
                "activity": parsed["activity"],
                "admet_constraints": parsed["admet_constraints"],
                "synthetic_constraints": parsed["synthetic_constraints"],
                "objectives_priority": parsed["objectives_priority"],
                "scaffold_hints": parsed["scaffold_hints"],
                "task": parsed["task"],
                "n_samples": parsed["n_samples"],
                "constraints": constraints,
            },
        }
        beliefs_to_persist = [parsed_belief]
        cached_compiled = await self._compiled_cig_from_shared_crg(
            run_id,
            str(intent_text),
        )
        if cached_compiled is not None:
            result.update(cached_compiled)
            result["cached"] = True
            result["cache_source"] = "shared_crg"
        elif self.cig_compiler_client is not None:
            compiled = await _compile_cig(
                self.cig_compiler_client,
                {
                    "project_id": project_id,
                    "nl_query": str(intent_text),
                    "seed": data.get("seed"),
                },
            )
            result.update(
                {
                    key: compiled[key]
                    for key in ("cig", "hciv", "intent_cone")
                    if key in compiled
                }
            )
            compiled_belief = self.crg.add_belief(
                subject=str(intent_text),
                predicate="compiled_cig",
                obj=_compiled_cig_object_value(compiled),
                confidence=confidence,
                source_agent=self.name,
            )
            beliefs_to_persist.append(compiled_belief)
        for belief in beliefs_to_persist:
            await self._persist_belief(
                belief,
                project_id=project_id,
                run_id=run_id,
            )
        return result

    async def _compiled_cig_from_shared_crg(
        self,
        run_id: str,
        intent_text: str,
    ) -> dict[str, Any] | None:
        if not run_id or self.crg_repository is None:
            return None
        read_crg = getattr(self.crg_repository, "get_run_crg", None)
        if not callable(read_crg):
            return None
        crg = read_crg(run_id)
        if inspect.isawaitable(crg):
            crg = await crg
        if not isinstance(crg, dict):
            return None
        for belief in crg.get("beliefs", []):
            cached = _compiled_cig_from_belief(belief, intent_text)
            if cached is not None:
                return cached
        return None

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


def _confidence(parsed: dict) -> float:
    token_count = len(parsed.get("tokens", []))
    if token_count >= 4:
        return 0.9
    if token_count >= 2:
        return 0.75
    return 0.6


def _compiled_cig_object_value(compiled: dict) -> str:
    return json.dumps(
        {
            key: compiled[key]
            for key in ("cig", "hciv", "intent_cone")
            if key in compiled
        },
        sort_keys=True,
    )


def _compiled_cig_from_belief(
    belief: object,
    intent_text: str,
) -> dict[str, Any] | None:
    if not isinstance(belief, dict):
        return None
    if str(belief.get("predicate") or "") != "compiled_cig":
        return None
    if str(belief.get("subject") or "") != intent_text:
        return None
    raw_value = belief.get("object_value", belief.get("object"))
    if not isinstance(raw_value, str):
        return None
    try:
        payload = json.loads(raw_value)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    compiled = {
        key: payload[key]
        for key in ("cig", "hciv", "intent_cone")
        if isinstance(payload.get(key), dict)
    }
    if set(compiled) != {"cig", "hciv", "intent_cone"}:
        return None
    return compiled


def _build_cig_compiler_client(cig_compiler_target: str | None):
    target = cig_compiler_target or os.environ.get("CIG_COMPILER_TARGET", "")
    return CIGCompilerGrpcClient(target) if target else None


async def _compile_cig(client: Any, request: dict) -> dict:
    if hasattr(client, "compile_intent"):
        result = client.compile_intent(request)
    elif callable(client):
        result = client(request)
    else:
        raise TypeError("cig compiler client must expose compile_intent(request) or be callable")
    if inspect.isawaitable(result):
        result = await result
    if not isinstance(result, dict):
        raise TypeError("cig compiler client must return a dictionary")
    return result


def _compiled_intent_from_proto(response) -> dict:
    return {
        "cig": _cig_to_dict(response.cig),
        "hciv": _hciv_to_dict(response.hciv),
        "intent_cone": _intent_cone_to_dict(response.intent_cone),
    }


def _cig_to_dict(cig) -> dict:
    return {
        "project_id": str(cig.project_id),
        "objectives": [
            {
                "id": str(item.id),
                "name": str(item.name),
                "type": cig_pb2.ObjectiveType.Name(item.type),
                "target_value": float(item.target_value),
                "target_min": item.target_min if item.HasField("target_min") else None,
                "target_max": item.target_max if item.HasField("target_max") else None,
                "property": str(item.property),
                "weight": float(item.weight),
                "pareto_tier": int(item.pareto_tier),
            }
            for item in cig.objectives
        ],
        "edges": [
            {
                "source_id": str(item.source_id),
                "target_id": str(item.target_id),
                "relation": str(item.relation),
                "strength": float(item.strength),
            }
            for item in cig.edges
        ],
        "hyperedges": [
            {
                "source_ids": [str(source_id) for source_id in item.source_ids],
                "target_ids": [str(target_id) for target_id in item.target_ids],
                "relation": str(item.relation),
                "strength": float(item.strength),
            }
            for item in cig.hyperedges
        ],
        "constraints": {str(key): str(value) for key, value in cig.constraints.items()},
        "created_by": str(cig.created_by),
    }


def _hciv_to_dict(hciv) -> dict:
    payload = {
        "coordinates": [float(item) for item in hciv.coordinates],
        "curvature": float(hciv.curvature),
        "molecule_smiles": str(hciv.molecule_smiles),
    }
    if hciv.HasField("parent_hciv_id"):
        payload["parent_hciv_id"] = str(hciv.parent_hciv_id)
    return payload


def _intent_cone_to_dict(cone) -> dict:
    return {
        "axis": [float(item) for item in cone.axis],
        "half_angle": float(cone.half_angle),
        "curvature": float(cone.curvature),
        "property_weights": {
            str(key): float(value)
            for key, value in cone.property_weights.items()
        },
    }


def _field(record: Any, name: str, fallback: Any) -> Any:
    if isinstance(record, dict):
        return record.get(name, fallback)
    return getattr(record, name, fallback)
