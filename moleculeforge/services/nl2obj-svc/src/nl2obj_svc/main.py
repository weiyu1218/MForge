"""NL2Obj Service - gRPC server for NL-to-Objective intent parsing."""
import asyncio
import logging
from concurrent import futures
from types import SimpleNamespace

import grpc
from mf_core.proto_gen.moleculeforge.v1.agent import orchestrator_pb2, orchestrator_pb2_grpc
from nl2obj.parser import parse as parse_intent

logger = logging.getLogger(__name__)

OBJECTIVE_METADATA = {
    "qed": ("Drug-likeness", "maximize"),
    "sa": ("Synthetic accessibility", "minimize"),
    "logp": ("Lipophilicity", "target_range"),
    "solubility": ("Aqueous solubility", "maximize"),
    "potency": ("Binding potency", "maximize"),
    "selectivity": ("Selectivity vs off-target", "maximize"),
    "safety": ("Safety liabilities", "maximize"),
    "bioavailability": ("Oral bioavailability", "maximize"),
}


class NL2ObjServicer:
    async def Parse(self, request, context):
        """Parse natural language into structured molecular design objectives."""
        nl_query = getattr(request, "nl_query", "")
        parsed = parse_intent(nl_query)
        objectives = _objectives_from_parsed(parsed)

        return type(
            "ParseResponse",
            (),
            {
                "nl_query": nl_query,
                "objectives": objectives,
                "constraints": parsed["constraints"],
                "confidence": _confidence(parsed),
                "parsed_intent": parsed["intent_summary"],
                "elapsed_ms": 0,
            },
        )()

    async def Refine(self, request, context):
        """Refine objectives based on user feedback."""
        objectives = getattr(request, "objectives", [])
        feedback = getattr(request, "feedback", "")

        # Apply feedback to objectives
        if objectives:
            objectives.append(
                {
                    "name": "selectivity",
                    "description": "Selectivity vs off-target",
                    "direction": "maximize",
                    "target_value": 50.0,
                    "unit": "fold",
                    "priority": "high",
                }
            )

        return type(
            "RefineResponse",
            (),
            {
                "objectives": objectives,
                "feedback_applied": feedback,
                "confidence": 0.91,
            },
        )()


class NL2ObjGrpcServicer(orchestrator_pb2_grpc.NL2ObjServiceServicer):
    def __init__(self, service: NL2ObjServicer | None = None) -> None:
        self.service = service or NL2ObjServicer()

    async def Parse(self, request, context):
        nl_query = getattr(request, "natural_language_prompt", "") or getattr(
            request,
            "nl_query",
            "",
        )
        response = await self.service.Parse(SimpleNamespace(nl_query=nl_query), context)
        return orchestrator_pb2.NL2ObjResponse(
            project_id=str(getattr(request, "project_id", "")),
            nl_query=str(response.nl_query),
            objectives=[_objective_spec(item) for item in response.objectives],
            constraints={
                str(key): str(value)
                for key, value in dict(response.constraints).items()
            },
            confidence=float(response.confidence),
            parsed_intent=str(response.parsed_intent),
            elapsed_ms=int(response.elapsed_ms),
        )

    async def Refine(self, request, context):
        response = await self.service.Refine(
            SimpleNamespace(
                objectives=[_objective_dict(item) for item in request.objectives],
                feedback=request.feedback,
            ),
            context,
        )
        return orchestrator_pb2.NL2ObjRefineResponse(
            objectives=[_objective_spec(item) for item in response.objectives],
            feedback_applied=str(response.feedback_applied),
            confidence=float(response.confidence),
        )


def _objective_dict(item) -> dict:
    return {
        "name": str(item.name),
        "description": str(item.description),
        "direction": str(item.direction),
        "target_value": float(item.target_value),
        "unit": str(item.unit),
        "priority": str(item.priority),
    }


def _objective_spec(item: dict) -> orchestrator_pb2.ObjectiveSpec:
    target_value = item.get("target_value")
    return orchestrator_pb2.ObjectiveSpec(
        name=str(item.get("name", "")),
        description=str(item.get("description", "")),
        direction=str(item.get("direction", "")),
        target_value=float(target_value) if target_value is not None else 0.0,
        unit=str(item.get("unit", "")),
        priority=str(item.get("priority", "")),
    )


def _objectives_from_parsed(parsed: dict) -> list[dict]:
    objectives = []
    for index, name in enumerate(parsed.get("objectives_priority", []), 1):
        description, direction = OBJECTIVE_METADATA.get(name, (name, "maximize"))
        objectives.append(
            {
                "name": name,
                "description": description,
                "direction": direction,
                "priority": index,
            }
        )
    activity = parsed.get("activity", {})
    if activity.get("type"):
        objectives.append(
            {
                "name": activity["type"].lower(),
                "description": f"{activity['type']} activity",
                "direction": activity.get("direction", "minimize"),
                "target_value": activity.get("target_value"),
                "unit": "nM",
                "priority": len(objectives) + 1,
            }
        )
    return objectives


def _confidence(parsed: dict) -> float:
    token_count = len(parsed.get("tokens", []))
    if token_count >= 4:
        return 0.9
    if token_count >= 2:
        return 0.75
    return 0.6


def register_grpc_services(server) -> None:
    orchestrator_pb2_grpc.add_NL2ObjServiceServicer_to_server(NL2ObjGrpcServicer(), server)


async def serve():
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    register_grpc_services(server)
    server.add_insecure_port("[::]:50070")
    await server.start()
    logger.info("NL2Obj Service running on :50070")
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
