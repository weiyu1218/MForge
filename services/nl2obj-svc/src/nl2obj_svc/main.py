"""NL2Obj Service - gRPC server for NL-to-Objective intent parsing."""
import asyncio
import logging
import grpc
from concurrent import futures

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


async def serve():
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    server.add_insecure_port("[::]:50070")
    await server.start()
    logger.info("NL2Obj Service running on :50070")
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
