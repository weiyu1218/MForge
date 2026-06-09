"""Scientific Critic Agent Service - gRPC server for independent LLM review."""
import asyncio
from concurrent import futures

import grpc
from mf_core.proto_gen.moleculeforge.v1.agent import critic_pb2, critic_pb2_grpc


class CriticServicer:
    def __init__(self, agent=None):
        self.agent = agent

    async def Evaluate(self, request, context):
        smiles = getattr(request, "molecule_smiles", "")
        result = await self._agent().evaluate_molecule(
            {
                "smiles": smiles,
                "properties": _properties_from_request(request),
            }
        )
        return _batch_result_from_agent_result(
            result,
            project_id=getattr(request, "project_id", ""),
        )

    async def EvaluateStream(self, request_iterator, context):
        async for request in request_iterator:
            yield await self.Evaluate(request, context)

    async def Review(self, request, context):
        """Review a candidate molecule with scientific critique."""
        smiles = getattr(request, "smiles", "")
        result = await self._agent().evaluate_molecule({"smiles": smiles, "properties": {}})

        return type(
            "CritiqueResponse",
            (),
            {
                "smiles": smiles,
                "critique": result,
                "elapsed_ms": 0,
                "model": "scientific_critic_rules",
            },
        )()

    async def BatchReview(self, request, context):
        """Batch review multiple molecules."""
        results = []
        for req in getattr(request, "requests", []):
            results.append(await self.Review(req, context))
        return type(
            "BatchCritiqueResponse",
            (),
            {"results": results, "total_elapsed_ms": 5000},
        )()

    def _agent(self):
        if self.agent is None:
            from critic_agent.agent import ScientificCriticAgent

            self.agent = ScientificCriticAgent()
        return self.agent


def _properties_from_request(request) -> dict[str, float]:
    properties: dict[str, float] = {}
    for feedback in getattr(request, "rule_results", []):
        for key, value in getattr(feedback, "metric_values", {}).items():
            properties[str(key)] = float(value)
    return properties


def _batch_result_from_agent_result(result: dict, project_id: str):
    feedback = [_feedback_from_rule(result["smiles"], row) for row in result["rule_results"]]
    scores = [item.score for item in feedback]
    aggregate_score = sum(scores) / len(scores) if scores else 0.0
    return critic_pb2.CriticBatchResult(
        molecule_smiles=str(result["smiles"]),
        project_id=str(project_id),
        rule_results=feedback,
        all_passed=str(result.get("verdict")) == "pass",
        rules_evaluated=int(result.get("total_rules", len(feedback))),
        rules_passed=int(result.get("passed", 0)),
        aggregate_score=aggregate_score,
    )


def _feedback_from_rule(smiles: str, row: dict):
    verdict = str(row.get("verdict", "error"))
    rule_name = str(row.get("rule_name", ""))
    return critic_pb2.CriticFeedback(
        molecule_smiles=smiles,
        rule_id=str(row.get("rule_id", "")),
        rule_name=rule_name,
        verdict=verdict,
        score=float(row.get("score", 0.0) or 0.0),
        reasoning=str(row.get("reasoning", "")),
        violated_constraints=[rule_name] if verdict == "fail" and rule_name else [],
        satisfied_constraints=[rule_name] if verdict == "pass" and rule_name else [],
    )


async def serve():
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=6))
    critic_pb2_grpc.add_CriticServiceServicer_to_server(CriticServicer(), server)
    server.add_insecure_port("[::]:50063")
    await server.start()
    print("Critic Service running on :50063")
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
