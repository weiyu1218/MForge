"""Scientific Critic Agent Service - gRPC server for independent LLM review."""

import asyncio
from concurrent import futures

import grpc
from mf_core.proto_gen.moleculeforge.v1.agent import critic_pb2, critic_pb2_grpc


class CriticServicer(critic_pb2_grpc.CriticServiceServicer):
    def __init__(self, agent=None):
        self.agent = agent

    async def Evaluate(self, request, context):
        try:
            payload = _critic_request_payload(request)
        except (TypeError, ValueError) as exc:
            return await _abort_invalid_request(context, exc)
        result = await self._agent().evaluate_molecule(payload)
        return _batch_result_from_agent_result(result, request)

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


def _critic_request_payload(request) -> dict:
    fields = {}
    for field in (
        "molecule_smiles",
        "project_id",
        "run_id",
        "request_id",
        "schema_version",
        "candidate_id",
        "canonical_smiles",
    ):
        value = getattr(request, field, None)
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f"{field} must be a non-empty trimmed string")
        fields[field] = value
    if fields["schema_version"] != "critic.batch.v1":
        raise ValueError("schema_version must be critic.batch.v1")
    if fields["canonical_smiles"] != fields["molecule_smiles"]:
        raise ValueError("canonical_smiles must match molecule_smiles")
    if not request.HasField("candidate_index") or request.candidate_index < 0:
        raise ValueError("candidate_index must be present and non-negative")
    return {
        "workflow_scope": "full",
        "project_id": fields["project_id"],
        "run_id": fields["run_id"],
        "request_id": fields["request_id"],
        "schema_version": fields["schema_version"],
        "candidate_id": fields["candidate_id"],
        "candidate_index": int(request.candidate_index),
        "canonical_smiles": fields["canonical_smiles"],
        "smiles": fields["molecule_smiles"],
        "properties": _properties_from_request(request),
    }


async def _abort_invalid_request(context, error: Exception):
    if context is not None and hasattr(context, "abort"):
        await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
    raise error


def _batch_result_from_agent_result(result: dict, request):
    feedback = [_feedback_from_rule(result["smiles"], row) for row in result["rule_results"]]
    scores = [item.score for item in feedback]
    aggregate_score = sum(scores) / len(scores) if scores else 0.0
    return critic_pb2.CriticBatchResult(
        molecule_smiles=str(result["smiles"]),
        project_id=str(request.project_id),
        rule_results=feedback,
        all_passed=str(result.get("verdict")) == "pass",
        rules_evaluated=len(feedback),
        rules_passed=int(result.get("passed", 0)),
        aggregate_score=aggregate_score,
        candidate_id=str(request.candidate_id),
        candidate_index=int(request.candidate_index),
        canonical_smiles=str(request.canonical_smiles),
        run_id=str(request.run_id),
        request_id=str(request.request_id),
        schema_version=str(request.schema_version),
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
