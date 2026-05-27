"""Generator Router Service - gRPC server for task-aware generator routing."""
import asyncio
import logging
import grpc
from concurrent import futures

import torch
from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2_grpc
from mf_core.routing.task_router import GENERATOR_NAMES, TaskAwareRouter, TaskProfile

logger = logging.getLogger(__name__)


class GeneratorRouterServicer:
    def __init__(self):
        self.router = TaskAwareRouter(n_generators=len(GENERATOR_NAMES))

    async def Route(self, request, context):
        """Route a generation request to generators using the shared TAR."""
        n_select = max(1, min(int(getattr(request, "n_select", 2) or 2), len(GENERATOR_NAMES)))
        profile = _profile_from_request(request)
        hciv = torch.zeros(self.router.hciv_dim)
        weights = self.router.forward(hciv, profile)
        selected = sorted(weights.items(), key=lambda item: item[1], reverse=True)[:n_select]
        selected_generators = [name for name, _ in selected]
        selected_weights = [weight for _, weight in selected]
        total = sum(selected_weights)
        if total > 0:
            selected_weights = [weight / total for weight in selected_weights]

        return type(
            "RouteResponse",
            (),
            {
                "request_id": getattr(request, "request_id", ""),
                "selected_generators": selected_generators,
                "selection_weights": selected_weights,
                "strategy": "task_aware_router",
                "expected_rewards": [weights[name] for name in selected_generators],
                "targets": [
                    {
                        "generator_id": name,
                        "weight": weight,
                        "endpoint": f"{name}:50051",
                    }
                    for name, weight in zip(selected_generators, selected_weights)
                ],
            },
        )()

    async def SubmitFeedback(self, request, context):
        """Submit reward feedback to online learner."""
        generator_name = getattr(request, "generator_name", "")
        generator_idx = getattr(request, "generator_idx", None)
        if not generator_name and generator_idx is not None:
            generator_name = GENERATOR_NAMES[int(generator_idx)]
        reward = getattr(request, "reward", 0.0)
        self.router.update_with_feedback(generator_name, reward)
        return type(
            "FeedbackResponse",
            (),
            {"acknowledged": True, "generator_name": generator_name},
        )()

    async def GetWeights(self, request, context):
        """Get current generator selection weights."""
        weights = self.router.forward(torch.zeros(self.router.hciv_dim), TaskProfile())
        return type(
            "WeightsResponse",
            (),
            {
                "generator_names": list(GENERATOR_NAMES),
                "weights": [weights[name] for name in GENERATOR_NAMES],
                "counts": [
                    self.router.oracle_history[name]["n_calls"]
                    for name in GENERATOR_NAMES
                ],
                "rewards": [
                    self.router.oracle_history[name]["avg_hvi"]
                    for name in GENERATOR_NAMES
                ],
            },
        )()


def _profile_from_request(request) -> TaskProfile:
    prior_weights = {}
    request_weights = list(getattr(request, "generator_weights", []) or [])
    for name, weight in zip(GENERATOR_NAMES, request_weights):
        prior_weights[name] = float(weight)
    return TaskProfile(
        target_family=str(getattr(request, "target_family", "")),
        stage=str(getattr(request, "stage", "hit_finding") or "hit_finding"),
        data_richness=float(getattr(request, "data_richness", 100.0) or 100.0),
        fto_risk=float(getattr(request, "fto_risk", 0.5) or 0.5),
        novelty_demand=float(getattr(request, "novelty_demand", 0.5) or 0.5),
        multi_target=bool(getattr(request, "multi_target", False)),
        sa_constraint=float(getattr(request, "sa_constraint", 4.0) or 4.0),
        n_samples=int(getattr(request, "n_samples", 100) or 100),
        prior_weights=prior_weights,
    )


async def serve():
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    router_pb2_grpc.add_GeneratorRouterServiceServicer_to_server(
        GeneratorRouterServicer(),
        server,
    )
    server.add_insecure_port("[::]:50052")
    await server.start()
    logger.info("Generator Router Service running on :50052")
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
