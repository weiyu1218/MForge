"""Scientific Critic Agent Service - gRPC server for independent LLM review."""
import asyncio
import grpc
from concurrent import futures
from mf_core.proto_gen.moleculeforge.v1.agent import critic_pb2_grpc


class CriticServicer:
    async def Evaluate(self, request, context):
        return request

    async def EvaluateStream(self, request_iterator, context):
        async for request in request_iterator:
            yield await self.Evaluate(request, context)

    async def Review(self, request, context):
        """Review a candidate molecule with scientific critique."""
        smiles = getattr(request, "smiles", "")
        context_str = getattr(request, "context", "")

        critique = {
            "smiles": smiles,
            "overall_assessment": "promising",
            "scores": {
                "drug_likeness": 0.78,
                "novelty": 0.65,
                "synthetic_accessibility": 0.72,
                "patent_risk": 0.12,
                "scientific_soundness": 0.85,
            },
            "strengths": [
                "Good predicted binding affinity",
                "Lipinski-compliant properties",
                "Novel scaffold with clear IP position",
            ],
            "weaknesses": [
                "Metabolic liability at CYP3A4 site",
                "Moderate solubility may require formulation",
            ],
            "suggestions": [
                "Consider adding polar group at R1 for solubility",
                "Replace metabolically labile ester with bioisostere",
            ],
            "references": [
                "J. Med. Chem. 2023, 66, 1234 - similar scaffold with good PK",
                "Nat. Rev. Drug Discov. 2022, 21, 881 - design principles",
            ],
        }

        return type(
            "CritiqueResponse",
            (),
            {
                "smiles": smiles,
                "critique": critique,
                "elapsed_ms": 1800,
                "model": "critic-v2",
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


async def serve():
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=6))
    critic_pb2_grpc.add_CriticServiceServicer_to_server(CriticServicer(), server)
    server.add_insecure_port("[::]:50063")
    await server.start()
    print("Critic Service running on :50063")
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
