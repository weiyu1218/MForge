"""CIG Compiler Service - gRPC server for NL-to-Chemical-Intent-Graph parsing."""
import asyncio
from concurrent import futures
from time import perf_counter

import grpc

from cig_compiler_svc.domain.compiler import CIGCompiler


async def _abort_unavailable(context, message: str):
    if context is not None and hasattr(context, "abort"):
        await context.abort(grpc.StatusCode.FAILED_PRECONDITION, message)
    raise RuntimeError(message)


class CIGCompilerServicer:
    def __init__(self, compiler: CIGCompiler | None = None) -> None:
        self.compiler = compiler or CIGCompiler()

    async def Compile(self, request, context):
        """Compile natural language into a Chemical Intent Graph (CIG)."""
        start = perf_counter()
        nl_query = getattr(request, "nl_query", "")
        seed = getattr(request, "seed", None)

        try:
            cig, hciv, cone = await self.compiler.compile(nl_query, seed=seed)
        except RuntimeError as exc:
            return await _abort_unavailable(context, str(exc))

        return type(
            "CIGResponse",
            (),
            {
                "cig": cig.model_dump(mode="json"),
                "hciv": hciv.model_dump(mode="json"),
                "intent_cone": cone.model_dump(mode="json"),
                "parse_confidence": None,
                "ambiguities": [],
                "elapsed_ms": int((perf_counter() - start) * 1000),
            },
        )()

    async def Validate(self, request, context):
        """Validate a CIG for consistency and completeness."""
        cig = getattr(request, "cig", {})
        goals = cig.get("goals", [])

        issues = []
        if not goals:
            issues.append("No goals defined in CIG")
        if not cig.get("edges"):
            issues.append("No edges defined in CIG")
        valid = len(issues) == 0

        return type(
            "ValidationResponse",
            (),
            {
                "valid": valid,
                "issues": issues,
                "warnings": [],
                "suggestions": [],
            },
        )()

    async def Refine(self, request, context):
        """Refine a CIG based on feedback or additional context."""
        return await _abort_unavailable(
            context,
            "CIG refinement runner is not configured",
        )


async def serve():
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    server.add_insecure_port("[::]:50061")
    await server.start()
    print("CIG Compiler Service running on :50061")
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
