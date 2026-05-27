"""Orchestrator Service - FastAPI + gRPC server for LangGraph-driven design loops."""
import asyncio
import logging
import os
from concurrent import futures
from datetime import datetime, timezone

import grpc
from fastapi import FastAPI, HTTPException
from orchestrator.workflow.graph_builder import WorkflowGraph, create_initial_state

rest_app = FastAPI(title="Orchestrator Service", version="0.1.0")
_RUNS: dict[str, dict] = {}
LOGGER = logging.getLogger(__name__)


@rest_app.get("/health")
async def health():
    return {"status": "healthy", "engine": "langgraph", "runs": len(_RUNS)}


@rest_app.post("/v1/orchestrator/design")
async def start_design(request: dict):
    """Start a new molecular design workflow."""
    design_id = f"design-{datetime.now(timezone.utc).timestamp():.0f}"
    nl_input = request.get("nl_input") or request.get("intent")
    if not nl_input:
        raise HTTPException(status_code=400, detail="nl_input is required")
    workflow_scope = str(
        request.get("workflow_scope")
        or os.environ.get("ORCHESTRATOR_WORKFLOW_SCOPE")
        or "state_only"
    )
    state = create_initial_state(
        str(nl_input),
        run_id=request.get("run_id"),
        trace_id=request.get("trace_id"),
        artifact_ids=request.get("artifact_ids") or [],
        workflow_scope=workflow_scope,
    )
    state["request"] = dict(request)
    state["validation_passed"] = bool(request.get("validation_passed", True))
    state["max_refinements"] = int(request.get("max_refinements", 1))
    clients = request.get("clients")
    if clients is None and workflow_scope == "engineering":
        clients = EngineeringWorkflowClients()
    compiled = WorkflowGraph(clients=clients, workflow_scope=workflow_scope).build()
    final_state = await compiled.ainvoke(state)
    status = "completed" if final_state.get("status") != "PAUSED" else "paused"
    _RUNS[design_id] = {
        "design_id": design_id,
        "status": status,
        "state": final_state,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return {
        "design_id": design_id,
        "run_id": final_state.get("run_id"),
        "trace_id": final_state.get("trace_id"),
        "status": status,
        "current_stage": final_state.get("status"),
        "artifact_ids": final_state.get("artifact_ids", []),
        "history": final_state.get("history", []),
        "pipeline": final_state.get("history", []),
        "events": final_state.get("events", []),
        "state": final_state,
    }


@rest_app.get("/v1/orchestrator/{design_id}")
async def get_design_status(design_id: str):
    """Get design workflow status from the LangGraph state machine."""
    run = _RUNS.get(design_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Unknown design_id: {design_id}")
    state = run["state"]
    return {
        "design_id": design_id,
        "run_id": state.get("run_id"),
        "trace_id": state.get("trace_id"),
        "status": run["status"],
        "current_stage": state.get("status"),
        "artifact_ids": state.get("artifact_ids", []),
        "history": state.get("history", []),
        "stages_completed": len(state.get("history", [])),
        "stages_total": len(state.get("history", [])),
        "events": state.get("events", []),
        "state": state,
    }


@rest_app.post("/v1/orchestrator/{design_id}/pause")
async def pause_design(design_id: str):
    """Pause a running design workflow."""
    run = _RUNS.get(design_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Unknown design_id: {design_id}")
    run["status"] = "paused"
    return {"design_id": design_id, "status": "paused"}


@rest_app.post("/v1/orchestrator/{design_id}/resume")
async def resume_design(design_id: str):
    """Resume a paused design workflow."""
    run = _RUNS.get(design_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Unknown design_id: {design_id}")
    if run["status"] == "paused":
        run["status"] = "completed"
    return {"design_id": design_id, "status": run["status"]}


class OrchestratorServicer:
    async def StartPipeline(self, request, context):
        """gRPC: Start a new design pipeline."""
        project_id = getattr(request, "project_id", "")
        objectives = getattr(request, "objectives", [])
        response = await start_design(
            {
                "nl_input": getattr(request, "nl_input", "") or project_id or "design pipeline",
                "validation_passed": True,
                "workflow_scope": getattr(request, "workflow_scope", "state_only"),
            }
        )

        return type(
            "PipelineResponse",
            (),
            {
                "design_id": response["design_id"],
                "run_id": response["run_id"],
                "trace_id": response["trace_id"],
                "project_id": project_id,
                "status": response["status"],
                "n_objectives": len(objectives),
            },
        )()

    async def GetPipelineState(self, request, context):
        """gRPC: Get pipeline state from LangGraph."""
        design_id = getattr(request, "design_id", "")
        state = await get_design_status(design_id)
        return type(
            "PipelineStateResponse",
            (),
            {
                "design_id": design_id,
                "current_stage": state["current_stage"],
                "state_json": str(state["state"]),
            },
        )()


class EngineeringWorkflowClients:
    """Local, resource-light clients for the reduced engineering workflow."""

    async def compile_intent(self, state: dict) -> dict:
        from cig_compiler_svc.domain.compiler import CIGCompiler, CompilerMode, EncodingMode

        compiler = CIGCompiler(
            mode=CompilerMode.LOCAL_DEMO,
            encoding_mode=EncodingMode.HASH,
            enable_grounding=False,
        )
        cig, hciv, cone = await compiler.compile(state["nl_input"])
        return {
            "cig": cig.model_dump(mode="json"),
            "hciv": hciv.model_dump(mode="json"),
            "intent_cone": cone.model_dump(mode="json"),
        }

    async def generate_candidates(self, state: dict) -> list[dict]:
        from mf_generators.rdkit_random import RDKitRandomGenerator

        request = state.get("request", {})
        n_samples = int(request.get("n_samples", request.get("batch_size", 8)) or 8)
        seed = request.get("seed")
        generator = RDKitRandomGenerator(seed=int(seed) if seed is not None else 42)
        candidates = []
        async for molecule in generator.generate(
            state.get("hciv"),
            state.get("intent_cone"),
            state.get("cig"),
            n_samples=n_samples,
            seed=int(seed) if seed is not None else 42,
        ):
            candidates.append(molecule.model_dump(mode="json"))
        return candidates

    async def validate_candidates(self, state: dict) -> dict:
        from mf_oracles.rdkit_oracle.oracle import RDKitOracle

        candidates = list(state.get("candidates", []))
        if not candidates:
            return {"passed": False, "results": [], "reason": "no candidates generated"}
        smiles = [candidate["canonical_smiles"] for candidate in candidates]
        results = await RDKitOracle().evaluate(smiles, ["admet_score"])
        threshold = float(state.get("request", {}).get("l0_threshold", 0.0))
        rows = [
            {"smiles": smiles_item, **scores}
            for smiles_item, scores in results.items()
        ]
        return {
            "passed": any(float(row.get("admet_score", 0.0)) >= threshold for row in rows),
            "threshold": threshold,
            "results": rows,
        }

    async def plan_routes(self, state: dict) -> dict:
        if not os.environ.get("AIZYNTH_CONFIG_PATH"):
            return {
                "skipped": True,
                "reason": "AIZYNTH_CONFIG_PATH is not configured",
            }
        candidates = list(state.get("candidates", []))
        if not candidates:
            return {"skipped": True, "reason": "no candidate available for retrosynthesis"}
        from mf_retrosyn.aizynth.retrosyn import AiZynthRetrosyn

        planner = AiZynthRetrosyn.from_env()
        routes = await planner.find_routes(candidates[0]["canonical_smiles"], max_routes=3)
        return {"skipped": False, "routes": routes}

    async def review_candidates(self, state: dict) -> dict:
        candidates = list(state.get("candidates", []))
        if not candidates:
            return {"verdict": "fail", "reason": "no candidate available for critic"}
        from critic_agent.agent import ScientificCriticAgent

        properties = {}
        validation_rows = state.get("validation", {}).get("results", [])
        if validation_rows:
            properties = dict(validation_rows[0])
        return await ScientificCriticAgent().evaluate_molecule(
            {
                "smiles": candidates[0]["canonical_smiles"],
                "properties": properties,
            }
        )


async def serve_grpc():
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    server.add_insecure_port("[::]:50071")
    await server.start()
    LOGGER.info("Orchestrator gRPC Service running on :50071")
    await server.wait_for_termination()


if __name__ == "__main__":
    import uvicorn

    async def main():
        asyncio.create_task(serve_grpc())
        config = uvicorn.Config(rest_app, host="0.0.0.0", port=8011, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()

    asyncio.run(main())
