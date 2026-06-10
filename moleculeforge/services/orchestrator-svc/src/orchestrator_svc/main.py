"""Orchestrator Service - FastAPI + gRPC server for LangGraph-driven design loops."""
import asyncio
import json
import logging
import math
import os
import re
import struct
from concurrent import futures
from datetime import UTC, datetime
from types import SimpleNamespace

import grpc
from fastapi import FastAPI, HTTPException
from mf_core.geometry.lorentz import normalize_lorentz_embedding
from mf_core.proto_gen.moleculeforge.v1.agent import orchestrator_pb2, orchestrator_pb2_grpc
from orchestrator.workflow.graph_builder import WorkflowGraph, create_initial_state

rest_app = FastAPI(title="Orchestrator Service", version="0.1.0")
_RUNS: dict[str, dict] = {}
LOGGER = logging.getLogger(__name__)
_CURRENT_HFM_LORENTZ_DIM = 129
_FULL_WORKFLOW_BLOCKING_CRITIC_RULE_IDS = [
    "rule_001",
    "rule_004",
    "rule_005",
    "rule_014",
    "rule_015",
    "rule_016",
    "rule_017",
    "rule_018",
    "rule_019",
    "rule_020",
    "rule_021",
    "rule_022",
    "rule_024",
    "rule_025",
    "rule_026",
    "rule_027",
    "rule_028",
    "rule_029",
    "rule_030",
    "rule_045",
    "rule_046",
    "rule_049",
    "rule_050",
    "rule_051",
    "rule_052",
    "rule_053",
    "rule_054",
    "rule_055",
    "rule_056",
    "rule_057",
    "rule_058",
    "rule_059",
    "rule_070",
    "rule_074",
    "rule_076",
    "rule_087",
    "rule_088",
    "rule_089",
    "rule_090",
    "rule_091",
    "rule_092",
    "rule_098",
    "rule_099",
    "rule_100",
    "crg_validation_status",
    "crg_retrosyn_routes",
]


@rest_app.get("/health")
async def health():
    return {"status": "healthy", "engine": "langgraph", "runs": len(_RUNS)}


@rest_app.post("/v1/orchestrator/design")
async def start_design(request: dict):
    """Start a new molecular design workflow."""
    design_id = f"design-{datetime.now(UTC).timestamp():.0f}"
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
    if clients is None and workflow_scope == "full":
        clients = FullWorkflowClients()
    compiled = WorkflowGraph(clients=clients, workflow_scope=workflow_scope).build()
    final_state = await compiled.ainvoke(state)
    if workflow_scope == "full":
        await _record_workflow_provenance(final_state)
    status = _run_status(final_state)
    _RUNS[design_id] = {
        "design_id": design_id,
        "status": status,
        "state": final_state,
        "created_at": datetime.now(UTC).isoformat(),
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
                "run_id": getattr(request, "run_id", None) or None,
                "trace_id": getattr(request, "trace_id", None) or None,
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


class OrchestratorGrpcServicer(orchestrator_pb2_grpc.OrchestratorServiceServicer):
    def __init__(self, service: OrchestratorServicer | None = None):
        self.service = service or OrchestratorServicer()

    async def StartPipeline(self, request, context):
        response = await self.service.StartPipeline(request, context)
        return orchestrator_pb2.PipelineResponse(
            design_id=str(response.design_id),
            run_id=str(response.run_id),
            trace_id=str(response.trace_id),
            project_id=str(response.project_id),
            status=str(response.status),
            n_objectives=int(response.n_objectives),
        )

    async def GetPipelineState(self, request, context):
        response = await self.service.GetPipelineState(request, context)
        return orchestrator_pb2.PipelineStateResponse(
            design_id=str(response.design_id),
            current_stage=str(response.current_stage),
            state_json=str(response.state_json),
        )


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
        from mf_chem.predict.engine import MolPredictEngine
        from mf_oracles.rdkit_oracle.oracle import RDKitOracle

        candidates = list(state.get("candidates", []))
        if not candidates:
            return {"passed": False, "results": [], "reason": "no candidates generated"}
        smiles = [candidate["canonical_smiles"] for candidate in candidates]
        results = await RDKitOracle().evaluate(smiles, ["admet_score"])
        threshold = float(state.get("request", {}).get("l0_threshold", 0.0))
        predictor = MolPredictEngine(device_ids=[])
        rows = []
        for smiles_item, scores in results.items():
            row = {
                **_engineering_candidate_properties(predictor, smiles_item),
                **scores,
            }
            rows.append(_normalise_engineering_critic_properties(row))
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
            properties = dict(_best_engineering_validation_row(validation_rows))
        return await ScientificCriticAgent().evaluate_molecule(
            {
                "smiles": _best_engineering_candidate_smiles(state),
                "properties": properties,
            }
        )


class FullWorkflowClients(EngineeringWorkflowClients):
    async def generate_candidates(self, state: dict) -> list[dict]:
        request = state.get("request", {})
        n_samples = int(request.get("n_samples", request.get("batch_size", 4)) or 4)
        generator_params = dict(request.get("generator_params") or {})
        generator_params.setdefault("sampling_seed", int(request.get("seed", 42) or 42))
        await _attach_generation_feedback(generator_params, state)
        generation_strategy = str(request.get("generation_strategy") or "")
        if generation_strategy and generation_strategy != "hfm_3d":
            return await _generate_with_generator_coord(
                state,
                request,
                n_samples,
                generator_params,
                generation_strategy,
            )
        from hfm_generator_svc.main import HFMGeneratorServicer

        service = HFMGeneratorServicer()
        response = await service.Generate(
            SimpleNamespace(
                project_id=str(state.get("run_id", "")),
                batch_size=n_samples,
                intent_cone=state.get("intent_cone"),
                generator_params=generator_params,
            ),
            None,
        )
        candidates = []
        for payload in response.molecules:
            candidates.append(json.loads(payload.decode("utf-8")))
        return _normalise_candidate_rows(candidates)

    async def validate_candidates(self, state: dict) -> dict:
        from boltz2_svc.main import Boltz2Servicer
        from mf_core.proto_gen.moleculeforge.v1.oracle import boltz2_pb2

        candidates = list(state.get("candidates", []))
        if not candidates:
            return {"passed": False, "results": [], "reason": "no candidates generated"}
        request = state.get("request", {})
        oracle_level = _requested_oracle_level(request)
        if oracle_level is not None:
            return await _validate_with_oracle_cascade(state, candidates, oracle_level)
        protein_pdb_id = str(request.get("protein_pdb_id") or "6OIM")
        smiles = [candidate["canonical_smiles"] for candidate in candidates]
        response = await Boltz2Servicer().PredictAffinity(
            boltz2_pb2.Boltz2BatchRequest(
                project_id=str(state.get("run_id", "")),
                protein_pdb_id=protein_pdb_id,
                ligand_smiles=smiles,
                ensemble_size=int(request.get("boltz_ensemble_size", 5) or 5),
            ),
            None,
        )
        rows = [
            {
                "smiles": affinity.ligand_smiles,
                "protein_pdb_id": affinity.protein_pdb_id,
                "delta_g_kcal_mol": affinity.delta_g_kcal_mol,
                "uncertainty": affinity.uncertainty,
                "ki_nm": affinity.ki_nm,
            }
            for affinity in response.affinities
        ]
        quality_gate = _affinity_quality_gate(request, state)
        for row in rows:
            row["passes_affinity_gate"] = _passes_affinity_gate(row, quality_gate)
        passed = bool(rows) and bool(quality_gate["configured"]) and any(
            row["passes_affinity_gate"] for row in rows
        )
        result = {
            "passed": passed,
            "results": rows,
            "protein_pdb_id": protein_pdb_id,
            "quality_gate": quality_gate,
        }
        if not quality_gate["configured"]:
            result["reason"] = "boltz_max_ki_nm or min_pkd is required for full workflow"
        return result

    async def plan_routes(self, state: dict) -> dict:
        from retrosyn_agent.agent import RetroSynAgent

        request = state.get("request", {})
        return await RetroSynAgent().process(
            {
                "project_id": str(request.get("project_id") or ""),
                "run_id": str(state.get("run_id", "")),
                "smiles": _first_candidate_smiles(state),
                "max_routes": int(
                    request.get("retrosyn_max_routes", request.get("max_routes", 3))
                    or 3
                ),
            }
        )

    async def assess_supply(self, state: dict) -> dict:
        route = _first_retrosyn_route_or_none(state)
        if route is None:
            return _unavailable_supply_result(state, "retrosyn.routes is empty")
        from supply_agent.agent import SupplyAgent

        return await SupplyAgent().process(
            {
                "project_id": str(state.get("request", {}).get("project_id") or ""),
                "run_id": str(state.get("run_id", "")),
                "smiles": _first_candidate_smiles(state),
                "building_blocks": _route_building_blocks(route),
            }
        )

    async def compile_synthesis(self, state: dict) -> dict:
        if _supply_feasibility(state) == "unavailable":
            return {
                "status": "skipped",
                "protocols": [],
                "skip_reason": "supply feasibility is unavailable",
            }
        route = _first_retrosyn_route_or_none(state)
        if route is None:
            return {
                "status": "skipped",
                "protocols": [],
                "skip_reason": "retrosyn.routes is empty",
            }
        from srb_agent.agent import SRBAgent

        return await SRBAgent().process(
            {
                "project_id": str(state.get("request", {}).get("project_id") or ""),
                "run_id": str(state.get("run_id", "")),
                "molecule": {"smiles": _first_candidate_smiles(state)},
                "retrosyn_route": route,
            }
        )

    async def review_candidates(self, state: dict) -> dict:
        candidates = list(state.get("candidates", []))
        if not candidates:
            return {"verdict": "fail", "reason": "no candidate available for critic"}
        from critic_agent.agent import ScientificCriticAgent

        request = dict(state.get("request") or {})
        smiles = _best_engineering_candidate_smiles(state)
        properties = _full_workflow_critic_properties(state, smiles)
        return await ScientificCriticAgent().evaluate_molecule(
            {
                "project_id": str(request.get("project_id") or ""),
                "run_id": str(state.get("run_id", "")),
                "smiles": smiles,
                "properties": properties,
            }
        )


async def _generate_with_generator_coord(
    state: dict,
    request: dict,
    n_samples: int,
    generator_params: dict,
    generation_strategy: str,
) -> list[dict]:
    from generator_coord.agent import GeneratorCoordAgent

    run_id = str(state.get("run_id", ""))
    payload = {
        "project_id": str(request.get("project_id") or ""),
        "run_id": run_id,
        "request_id": run_id,
        "generation_strategy": generation_strategy,
        "objectives": dict(request.get("objectives") or {}),
        "hciv": state.get("hciv"),
        "intent_cone": state.get("intent_cone"),
        "n_samples": n_samples,
        "batch_size": n_samples,
        "generator_params": dict(generator_params),
    }
    result = await GeneratorCoordAgent().process(payload)
    candidates = result.get("candidates")
    if not isinstance(candidates, list):
        raise RuntimeError("GeneratorCoordAgent must return candidates as a list")
    return _normalise_candidate_rows(candidates)


async def _attach_generation_feedback(generator_params: dict, state: dict) -> None:
    jmcg_feedback = await _jmcg_context_feedback_from_state(state)
    feedback = state.get("generation_feedback")
    if isinstance(feedback, list) and feedback:
        generator_params["generation_feedback"] = json.dumps(feedback, sort_keys=True)
        property_feedback = _property_jmcg_feedback_from_generation_feedback(
            feedback,
            run_id=str(state.get("run_id") or ""),
            project_id=str((state.get("request") or {}).get("project_id") or ""),
        )
        jmcg_feedback = _merge_jmcg_feedback(jmcg_feedback, property_feedback)
    if jmcg_feedback:
        generator_params["jmcg_feedback"] = json.dumps(
            jmcg_feedback,
            sort_keys=True,
        )


async def _jmcg_context_feedback_from_state(state: dict) -> dict | None:
    run_id = str(state.get("run_id") or "")
    request = dict(state.get("request") or {})
    project_id = str(request.get("project_id") or "")
    records = []
    intent_record = _intent_jmcg_feedback_record(state, run_id)
    if intent_record:
        records.append(intent_record)
    pocket_record = await _pocket_jmcg_feedback_record(state, run_id)
    if pocket_record:
        records.append(pocket_record)
    if not records:
        return None
    return {
        "schema": "moleculeforge.jmcg.feedback.v1",
        "run_id": run_id,
        "project_id": project_id,
        "records": records,
    }


def _merge_jmcg_feedback(left: dict | None, right: dict | None) -> dict | None:
    if not left:
        return right
    if not right:
        return left
    merged = dict(left)
    merged["records"] = list(left.get("records") or []) + list(right.get("records") or [])
    return merged


def _intent_jmcg_feedback_record(state: dict, run_id: str) -> dict | None:
    hciv = state.get("hciv")
    intent_cone = state.get("intent_cone")
    if not isinstance(hciv, dict) and not isinstance(intent_cone, dict):
        return None
    metadata = {}
    if isinstance(hciv, dict):
        metadata["has_hciv"] = True
        metadata["hciv_keys"] = sorted(str(key) for key in hciv.keys())
    if isinstance(intent_cone, dict):
        metadata["has_intent_cone"] = True
        metadata["intent_cone_keys"] = sorted(str(key) for key in intent_cone.keys())
        if "half_angle" in intent_cone:
            metadata["half_angle"] = intent_cone["half_angle"]
    record = {
        "kind": "intent",
        "source": "orchestrator_svc",
        "run_id": run_id,
        "subject": {"type": "intent", "id": run_id},
        "weight": 1.0,
        "polarity": "attract",
        "confidence": 1.0,
        "evidence_ids": [],
        "metadata": metadata,
    }
    embedding, embedding_metadata = _intent_feedback_embedding(state)
    if embedding is not None:
        record["humu_embedding"] = embedding
        record["metadata"].update(embedding_metadata)
    return record


async def _pocket_jmcg_feedback_record(state: dict, run_id: str) -> dict | None:
    cig = state.get("cig")
    if not isinstance(cig, dict):
        return None
    target_context = cig.get("target_context")
    if not isinstance(target_context, dict) or not target_context:
        return None
    pocket_metadata = {
        str(key): value
        for key, value in target_context.items()
        if _is_pocket_context_key(str(key))
    }
    if not pocket_metadata:
        return None
    pocket_id = str(
        target_context.get("pocket_id")
        or target_context.get("pdb_id")
        or target_context.get("target_id")
        or run_id
    )
    record = {
        "kind": "pocket",
        "source": "orchestrator_svc",
        "run_id": run_id,
        "subject": {"type": "pocket", "id": pocket_id},
        "weight": 1.0,
        "polarity": "attract",
        "confidence": 1.0,
        "evidence_ids": [],
        "metadata": pocket_metadata,
    }
    payload = _pocket_encoder_payload(target_context)
    if payload is not None:
        feedback = await _encode_pocket_humu_feedback(payload)
        if feedback is not None:
            record["humu_embedding"] = feedback["humu_embedding"]
            record["curvature"] = feedback["curvature"]
            record["source"] = feedback["source"]
            record["evidence_ids"] = feedback["evidence_ids"]
    return record


def _intent_feedback_embedding(state: dict) -> tuple[list[float] | None, dict]:
    intent_cone = state.get("intent_cone")
    if not isinstance(intent_cone, dict):
        return None, {}
    embedding = _valid_hfm_feedback_embedding(intent_cone.get("axis"))
    if embedding is None:
        return None, {}
    return embedding, {"embedding_source": "intent_cone.axis"}


def _valid_hfm_feedback_embedding(value: object) -> list[float] | None:
    return normalize_lorentz_embedding(
        value,
        expected_dim=_CURRENT_HFM_LORENTZ_DIM,
        curvature=1.0,
    )


def _pocket_encoder_payload(target_context: dict) -> dict | None:
    coords = target_context.get("coords") or target_context.get("coordinates")
    elements = target_context.get("elements")
    residues = target_context.get("residue_types") or target_context.get("residues")
    if (
        not isinstance(coords, list)
        or not isinstance(elements, list)
        or not isinstance(residues, list)
    ):
        return None
    if len(coords) != len(elements) or len(coords) != len(residues) or not coords:
        return None
    return {
        "coords": coords,
        "elements": elements,
        "residue_types": residues,
    }


async def _encode_pocket_humu_feedback(payload: dict) -> dict | None:
    target = os.environ.get("HUMU_ENCODER_TARGET", "").strip()
    if not target:
        return None
    from mf_core.proto_gen.moleculeforge.v1.humu import encoder_pb2, encoder_pb2_grpc

    channel = grpc.aio.insecure_channel(target)
    try:
        stub = encoder_pb2_grpc.HUMUEncoderServiceStub(channel)
        response = await stub.Encode(
            encoder_pb2.EncodeRequest(
                entity_type="pocket",
                input_data=json.dumps(payload, sort_keys=True).encode("utf-8"),
            )
        )
    except Exception as exc:
        LOGGER.warning("Skipping pocket HUMU feedback enrichment: %s", exc)
        return None
    finally:
        await channel.close()
    try:
        embedding = _valid_hfm_feedback_embedding(
            _float32_embedding_from_bytes(response.humu_embedding)
        )
    except ValueError as exc:
        LOGGER.warning("Skipping pocket HUMU feedback enrichment: %s", exc)
        return None
    if embedding is None:
        return None
    return {
        "humu_embedding": embedding,
        "curvature": float(response.curvature),
        "source": "humu_encoder_svc",
        "evidence_ids": ["humu_encoder:pocket"],
    }


def _float32_embedding_from_bytes(payload: bytes) -> list[float]:
    if len(payload) % 4 != 0:
        raise ValueError("HUMU embedding bytes must contain float32 values")
    return [float(item[0]) for item in struct.iter_unpack("<f", payload)]


def _is_pocket_context_key(key: str) -> bool:
    lowered = key.lower()
    return (
        "pocket" in lowered
        or lowered in {"pdb_id", "target_id", "binding_mode_prior"}
    )


def _property_jmcg_feedback_from_generation_feedback(
    feedback: list,
    run_id: str,
    project_id: str = "",
) -> dict | None:
    records = []
    for index, item in enumerate(feedback):
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or "workflow")
        records.append(
            {
                "kind": "property",
                "source": source,
                "run_id": run_id,
                "subject": {
                    "type": "workflow_feedback",
                    "id": f"{source}-{index}",
                },
                "weight": float(item.get("weight", 1.0)),
                "polarity": _property_feedback_polarity(item),
                "confidence": float(item.get("confidence", 1.0)),
                "evidence_ids": _feedback_evidence_ids(item.get("evidence_ids")),
                "metadata": _property_feedback_metadata(item),
            }
        )
    if not records:
        return None
    return {
        "schema": "moleculeforge.jmcg.feedback.v1",
        "run_id": run_id,
        "project_id": project_id,
        "records": records,
    }


def _property_feedback_polarity(feedback: dict) -> str:
    explicit = str(feedback.get("polarity") or "")
    if explicit in {"attract", "repel"}:
        return explicit
    if feedback.get("passed") is False:
        return "repel"
    verdict = str(feedback.get("verdict") or "").lower()
    if verdict in {"fail", "failed", "reject"}:
        return "repel"
    return "attract"


def _property_feedback_metadata(feedback: dict) -> dict:
    excluded = {
        "source",
        "weight",
        "confidence",
        "polarity",
        "evidence_ids",
        "humu_embedding",
        "route_humu_embedding",
    }
    return {
        str(key): value
        for key, value in feedback.items()
        if key not in excluded
    }


def _feedback_evidence_ids(value: object) -> list[str]:
    if value in (None, "", b""):
        return []
    if isinstance(value, bytes):
        return [value.decode("utf-8")]
    if isinstance(value, str):
        return [value]
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value]
    return [str(value)]


def _normalise_candidate_rows(candidates: list[dict]) -> list[dict]:
    rows = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise RuntimeError("generated candidates must be objects")
        row = dict(candidate)
        smiles = row.get("canonical_smiles") or row.get("smiles")
        if smiles:
            row["canonical_smiles"] = str(smiles)
        rows.append(row)
    return rows


def _engineering_candidate_properties(predictor, smiles: str) -> dict:
    prediction = predictor.predict_one(smiles)
    row = prediction.to_dict()
    admet = dict(row.get("admet") or {})
    row.update(admet)
    molecular_weight = row.get("molecular_weight")
    if molecular_weight is not None:
        row["mw"] = molecular_weight
    row["ring_count"] = row.get("rings", 0) or 0
    row["fsp3"] = row.get("fraction_csp3", 0.0) or 0.0
    row["n_rotatable_bonds"] = row.get("rotatable_bonds", 0) or 0
    row["num_aromatic_rings"] = row.get("aromatic_rings", 0) or 0
    row["num_h_bond_donors"] = row.get("hbd", 0) or 0
    row["num_h_bond_acceptors"] = row.get("hba", 0) or 0
    row["logd"] = row.get("logd", row.get("logp", 0.0) or 0.0)
    row["log_s"] = row.get("solubility_logS", 0.0) or 0.0
    row["clearance"] = row.get("clearance_ml_min_kg", 0.0) or 0.0
    row["oral_bioavailability"] = (row.get("bioavailability_pct", 0.0) or 0.0) / 100.0
    row["ppb"] = (row.get("ppb_pct", 0.0) or 0.0) / 100.0
    row["caco2_papp"] = 10 ** float(row.get("caco2_logPapp", -10.0) or -10.0)
    return _normalise_engineering_critic_properties(row)


def _normalise_engineering_critic_properties(row: dict) -> dict:
    pains_alerts = row.get("pains_alerts", 0)
    if isinstance(pains_alerts, list):
        row["pains_alerts"] = len(pains_alerts)
    row["pains_alert_count"] = int(row.get("pains_alerts", 0) or 0)
    herg_risk = row.get("herg_risk", 0.0)
    if isinstance(herg_risk, str):
        row["herg_risk"] = {"low": 0.1, "medium": 0.5, "high": 0.9}.get(
            herg_risk.lower(),
            0.0,
        )
    return row


def _full_workflow_critic_properties(state: dict, smiles: str) -> dict:
    properties = {}
    candidate = _candidate_row_for_smiles(state, smiles)
    if candidate:
        properties.update(_candidate_critic_properties(candidate, smiles))
    validation_rows = state.get("validation", {}).get("results", [])
    validation_row = _validation_row_for_smiles(validation_rows, smiles)
    if validation_row:
        properties.update(validation_row)
    properties.update(_srb_critic_properties(state))
    properties.update(_supply_critic_properties(state))
    properties.update(_request_critic_properties(state))
    properties["_critic_blocking_rule_ids"] = list(_FULL_WORKFLOW_BLOCKING_CRITIC_RULE_IDS)
    return _normalise_engineering_critic_properties(properties)


def _candidate_row_for_smiles(state: dict, smiles: str) -> dict:
    candidates = state.get("candidates")
    if not isinstance(candidates, list):
        return {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        candidate_smiles = str(candidate.get("canonical_smiles") or candidate.get("smiles") or "")
        if candidate_smiles == smiles:
            return dict(candidate)
    first = candidates[0] if candidates else {}
    return dict(first) if isinstance(first, dict) else {}


def _candidate_critic_properties(candidate: dict, smiles: str) -> dict:
    row = dict(candidate)
    if not _has_core_critic_properties(row):
        try:
            from mf_chem.predict.engine import MolPredictEngine

            enriched = _engineering_candidate_properties(MolPredictEngine(device_ids=[]), smiles)
            enriched.update(row)
            row = enriched
        except Exception as exc:
            LOGGER.warning("Skipping full workflow critic property enrichment: %s", exc)
    return row


def _has_core_critic_properties(row: dict) -> bool:
    return all(key in row for key in ("mw", "logp", "tpsa", "qed", "sa_score"))


def _validation_row_for_smiles(validation_rows: object, smiles: str) -> dict:
    rows = (
        [row for row in validation_rows if isinstance(row, dict)]
        if isinstance(validation_rows, list)
        else []
    )
    if not rows:
        return {}
    for row in rows:
        row_smiles = str(row.get("canonical_smiles") or row.get("smiles") or "")
        if row_smiles == smiles:
            return dict(row)
    return dict(_best_engineering_validation_row(rows))


def _supply_critic_properties(state: dict) -> dict:
    supply = state.get("supply")
    if not isinstance(supply, dict):
        return {}
    assessment = supply.get("supply_assessment")
    if not isinstance(assessment, dict):
        return {}
    total_blocks = int(assessment.get("total_blocks") or 0)
    available_blocks = int(assessment.get("commercially_available") or 0)
    properties = {
        "critical_material_suppliers": int(assessment.get("supplier_diversity") or 0),
        "estimated_cost_per_gram": float(assessment.get("avg_price_per_gram") or 0.0),
    }
    if total_blocks > 0:
        properties["building_block_availability"] = available_blocks / total_blocks
    return properties


def _srb_critic_properties(state: dict) -> dict:
    srb = state.get("srb")
    if not isinstance(srb, dict):
        return {}
    protocols = srb.get("protocols")
    if not isinstance(protocols, list) or not protocols:
        return {}
    protocol = protocols[0]
    if not isinstance(protocol, dict):
        return {}
    steps = protocol.get("steps")
    properties = {
        "estimated_cost_per_gram": float(protocol.get("total_estimated_cost_usd") or 0.0),
    }
    if isinstance(steps, list):
        properties["synthesis_steps"] = len(steps)
    return properties


def _request_critic_properties(state: dict) -> dict:
    request = state.get("request")
    if not isinstance(request, dict):
        return {}
    properties = {}
    for key in ("isoform_data_count", "kinase_selectivity_ratio", "cns_mpo", "bbb_score"):
        if key in request:
            properties[key] = request[key]
    return properties


def _best_engineering_validation_row(validation_rows: list) -> dict:
    rows = [row for row in validation_rows if isinstance(row, dict)]
    if not rows:
        return {}
    return max(
        rows,
        key=lambda row: (
            float(row.get("composite_score") or 0.0),
            float(row.get("admet_score") or 0.0),
        ),
    )


def _best_engineering_candidate_smiles(state: dict) -> str:
    rows = state.get("validation", {}).get("results", [])
    best = _best_engineering_validation_row(rows if isinstance(rows, list) else [])
    smiles = str(best.get("canonical_smiles") or best.get("smiles") or "")
    if smiles:
        return smiles
    return _first_candidate_smiles(state)


async def _merge_agent_beliefs_into_crg(final_state: dict, run_id: str) -> dict:
    from mf_core.db.repositories import build_shared_crg_repository_from_env

    crg = dict(final_state.get("crg") or {})
    if not run_id:
        return crg
    try:
        repo = build_shared_crg_repository_from_env()
    except Exception as exc:
        LOGGER.debug("Skipping CRG belief merge (repository unavailable): %s", exc)
        return crg
    if repo is None:
        return crg
    try:
        shared_crg = await repo.get_run_crg(run_id)
    except Exception as exc:
        LOGGER.warning("Failed to read shared CRG for run %s: %s", run_id, exc)
        return crg
    shared_beliefs = list(shared_crg.get("beliefs") or [])
    shared_edges = list(shared_crg.get("edges") or [])
    if not shared_beliefs and not shared_edges:
        return crg
    existing_ids = {
        str(b.get("id") or "")
        for b in (crg.get("beliefs") or [])
        if isinstance(b, dict)
    }
    merged_beliefs = list(crg.get("beliefs") or [])
    for belief in shared_beliefs:
        if not isinstance(belief, dict):
            continue
        if str(belief.get("id") or "") not in existing_ids:
            merged_beliefs.append(belief)
    existing_edge_keys = {
        (str(e.get("source_belief_id") or ""), str(e.get("target_belief_id") or ""))
        for e in (crg.get("edges") or [])
        if isinstance(e, dict)
    }
    merged_edges = list(crg.get("edges") or [])
    for edge in shared_edges:
        if not isinstance(edge, dict):
            continue
        key = (str(edge.get("source_belief_id") or ""), str(edge.get("target_belief_id") or ""))
        if key not in existing_edge_keys:
            merged_edges.append(edge)
    merged = dict(crg)
    merged["beliefs"] = merged_beliefs
    merged["edges"] = merged_edges
    merged["version"] = len(merged_beliefs) + len(merged_edges)
    return merged


async def _record_workflow_provenance(final_state: dict) -> None:
    from provenance_svc.main import ProvenanceRecord, create_record

    run_id = str(final_state.get("run_id", ""))
    artifact_id = f"artifact-{_safe_id(run_id)}-workflow-state"
    crg = await _merge_agent_beliefs_into_crg(final_state, run_id)
    if crg:
        crg["provenance_id"] = artifact_id
        final_state["crg"] = crg
    supply = final_state.get("supply") if isinstance(final_state.get("supply"), dict) else {}
    supply_assessment = (
        supply.get("supply_assessment")
        if isinstance(supply.get("supply_assessment"), dict)
        else {}
    )
    srb = final_state.get("srb") if isinstance(final_state.get("srb"), dict) else {}
    metadata = {
        "project_id": str(final_state.get("request", {}).get("project_id") or run_id),
        "run_id": run_id,
        "trace_id": str(final_state.get("trace_id", "")),
        "workflow_scope": str(final_state.get("workflow_scope", "")),
        "status": str(final_state.get("status", "")),
        "history": list(final_state.get("history", [])),
        "candidate_count": len(final_state.get("candidates", []) or []),
        "validation_passed": bool(final_state.get("validation_passed", False)),
        "retrosyn_route_count": len(final_state.get("retrosyn", {}).get("routes", []) or []),
        "supply_feasibility": str(supply_assessment.get("overall_feasibility", "")),
        "srb_protocol_count": len(srb.get("protocols", []) or []),
        "critic_verdict": str(final_state.get("critic", {}).get("verdict", "")),
        "crg": crg,
        "crg_belief_count": len(crg.get("beliefs", []) or []),
        "crg_edge_count": len(crg.get("edges", []) or []),
    }
    parent_ids = list(final_state.get("artifact_ids", []))
    record = await create_record(
        ProvenanceRecord(
            artifact_type="workflow_state",
            artifact_id=artifact_id,
            parent_ids=parent_ids,
            metadata=metadata,
        )
    )
    artifact_ids = list(final_state.get("artifact_ids", []))
    if artifact_id not in artifact_ids:
        artifact_ids.append(artifact_id)
    final_state["artifact_ids"] = artifact_ids
    final_state["provenance"] = {
        "recorded": True,
        "artifact_id": record["artifact_id"],
        "signature": record["signature"],
        "recorded_at": record.get("recorded_at", ""),
    }


def _affinity_quality_gate(request: dict, state: dict) -> dict:
    max_ki = _first_float_request(request, "boltz_max_ki_nm", "max_ki_nm")
    min_pkd = _first_float_request(request, "min_pkd", "boltz_min_pkd")
    if min_pkd is None:
        min_pkd = _extract_min_pkd(str(state.get("nl_input", "")))
    if max_ki is None and min_pkd is not None:
        max_ki = math.pow(10.0, 9.0 - min_pkd)
    return {
        "configured": max_ki is not None,
        "max_ki_nm": max_ki,
        "min_pkd": min_pkd,
    }


def _first_float_request(request: dict, *keys: str) -> float | None:
    for key in keys:
        value = request.get(key)
        if value not in (None, ""):
            return float(value)
    return None


def _extract_min_pkd(text: str) -> float | None:
    match = re.search(r"\bpKd\b\s*(?:>|>=|≥|above|over)\s*(\d+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1))
    return None


def _passes_affinity_gate(row: dict, quality_gate: dict) -> bool:
    max_ki = quality_gate.get("max_ki_nm")
    if max_ki is None:
        return False
    return float(row.get("ki_nm", math.inf)) <= float(max_ki)


def _requested_oracle_level(request: dict) -> int | None:
    for key in ("oracle_level", "max_oracle_level", "validation_oracle_level"):
        value = request.get(key)
        if value not in (None, ""):
            return int(value)
    return None


async def _validate_with_oracle_cascade(
    state: dict,
    candidates: list[dict],
    oracle_level: int,
) -> dict:
    from validation_agent.agent import ValidationAgent

    request = dict(state.get("request", {}) or {})
    agent = ValidationAgent()
    rows = []
    for candidate in candidates:
        smiles = _candidate_smiles(candidate, purpose="validation")
        payload = dict(request)
        payload["project_id"] = str(request.get("project_id") or "")
        payload["run_id"] = str(state.get("run_id", ""))
        payload["smiles"] = smiles
        payload["oracle_level"] = oracle_level
        result = await agent.process(payload)
        status = str(result.get("status") or "")
        overall_passed = bool(result.get("overall_passed", status == "validated"))
        rows.append(
            {
                "smiles": smiles,
                "status": status,
                "overall_passed": overall_passed,
                "max_oracle_level": result.get("max_oracle_level", oracle_level),
                "cascade": dict(result.get("cascade") or {}),
                "upgrade_path": list(result.get("upgrade_path") or []),
            }
        )
    return {
        "passed": any(bool(row.get("overall_passed")) for row in rows),
        "results": rows,
        "validation_mode": "adaptive_oracle_cascade",
        "oracle_level": oracle_level,
    }


def _candidate_smiles(candidate: dict, purpose: str) -> str:
    if not isinstance(candidate, dict):
        raise RuntimeError("candidate entries must be objects")
    smiles = str(candidate.get("canonical_smiles") or candidate.get("smiles") or "")
    if not smiles:
        raise RuntimeError(f"candidate canonical_smiles is required for full workflow {purpose}")
    return smiles


def _first_candidate_smiles(state: dict) -> str:
    candidates = list(state.get("candidates", []) or [])
    if not candidates:
        raise RuntimeError("candidates are required for full workflow synthesis")
    return _candidate_smiles(candidates[0], purpose="synthesis")


def _supply_feasibility(state: dict) -> str:
    supply = state.get("supply")
    if not isinstance(supply, dict):
        return ""
    assessment = supply.get("supply_assessment")
    if not isinstance(assessment, dict):
        return ""
    return str(assessment.get("overall_feasibility") or "").lower()


def _first_retrosyn_route(state: dict) -> dict:
    route = _first_retrosyn_route_or_none(state)
    if route is None:
        raise RuntimeError("retrosyn.routes is required for full workflow synthesis")
    return route


def _first_retrosyn_route_or_none(state: dict) -> dict | None:
    retrosyn = state.get("retrosyn")
    if not isinstance(retrosyn, dict):
        return None
    routes = retrosyn.get("routes")
    if not isinstance(routes, list) or not routes:
        return None
    route = routes[0]
    if not isinstance(route, dict):
        raise RuntimeError("retrosyn route entries must be objects")
    return route


def _unavailable_supply_result(state: dict, reason: str) -> dict:
    return {
        "agent": "supply_agent",
        "status": "assessed",
        "smiles": _first_candidate_smiles(state),
        "skip_reason": reason,
        "supply_assessment": {
            "total_blocks": 0,
            "commercially_available": 0,
            "avg_price_per_gram": 0.0,
            "avg_lead_time_days": 0.0,
            "supplier_diversity": 0,
            "overall_feasibility": "unavailable",
        },
        "block_assessments": [],
    }


def _route_building_blocks(route: dict) -> list[dict]:
    blocks = route.get("building_blocks")
    if isinstance(blocks, list) and blocks:
        return list(blocks)
    extracted = []
    seen = set()
    for step in route.get("steps", []) or []:
        if not isinstance(step, dict):
            continue
        for key in ("building_blocks", "reactants"):
            values = step.get(key)
            if not isinstance(values, list):
                continue
            for value in values:
                smiles = _block_smiles(value)
                if smiles and smiles not in seen:
                    seen.add(smiles)
                    extracted.append({"smiles": smiles})
    if not extracted:
        raise RuntimeError("retrosyn route building_blocks are required for supply assessment")
    return extracted


def _block_smiles(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("smiles") or value.get("building_block_smiles") or "")
    return ""


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "run"


def _run_status(final_state: dict) -> str:
    current = str(final_state.get("status", ""))
    if current == "PAUSED":
        return "paused"
    if current == "ESCALATING":
        return "escalated"
    return "completed"


async def serve_grpc():
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    register_grpc_services(server)
    server.add_insecure_port("[::]:50071")
    await server.start()
    LOGGER.info("Orchestrator gRPC Service running on :50071")
    await server.wait_for_termination()


def register_grpc_services(server) -> None:
    orchestrator_pb2_grpc.add_OrchestratorServiceServicer_to_server(
        OrchestratorGrpcServicer(),
        server,
    )


if __name__ == "__main__":
    import uvicorn

    async def main():
        asyncio.create_task(serve_grpc())
        config = uvicorn.Config(rest_app, host="0.0.0.0", port=8011, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()

    asyncio.run(main())
