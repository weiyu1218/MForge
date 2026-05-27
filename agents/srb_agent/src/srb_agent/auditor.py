"""SRB auditor — provenance event factory."""
from __future__ import annotations

import hashlib
import json
import time


class AuditEvent:
    def __init__(self, actor: str, action: str, run_id: str, payload_summary: dict):
        self.actor = actor
        self.action = action
        self.run_id = run_id
        self.payload_summary = payload_summary
        self.timestamp = time.time()
        raw = json.dumps(payload_summary, sort_keys=True).encode()
        self.content_hash = hashlib.sha256(raw).hexdigest()
        self.signature = None


def make_compile_start_event(run_id: str, smiles: str, route_id: str) -> AuditEvent:
    return AuditEvent(
        actor="SRBAgent",
        action="srb.compile.start",
        run_id=run_id,
        payload_summary={"smiles": smiles, "route_id": route_id},
    )


def make_step_yield_event(run_id: str, step_id: str, reaction_type: str, yield_est: float, uncertainty: float) -> AuditEvent:
    return AuditEvent(
        actor="SRBAgent.yield_estimator",
        action="srb.step.yield_estimated",
        run_id=run_id,
        payload_summary={
            "step_id": step_id,
            "reaction_type": reaction_type,
            "yield_estimate": yield_est,
            "uncertainty": uncertainty,
        },
    )


def make_compile_completed_event(run_id: str, ssp_id: str, total_yield: float, total_cost: float, n_steps: int) -> AuditEvent:
    return AuditEvent(
        actor="SRBAgent",
        action="srb.compile.completed",
        run_id=run_id,
        payload_summary={
            "ssp_id": ssp_id,
            "total_yield": total_yield,
            "total_cost": total_cost,
            "n_steps": n_steps,
        },
    )


def make_xdl_export_event(run_id: str, ssp_id: str, xdl_size_bytes: int) -> AuditEvent:
    return AuditEvent(
        actor="XDLCompiler",
        action="srb.xdl.exported",
        run_id=run_id,
        payload_summary={"ssp_id": ssp_id, "xdl_size_bytes": xdl_size_bytes},
    )
