"""Reasoning workbench pipeline: NL -> objectives -> generation -> scoring -> novelty.

This module keeps the `/v1/reason/*` UI workflow stable. The CoreArchitecture
v2 orchestrator path is `services/orchestrator-svc` and its LangGraph workflow.
"""
from __future__ import annotations

import asyncio
import sqlite3
import time
import uuid
from datetime import datetime, timezone

from mf_chem.predict import get_default_engine
from mf_core.db import store


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Constraint filtering
# ---------------------------------------------------------------------------


def _passes_constraints(props: dict, constraints: dict) -> tuple[bool, list[str]]:
    """Apply numeric + SMARTS constraints from objectives. Returns (ok, reasons)."""
    reasons: list[str] = []
    ok = True

    def in_range(val, rng):
        lo, hi = rng if isinstance(rng, list) else (None, None)
        if lo is not None and val < lo:
            return False
        if hi is not None and val > hi:
            return False
        return True

    if "molecular_weight" in constraints and isinstance(constraints["molecular_weight"], list):
        if not in_range(props["molecular_weight"], constraints["molecular_weight"]):
            ok = False
            reasons.append("molecular_weight out of range")
    if "logp" in constraints and isinstance(constraints["logp"], list):
        if not in_range(props["logp"], constraints["logp"]):
            ok = False
            reasons.append("logp out of range")
    if (cap := constraints.get("hbd_max")) is not None and props["hbd"] > cap:
        ok = False
        reasons.append("hbd exceeds limit")
    if (cap := constraints.get("hba_max")) is not None and props["hba"] > cap:
        ok = False
        reasons.append("hba exceeds limit")
    if (cap := constraints.get("tpsa_max")) is not None and props["tpsa"] > cap:
        ok = False
        reasons.append("tpsa exceeds limit")
    if (
        (cap := constraints.get("rotatable_bonds_max")) is not None
        and props["rotatable_bonds"] > cap
    ):
        ok = False
        reasons.append("rotatable_bonds exceeds limit")
    if (floor := constraints.get("qed_min")) is not None and (props.get("qed") or 0.0) < floor:
        ok = False
        reasons.append("qed below minimum")
    if (cap := constraints.get("sa_max")) is not None and (props.get("sa_score") or 99) > cap:
        ok = False
        reasons.append("sa_score exceeds limit")
    return ok, reasons


def _passes_smarts(smiles: str, constraints: dict) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    try:
        from rdkit import Chem
    except ImportError:
        return True, reasons
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, ["invalid"]
    for sm in constraints.get("must_include_smarts") or []:
        patt = Chem.MolFromSmarts(sm)
        if patt is not None and not mol.HasSubstructMatch(patt):
            reasons.append(f"missing required group {sm}")
    for sm in constraints.get("must_exclude_smarts") or []:
        patt = Chem.MolFromSmarts(sm)
        if patt is not None and mol.HasSubstructMatch(patt):
            reasons.append(f"contains forbidden group {sm}")
    return len(reasons) == 0, reasons


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _seed_pool(objectives: dict, mult: int = 4) -> list[str]:
    """Build candidate SMILES via RDKit-Random over hint scaffolds.

    Strategy:
      1. Always include the original scaffold hints unmutated as anchors.
      2. Pad the pool with mutated variants up to (n_samples * mult).
      3. Add small set of templates that already match common SMARTS hints
         (Michael acceptor, sulfonamide, amide) so include-filters retain
         meaningful candidates.
    """
    import random

    from mf_generators.rdkit_random.mutator import random_mutate

    n = max(8, objectives.get("n_samples", 24) * mult)
    rng = random.Random(objectives.get("seed", 42))
    base = list(objectives.get("scaffold_hints") or [])
    # Templates that satisfy common required SMARTS
    smarts_templates = {
        "C=CC(=O)": [  # Michael acceptor warhead
            "C=CC(=O)Nc1ccccc1",
            "C=CC(=O)Nc1ccc(F)cc1",
            "C=CC(=O)N1CCNCC1",
            "C=CC(=O)NC[C@H](N)c1ccccc1",
            "C=CC(=O)Nc1cc(N)ncn1",
        ],
        "S(=O)(=O)N": [  # sulfonamide
            "Nc1ccc(S(N)(=O)=O)cc1",
            "Cc1ccc(S(N)(=O)=O)cc1",
            "O=S(=O)(N)c1ccncc1",
            "O=S(=O)(N)c1ccc(C(=O)O)cc1",
        ],
        "C(=O)N": [  # amide
            "CC(=O)Nc1ccccc1",
            "O=C(N)c1ccncc1",
            "O=C(N)c1ccc(O)cc1",
            "CC(=O)NCC(=O)O",
        ],
        "C(=O)O": [  # carboxylic acid
            "OC(=O)c1ccccc1",
            "OC(=O)c1ccc(O)cc1",
            "OC(=O)CC(=O)O",
        ],
        "C(F)(F)F": [
            "FC(F)(F)c1ccccc1",
            "FC(F)(F)Cn1cnc2c1ccccc2",
        ],
    }
    constraints = objectives.get("constraints") or {}
    for s in (constraints.get("must_include_smarts") or []):
        for cand in smarts_templates.get(s, []):
            if cand not in base:
                base.append(cand)
    if not base:
        # Default chemotype palette covering aromatic, aliphatic, polar templates
        base = [
            "CC(=O)Oc1ccccc1C(=O)O", "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
            "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "CC(=O)Nc1ccc(O)cc1",
            "Nc1ccc(S(N)(=O)=O)cc1", "OC(=O)c1ccccc1O",
            "c1ccncc1", "c1ccc2ncncc2c1",
            "C[C@H](N)Cc1ccccc1", "CCN(CC)CC",
        ]
    # Original anchors first
    out: list[str] = list(base)
    while len(out) < n:
        tmpl = rng.choice(base)
        cand = random_mutate(tmpl, n_mutations=rng.randint(1, 2)) or tmpl
        out.append(cand)
    return out


# ---------------------------------------------------------------------------
# Reasoning orchestrator
# ---------------------------------------------------------------------------


class ReasoningPipeline:
    """Orchestrates NL → CIG → generate → score → filter → novelty → rank.

    Runs the entire pipeline in a single asyncio task. Subscribers register
    via `subscribe()` to receive every reasoning step as it happens. After
    completion the run + steps + results are persisted in SQLite.
    """

    _STAGE_ORDER = (
        "nl_parse",
        "objectives",
        "generation",
        "encoding",
        "scoring",
        "constraint_filter",
        "novelty",
        "ranking",
        "summary",
    )

    def __init__(self) -> None:
        self._runs: dict[str, dict] = {}
        self._listeners: dict[str, list[asyncio.Queue]] = {}

    # ------- public lifecycle -------

    def submit(self, intent: str, project_id: str | None = None,
               extra: dict | None = None) -> str:
        run_id = f"run-{uuid.uuid4().hex[:10]}"
        # Auto-create the project so FK constraints don't blow up on first run.
        if project_id:
            try:
                store.insert_project(
                    project_id=project_id,
                    name=project_id,
                    description="auto-created by reasoning pipeline",
                    created_at=_now(),
                )
            except sqlite3.IntegrityError as exc:
                message = str(exc).upper()
                if "UNIQUE" not in message and "PRIMARY KEY" not in message:
                    raise
        self._runs[run_id] = {
            "run_id": run_id,
            "project_id": project_id,
            "intent": intent,
            "status": "queued",
            "created_at": _now(),
            "extra": extra or {},
            "steps": [],
            "results": [],
            "objectives": None,
        }
        store.insert_run({
            "run_id": run_id, "project_id": project_id, "intent": intent,
            "objectives": {}, "status": "queued", "created_at": _now(),
        })
        asyncio.create_task(self._run(run_id))
        return run_id

    def get(self, run_id: str) -> dict | None:
        # Prefer in-memory snapshot, fall back to DB.
        if run_id in self._runs:
            return self._runs[run_id]
        run = store.get_run(run_id)
        if not run:
            return None
        run["steps"] = store.get_reasoning_steps(run_id)
        run["results"] = store.get_run_results(run_id)
        return run

    def list_runs(self, limit: int = 50) -> list[dict]:
        return store.list_runs(limit=limit)

    async def subscribe(self, run_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._listeners.setdefault(run_id, []).append(q)
        # Replay anything already produced so late subscribers are caught up.
        for step in self._runs.get(run_id, {}).get("steps", []):
            await q.put({"type": "step", **step})
        return q

    def _broadcast(self, run_id: str, event: dict) -> None:
        for q in self._listeners.get(run_id, []):
            q.put_nowait(event)

    # ------- main run -------

    async def _run(self, run_id: str) -> None:
        run = self._runs[run_id]
        run["status"] = "running"
        store.update_run(run_id, status="running")
        try:
            await self._stage_nl(run_id)
            await self._stage_objectives(run_id)
            await self._stage_generation(run_id)
            await self._stage_scoring(run_id)
            await self._stage_filter(run_id)
            await self._stage_novelty(run_id)
            await self._stage_ranking(run_id)
            await self._stage_summary(run_id)
        except Exception as e:  # noqa: BLE001
            run["status"] = "failed"
            run["error"] = f"{type(e).__name__}: {e}"
            self._emit_step(run_id, "summary", "Run failed",
                            f"{type(e).__name__}: {e}", payload={})
            store.update_run(run_id, status="failed", finished_at=_now())
        finally:
            self._broadcast(run_id, {"type": "done", "run_id": run_id})

    # ------- stages -------

    async def _stage_nl(self, run_id: str) -> None:
        run = self._runs[run_id]
        from nl2obj.parser import parse
        objectives = parse(run["intent"])
        run["objectives"] = objectives
        store.update_run(run_id, objectives=objectives)
        self._emit_step(
            run_id, "nl_parse",
            "Natural-language parsing",
            f"Recognised {len(objectives['tokens'])} structured tokens.",
            payload={"tokens": objectives["tokens"], "objectives": objectives},
        )

    async def _stage_objectives(self, run_id: str) -> None:
        run = self._runs[run_id]
        obj = run["objectives"]
        target_str = ", ".join(obj["targets"]) or "(none)"
        ind_str = ", ".join(obj["indications"]) or "(none)"
        c = obj["constraints"]
        cons_lines = []
        if isinstance(c.get("molecular_weight"), list):
            cons_lines.append(f"MW range = {c['molecular_weight']}")
        if isinstance(c.get("logp"), list):
            cons_lines.append(f"logP range = {c['logp']}")
        for k in ("hbd_max", "hba_max", "tpsa_max", "qed_min", "sa_max", "rotatable_bonds_max"):
            if c.get(k) is not None:
                cons_lines.append(f"{k} = {c[k]}")
        for s in c.get("must_include_smarts") or []:
            cons_lines.append(f"must include SMARTS: {s}")
        for s in c.get("must_exclude_smarts") or []:
            cons_lines.append(f"must exclude SMARTS: {s}")
        detail = (
            f"task = {obj['task']} · targets = {target_str} · areas = {ind_str} · "
            f"priorities = {', '.join(obj['objectives_priority'])} · n = {obj['n_samples']}"
        )
        self._emit_step(
            run_id, "objectives",
            "Objective specification (CIG)",
            detail,
            payload={"objectives": obj, "constraints_human": cons_lines},
        )

    async def _stage_generation(self, run_id: str) -> None:
        run = self._runs[run_id]
        obj = run["objectives"]
        seeds = _seed_pool(obj)
        run["seed_pool"] = seeds
        # Run a tiny RDKit canonicalisation pass for the trace
        from rdkit import Chem
        unique = []
        seen = set()
        for s in seeds:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue
            cs = Chem.MolToSmiles(mol, canonical=True)
            if cs in seen:
                continue
            seen.add(cs)
            unique.append(s)
        run["unique_pool"] = unique
        self._emit_step(
            run_id, "generation",
            "Candidate generation",
            f"Sampled {len(seeds)} candidates from {len(obj['scaffold_hints']) or 'default'} "
            f"scaffold(s); {len(unique)} valid + de-duplicated.",
            payload={
                "n_sampled": len(seeds),
                "n_unique": len(unique),
                "scaffold_seeds": obj["scaffold_hints"],
                "examples": unique[:5],
            },
        )

    async def _stage_scoring(self, run_id: str) -> None:
        run = self._runs[run_id]
        engine = get_default_engine()
        loop = asyncio.get_event_loop()
        t0 = time.time()
        results = await loop.run_in_executor(
            None, engine.predict_batch, run["unique_pool"],
        )
        elapsed = time.time() - t0
        run["scored"] = [r.to_dict() for r in results if r.valid]
        self._emit_step(
            run_id, "scoring",
            "GPU multi-device scoring",
            f"Scored {len(run['scored'])} valid molecules in {elapsed*1000:.0f} ms "
            f"across {len(engine.devices)} device(s).",
            payload={
                "devices": engine.devices,
                "n_valid": len(run["scored"]),
                "n_invalid": len(run["unique_pool"]) - len(run["scored"]),
                "elapsed_ms": int(elapsed * 1000),
            },
        )

    async def _stage_filter(self, run_id: str) -> None:
        run = self._runs[run_id]
        constraints = run["objectives"]["constraints"]
        kept: list[dict] = []
        rejections: list[dict] = []
        for r in run["scored"]:
            ok_num, num_reasons = _passes_constraints(r, constraints)
            ok_smarts, smarts_reasons = _passes_smarts(r["canonical_smiles"], constraints)
            reasons = num_reasons + smarts_reasons
            if ok_num and ok_smarts:
                kept.append(r)
            else:
                rejections.append({
                    "smiles": r["canonical_smiles"],
                    "reasons": reasons,
                })
        run["kept"] = kept
        run["rejections"] = rejections
        self._emit_step(
            run_id, "constraint_filter",
            "Constraint screening",
            f"{len(kept)} kept · {len(rejections)} rejected. "
            "Filter applied numeric ranges (MW/logP/HBD/HBA/TPSA/QED/SA), "
            "and SMARTS include/exclude rules.",
            payload={
                "n_kept": len(kept),
                "n_rejected": len(rejections),
                "rejection_examples": rejections[:5],
                "kept_examples": [r["canonical_smiles"] for r in kept[:5]],
            },
        )

    async def _stage_novelty(self, run_id: str) -> None:
        run = self._runs[run_id]
        novel: list[dict] = []
        known: list[dict] = []
        for r in run["kept"]:
            match = store.lookup_known(r.get("inchi_key"), r["canonical_smiles"])
            entry = dict(r)
            if match:
                entry["is_novel"] = False
                entry["known_match"] = {
                    "name": match["name"],
                    "drugbank_id": match.get("drugbank_id"),
                    "indications": match.get("indications"),
                    "target": match.get("target"),
                }
                known.append(entry)
            else:
                entry["is_novel"] = True
                entry["known_match"] = None
                novel.append(entry)
        run["novel"] = novel
        run["known"] = known
        self._emit_step(
            run_id, "novelty",
            "Novelty assessment",
            f"{len(novel)} novel · {len(known)} known reference matches.",
            payload={
                "n_novel": len(novel),
                "n_known": len(known),
                "known_hits": [
                    {"smiles": e["canonical_smiles"], "name": e["known_match"]["name"]}
                    for e in known[:5]
                ],
            },
        )

    async def _stage_ranking(self, run_id: str) -> None:
        run = self._runs[run_id]
        priority = run["objectives"]["objectives_priority"]

        def utility(r: dict) -> float:
            score = 0.0
            if "qed" in priority:
                score += 1.0 * float(r.get("qed") or 0.0)
            if "sa" in priority:
                score += 0.6 * (10.0 - float(r.get("sa_score") or 10.0)) / 10.0
            if "logp" in priority:
                score -= 0.3 * abs(float(r.get("logp") or 5.0) - 2.5) / 5.0
            if "potency" in priority:
                score += 0.4 * float(r.get("qed") or 0.0)
            if "safety" in priority:
                score += 0.3 * float(r.get("admet", {}).get("herg_ic50_uM", 0.0)) / 20.0
            if "bioavailability" in priority:
                score += 0.4 * float(r.get("admet", {}).get("bioavailability_pct", 0.0)) / 100.0
            if "selectivity" in priority:
                score += 0.2 * float(r.get("qed") or 0.0)
            return score

        all_kept = run["novel"] + run["known"]
        # Pareto on (qed, -sa, -|logp - 2.5|)
        objs = [
            (
                float(r.get("qed") or 0.0),
                -float(r.get("sa_score") or 10.0),
                -abs(float(r.get("logp") or 5.0) - 2.5),
            )
            for r in all_kept
        ]
        is_pareto = [True] * len(all_kept)
        for i, oi in enumerate(objs):
            for j, oj in enumerate(objs):
                if i == j:
                    continue
                if all(oj[k] >= oi[k] for k in range(3)) and any(oj[k] > oi[k] for k in range(3)):
                    is_pareto[i] = False
                    break
        for r, p in zip(all_kept, is_pareto):
            r["pareto_optimal"] = bool(p)
        ranked = sorted(all_kept, key=utility, reverse=True)
        for i, r in enumerate(ranked, start=1):
            r["rank"] = i
        run["ranked"] = ranked
        self._emit_step(
            run_id, "ranking",
            "Multi-objective ranking",
            f"Sorted by weighted utility over priorities {priority}; "
            f"{sum(is_pareto)} Pareto-optimal candidate(s).",
            payload={
                "priority": priority,
                "n_pareto": int(sum(is_pareto)),
                "top": [r["canonical_smiles"] for r in ranked[:5]],
            },
        )

    async def _stage_summary(self, run_id: str) -> None:
        run = self._runs[run_id]
        ranked = run["ranked"]
        n_novel = sum(1 for r in ranked if r.get("is_novel"))
        n_known = len(ranked) - n_novel
        # Persist results to DB
        rows = []
        for r in ranked:
            rows.append({
                "rank": r["rank"],
                "smiles": r["smiles"],
                "canonical_smiles": r["canonical_smiles"],
                "inchi_key": r.get("inchi_key"),
                "is_novel": r.get("is_novel", True),
                "known_match": r.get("known_match"),
                "properties": {
                    k: r.get(k) for k in (
                        "molecular_weight", "logp", "tpsa", "hbd", "hba",
                        "rotatable_bonds", "aromatic_rings", "fraction_csp3",
                        "qed", "sa_score", "lipinski_violations", "formula",
                        "drug_likeness", "admet", "humu_embedding_norm",
                        "humu_embedding_dim", "device",
                    )
                },
                "pareto_optimal": r.get("pareto_optimal", False),
                "composite_score": r.get("composite_score"),
            })
        store.insert_run_results(run_id, rows)
        engine = get_default_engine()
        store.update_run(
            run_id,
            status="completed",
            finished_at=_now(),
            summary=(
                f"{len(ranked)} candidate(s) ranked · {n_novel} novel · {n_known} known. "
                f"Top: {ranked[0]['canonical_smiles'] if ranked else '—'}"
            ),
            devices_used=engine.devices,
            n_candidates=len(ranked),
            n_novel=n_novel,
            n_known=n_known,
        )
        run["status"] = "completed"
        run["finished_at"] = _now()
        run["results"] = rows
        self._emit_step(
            run_id, "summary",
            "Run complete",
            f"{len(ranked)} ranked candidate(s) — {n_novel} novel, {n_known} known.",
            payload={"n_candidates": len(ranked), "n_novel": n_novel, "n_known": n_known},
        )

    # ------- helpers -------

    def _emit_step(self, run_id: str, stage: str, title: str,
                   detail: str | None, payload: dict | None) -> None:
        run = self._runs[run_id]
        index = len(run["steps"])
        ts = _now()
        step = {
            "step_index": index,
            "stage": stage,
            "title": title,
            "detail": detail,
            "payload": payload or {},
            "timestamp": ts,
        }
        run["steps"].append(step)
        store.insert_reasoning_step(
            run_id=run_id, step_index=index, stage=stage, title=title,
            detail=detail, payload=payload, timestamp=ts,
        )
        self._broadcast(run_id, {"type": "step", **step})


_default_pipeline: ReasoningPipeline | None = None


def get_pipeline() -> ReasoningPipeline:
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = ReasoningPipeline()
    return _default_pipeline
