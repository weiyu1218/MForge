"""GraphRepository — Neo4j-backed CRG operations."""
from __future__ import annotations

from typing import Any


class GraphRepository:
    """Repository for chemical reasoning graph operations via Neo4j."""

    def __init__(self, driver: Any) -> None:
        self.driver = driver

    async def write_transforms_to(
        self, from_inchikey: str, to_inchikey: str, via: str, confidence: float
    ) -> None:
        query = (
            "MERGE (a:Molecule {inchikey: $from_ik}) "
            "MERGE (b:Molecule {inchikey: $to_ik}) "
            "MERGE (a)-[r:TRANSFORMS_TO {via: $via, confidence: $confidence}]->(b)"
        )
        async with self.driver.session() as session:
            await session.run(
                query,
                from_ik=from_inchikey,
                to_ik=to_inchikey,
                via=via,
                confidence=confidence,
            )

    async def write_binds_to(
        self,
        inchikey: str,
        uniprot_id: str,
        source: str,
        affinity: float,
        method: str,
    ) -> None:
        query = (
            "MERGE (a:Molecule {inchikey: $inchikey}) "
            "MERGE (p:Protein {uniprot_id: $uniprot_id}) "
            "MERGE (a)-[r:BINDS_TO {source: $source, affinity: $affinity, method: $method}]->(p)"
        )
        async with self.driver.session() as session:
            await session.run(
                query,
                inchikey=inchikey,
                uniprot_id=uniprot_id,
                source=source,
                affinity=affinity,
                method=method,
            )

    async def write_produced(
        self, run_id: str, inchikey: str, agent: str, timestamp: str = ""
    ) -> None:
        query = (
            "MERGE (r:Run {id: $run_id, agent: $agent, timestamp: $timestamp}) "
            "MERGE (m:Molecule {inchikey: $inchikey}) "
            "MERGE (r)-[rel:PRODUCED]->(m)"
        )
        async with self.driver.session() as session:
            await session.run(
                query,
                run_id=run_id,
                inchikey=inchikey,
                agent=agent,
                timestamp=timestamp,
            )

    async def write_has_belief(
        self,
        inchikey: str,
        belief_id: str,
        oracle: str,
        value: float,
        uncertainty: float = 0.0,
        created_at: str = "",
    ) -> None:
        query = (
            "MERGE (m:Molecule {inchikey: $inchikey}) "
            "MERGE (b:Belief {id: $belief_id, oracle: $oracle, created_at: $created_at}) "
            "MERGE (m)-[r:HAS_BELIEF {value: $value, uncertainty: $uncertainty}]->(b)"
        )
        async with self.driver.session() as session:
            await session.run(
                query,
                inchikey=inchikey,
                belief_id=belief_id,
                oracle=oracle,
                value=value,
                uncertainty=uncertainty,
                created_at=created_at,
            )

    async def write_workflow_belief(
        self,
        *,
        project_id: str,
        run_id: str,
        belief_id: str,
        subject: str,
        predicate: str,
        object_value: str,
        confidence: float,
        source_agent: str,
        timestamp_ns: int,
        evidence_ids: list[str],
    ) -> None:
        query = (
            "MERGE (r:Run {id: $run_id}) "
            "SET r.project_id = $project_id "
            "MERGE (b:Belief {id: $belief_id}) "
            "SET b.subject = $subject, "
            "b.predicate = $predicate, "
            "b.object = $object_value, "
            "b.confidence = $confidence, "
            "b.source_agent = $source_agent, "
            "b.timestamp_ns = $timestamp_ns, "
            "b.evidence_ids = $evidence_ids "
            "MERGE (r)-[:HAS_WORKFLOW_BELIEF]->(b)"
        )
        async with self.driver.session() as session:
            await session.run(
                query,
                project_id=project_id,
                run_id=run_id,
                belief_id=belief_id,
                subject=subject,
                predicate=predicate,
                object_value=object_value,
                confidence=confidence,
                source_agent=source_agent,
                timestamp_ns=timestamp_ns,
                evidence_ids=evidence_ids,
            )

    async def write_crg_edge(
        self,
        *,
        source_belief_id: str,
        target_belief_id: str,
        relation: str,
        weight: float,
    ) -> None:
        query = (
            "MERGE (s:Belief {id: $source_belief_id}) "
            "MERGE (t:Belief {id: $target_belief_id}) "
            "MERGE (s)-[r:CRG_EDGE {relation: $relation}]->(t) "
            "SET r.weight = $weight"
        )
        async with self.driver.session() as session:
            await session.run(
                query,
                source_belief_id=source_belief_id,
                target_belief_id=target_belief_id,
                relation=relation,
                weight=weight,
            )

    async def get_run_crg(self, run_id: str) -> dict[str, Any]:
        query = (
            "MATCH (r:Run {id: $run_id}) "
            "OPTIONAL MATCH (r)-[:HAS_WORKFLOW_BELIEF]->(b:Belief) "
            "OPTIONAL MATCH (b)-[edge:CRG_EDGE]->(target:Belief) "
            "RETURN r.project_id AS project_id, "
            "collect(DISTINCT {"
            "id: b.id, "
            "subject: b.subject, "
            "predicate: b.predicate, "
            "object: b.object, "
            "confidence: b.confidence, "
            "source_agent: b.source_agent, "
            "timestamp_ns: b.timestamp_ns, "
            "evidence_ids: b.evidence_ids"
            "}) AS beliefs, "
            "collect(DISTINCT {"
            "source_belief_id: b.id, "
            "target_belief_id: target.id, "
            "relation: edge.relation, "
            "weight: edge.weight"
            "}) AS edges"
        )
        async with self.driver.session() as session:
            result = await session.run(query, run_id=run_id)
            row = await result.single()
        if not row:
            return {"project_id": "", "beliefs": [], "edges": [], "version": 0}
        beliefs = [_belief_dict(item) for item in row.get("beliefs", [])]
        edges = [_crg_edge_dict(item) for item in row.get("edges", [])]
        beliefs = [item for item in beliefs if item]
        edges = [item for item in edges if item]
        return {
            "project_id": str(row.get("project_id") or ""),
            "beliefs": beliefs,
            "edges": edges,
            "version": len(beliefs) + len(edges),
        }

    async def write_artifact(
        self,
        *,
        artifact_id: str,
        artifact_type: str,
        project_id: str,
        run_id: str,
        trace_id: str,
        recorded_at: str,
        signature_type: str,
    ) -> None:
        query = (
            "MERGE (a:Artifact {id: $artifact_id}) "
            "SET a.type = $artifact_type, "
            "a.project_id = $project_id, "
            "a.run_id = $run_id, "
            "a.trace_id = $trace_id, "
            "a.recorded_at = $recorded_at, "
            "a.signature_type = $signature_type "
            "FOREACH (_ IN CASE WHEN $run_id = '' THEN [] ELSE [1] END | "
            "MERGE (r:Run {id: $run_id}) "
            "MERGE (r)-[:PRODUCED_ARTIFACT]->(a))"
        )
        async with self.driver.session() as session:
            await session.run(
                query,
                artifact_id=artifact_id,
                artifact_type=artifact_type,
                project_id=project_id,
                run_id=run_id,
                trace_id=trace_id,
                recorded_at=recorded_at,
                signature_type=signature_type,
            )

    async def write_artifact_parent(self, parent_id: str, child_id: str) -> None:
        query = (
            "MERGE (p:Artifact {id: $parent_id}) "
            "MERGE (c:Artifact {id: $child_id}) "
            "MERGE (p)-[:PARENT_OF]->(c)"
        )
        async with self.driver.session() as session:
            await session.run(query, parent_id=parent_id, child_id=child_id)

    async def get_artifact_chain_ids(self, artifact_id: str) -> list[str]:
        query = (
            "MATCH (a:Artifact)-[:PARENT_OF*0..]->(:Artifact {id: $artifact_id}) "
            "RETURN DISTINCT a.id AS artifact_id, a.recorded_at AS recorded_at "
            "ORDER BY recorded_at, artifact_id"
        )
        async with self.driver.session() as session:
            result = await session.run(query, artifact_id=artifact_id)
            rows = await result.data()
        return [str(row["artifact_id"]) for row in rows if row.get("artifact_id")]

    async def count_artifact_children(self, artifact_id: str) -> int:
        query = (
            "MATCH (:Artifact {id: $artifact_id})-[:PARENT_OF]->(child:Artifact) "
            "RETURN count(child) AS children"
        )
        async with self.driver.session() as session:
            result = await session.run(query, artifact_id=artifact_id)
            row = await result.single()
        return int(row["children"]) if row else 0

    async def count_artifacts(self) -> int:
        async with self.driver.session() as session:
            result = await session.run("MATCH (a:Artifact) RETURN count(a) AS artifacts")
            row = await result.single()
        return int(row["artifacts"]) if row else 0


def _belief_dict(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict) or not item.get("id"):
        return {}
    return {
        "id": str(item.get("id") or ""),
        "subject": str(item.get("subject") or ""),
        "predicate": str(item.get("predicate") or ""),
        "object": str(item.get("object") or ""),
        "confidence": float(item.get("confidence") or 0.0),
        "source_agent": str(item.get("source_agent") or ""),
        "timestamp_ns": int(item.get("timestamp_ns") or 0),
        "evidence_ids": list(item.get("evidence_ids") or []),
    }


def _crg_edge_dict(item: Any) -> dict[str, Any]:
    if (
        not isinstance(item, dict)
        or not item.get("source_belief_id")
        or not item.get("target_belief_id")
    ):
        return {}
    return {
        "source_belief_id": str(item.get("source_belief_id") or ""),
        "target_belief_id": str(item.get("target_belief_id") or ""),
        "relation": str(item.get("relation") or ""),
        "weight": float(item.get("weight") or 0.0),
    }
