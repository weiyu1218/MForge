"""Chemical Reasoning Graph for agent knowledge representation."""
from mf_core.types.crg import CRG, Belief, CRGEdge
import time
import uuid


class ChemicalReasoningGraph:
    """Manages a CRG (Chemical Reasoning Graph) for an agent's beliefs.

    Provides methods to add, update, and query beliefs with evidence
    tracking and versioning.
    """

    def __init__(self, project_id: str = ""):
        self.crg = CRG(project_id=project_id)

    def add_belief(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 0.0,
        source_agent: str = "",
        evidence_ids: list[str] | None = None,
    ) -> Belief:
        """Add a new belief to the graph.

        Args:
            subject: Subject entity of the belief.
            predicate: Predicate describing the relationship.
            obj: Object entity of the belief.
            confidence: Confidence score [0, 1].
            source_agent: Agent that produced this belief.
            evidence_ids: List of supporting evidence references.

        Returns:
            The newly created Belief object.
        """
        belief = Belief(
            id=str(uuid.uuid4()),
            subject=subject,
            predicate=predicate,
            object=obj,
            confidence=confidence,
            source_agent=source_agent,
            evidence_ids=evidence_ids or [],
        )
        self.crg.beliefs.append(belief)
        self.crg.version += 1
        return belief

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str = "supports",
        weight: float = 1.0,
    ) -> CRGEdge:
        """Add a relationship edge between two beliefs.

        Args:
            source_id: ID of the source belief.
            target_id: ID of the target belief.
            relation: Type of relationship (supports, contradicts, etc.).
            weight: Edge weight [-1, 1].

        Returns:
            The newly created CRGEdge.
        """
        edge = CRGEdge(
            source_belief_id=source_id,
            target_belief_id=target_id,
            relation=relation,
            weight=weight,
        )
        self.crg.edges.append(edge)
        self.crg.version += 1
        return edge

    def update_belief(self, belief_id: str, **kwargs) -> Belief | None:
        """Update properties of an existing belief.

        Args:
            belief_id: ID of the belief to update.
            **kwargs: Attributes to update (confidence, predicate, etc.).

        Returns:
            Updated Belief or None if not found.
        """
        for b in self.crg.beliefs:
            if b.id == belief_id:
                for k, v in kwargs.items():
                    if hasattr(b, k):
                        setattr(b, k, v)
                self.crg.version += 1
                return b
        return None

    def query(
        self, subject: str | None = None, predicate: str | None = None
    ) -> list[Belief]:
        """Query beliefs by subject and/or predicate.

        Args:
            subject: Filter by subject (None for any).
            predicate: Filter by predicate (None for any).

        Returns:
            List of matching Belief objects.
        """
        results = []
        for b in self.crg.beliefs:
            if (subject is None or b.subject == subject) and (
                predicate is None or b.predicate == predicate
            ):
                results.append(b)
        return results

    def to_crg(self) -> CRG:
        """Export the current state as a CRG model.

        Returns:
            The underlying CRG object.
        """
        return self.crg
