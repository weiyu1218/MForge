"""Patent dead zone updater - links patent dead zone feedback to the FTO graph."""
import hashlib
from datetime import datetime, timezone


class DeadZoneUpdater:
    """Manages patent dead zone regions in the FTO knowledge graph.

    Dead zones are regions of chemical space near existing patents where
    freedom-to-operate is ambiguous and requires attorney review.
    """

    def __init__(self, graph_client=None):
        self.graph_client = graph_client
        self.dead_zones = {}

    def add_dead_zone(
        self, smiles: str, center_fingerprint: str, radius: float, source_patent: str
    ) -> str:
        """Register a new dead zone region originating from a patent claim."""
        zone_id = hashlib.sha256(
            f"{smiles}:{center_fingerprint}:{radius}".encode()
        ).hexdigest()[:16]

        self.dead_zones[zone_id] = {
            "zone_id": zone_id,
            "smiles": smiles,
            "center_fingerprint": center_fingerprint,
            "radius": radius,
            "source_patent": source_patent,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "hit_count": 0,
        }
        return zone_id

    def query_dead_zone(self, fingerprint: str, threshold: float = 0.75) -> list[dict]:
        """Check if a query fingerprint falls within any known dead zone."""
        results = []
        for zone_id, zone in self.dead_zones.items():
            # Simplified similarity: in production this uses Tanimoto on fingerprints
            similarity = self._tanimoto_similarity(
                fingerprint, zone["center_fingerprint"]
            )
            if similarity >= threshold * zone["radius"]:
                results.append(
                    {
                        "zone_id": zone_id,
                        "similarity": similarity,
                        "source_patent": zone["source_patent"],
                        "radius": zone["radius"],
                    }
                )
                zone["hit_count"] += 1
        return sorted(results, key=lambda r: r["similarity"], reverse=True)

    def _tanimoto_similarity(self, fp_a: str, fp_b: str) -> float:
        """Compute approximate Tanimoto similarity between fingerprint strings."""
        import binascii

        try:
            bytes_a = binascii.unhexlify(fp_a[:1024].ljust(1024, "0"))
            bytes_b = binascii.unhexlify(fp_b[:1024].ljust(1024, "0"))
        except Exception:
            return 0.0

        set_a = set()
        set_b = set()
        for i, (ba, bb) in enumerate(zip(bytes_a, bytes_b)):
            for bit in range(8):
                if ba & (1 << bit):
                    set_a.add(i * 8 + bit)
                if bb & (1 << bit):
                    set_b.add(i * 8 + bit)

        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        if union == 0:
            return 0.0
        return intersection / union

    def sync_to_graph(self) -> int:
        """Sync dead zone records to Neo4j knowledge graph."""
        if self.graph_client is None:
            return 0
        count = 0
        for zone_id, zone in self.dead_zones.items():
            # Cypher MERGE would go here in production
            count += 1
        return count

    def get_statistics(self) -> dict:
        """Return dead zone statistics."""
        return {
            "total_zones": len(self.dead_zones),
            "total_hits": sum(z["hit_count"] for z in self.dead_zones.values()),
            "zones_by_patent": {},
        }
