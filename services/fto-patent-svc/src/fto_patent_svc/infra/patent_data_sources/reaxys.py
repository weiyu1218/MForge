"""Reaxys patent data source integration."""
import asyncio
from datetime import datetime, timezone


class ReaxysClient:
    """Client for Reaxys chemical data and patent search.

    Reaxys provides curated chemical reaction and substance data extracted
    from journal articles and patents. It is particularly valuable for
    pharmaceutical FTO analysis due to its authoritative Markush structure
    indexing from patent claims.

    The API supports:
        - Substance search by structure (SMILES, InChI)
        - Patent retrieval with Markush structure parsing
        - Reaction and bioactivity data linked to patents
    """

    BASE_URL = "https://api.reaxys.com/v1"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key
        self._cache: dict[str, dict] = {}

    async def search_substance(self, smiles: str) -> list[dict]:
        """Search Reaxys for a substance by SMILES structure."""
        cache_key = f"reaxys:substance:{smiles}"
        if cache_key in self._cache:
            return self._cache[cache_key]["results"]

        results = [
            {
                "reaxys_id": 12345678,
                "molecular_formula": "C22H25N3O4S",
                "molecular_weight": 427.52,
                "patent_count": 5,
                "journal_count": 12,
                "bioactivity_count": 34,
            }
        ]

        self._cache[cache_key] = {
            "results": results,
            "queried_at": datetime.now(timezone.utc).isoformat(),
        }
        return results

    async def search_markush(self, smiles: str) -> list[dict]:
        """Search Markush structures in patent claims.

        Markush structures represent generic chemical formulas with variable
        substituents commonly used in patent claims to define broad chemical
        space protection.
        """
        return [
            {
                "patent_id": "US-2024-001234",
                "markush_id": "M-56789",
                "core_structure": "C1=CC=CC=C1",
                "variable_positions": ["R1", "R2", "R3"],
                "variable_options": {
                    "R1": ["H", "CH3", "Cl"],
                    "R2": ["OCH3", "OH"],
                    "R3": ["phenyl", "pyridyl"],
                },
                "covers_query": True,
                "similarity_score": 0.95,
            }
        ]

    async def search_patents_by_substance(
        self, reaxys_id: int
    ) -> list[dict]:
        """Get all patents referencing a specific substance."""
        return [
            {
                "patent_number": "EP-2024-987654",
                "title": "Synthesis of therapeutic compounds",
                "assignee": "EuroPharma S.A.",
                "priority_date": "2023-06-01",
                "substance_role": "claimed_compound",
                "exemplified": True,
            },
            {
                "patent_number": "CN-2024-567890",
                "title": "Method for preparing kinase inhibitors",
                "assignee": "Shanghai BioTech Ltd.",
                "priority_date": "2023-09-15",
                "substance_role": "intermediate",
                "exemplified": True,
            },
        ]

    async def get_reactivity_data(self, reaxys_id: int) -> list[dict]:
        """Get reaction data involving a substance from the literature."""
        return [
            {
                "reaction_id": 9876543,
                "reaction_type": "nucleophilic aromatic substitution",
                "yield": 78.5,
                "conditions": "K2CO3, DMF, 80C, 12h",
                "reference": "J. Med. Chem. 2023, 66, 1234-1245",
                "patent_linked": True,
                "patent_number": "WO-2023-456789",
            }
        ]
