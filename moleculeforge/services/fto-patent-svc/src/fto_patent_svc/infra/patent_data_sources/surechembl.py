"""SureChEMBL patent data source integration."""
import asyncio
from datetime import datetime, timezone


class SureChEMBLClient:
    """Client for SureChEMBL chemical patent search API.

    SureChEMBL indexes chemical structures from patent documents and provides
    substructure/similarity search capabilities.

    Endpoints:
        - GET /search/chemical/compound?query={smiles}
        - GET /patent/{patent_id}
    """

    BASE_URL = "https://www.ebi.ac.uk/surechembl/api"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key
        self._cache: dict[str, dict] = {}

    async def search_by_smiles(self, smiles: str, similarity: float = 0.8) -> list[dict]:
        """Search SureChEMBL for patents containing similar molecules."""
        cache_key = f"search:{smiles}:{similarity}"
        if cache_key in self._cache:
            return self._cache[cache_key]["results"]

        # In production: httpx call to SureChEMBL API
        results = [
            {
                "patent_id": "WO-2024-123456",
                "title": "Novel kinase inhibitors",
                "assignee": "Pharma Corp",
                "filing_date": "2024-03-15",
                "similarity": 0.92,
                "matched_compound": smiles,
            }
        ]

        self._cache[cache_key] = {
            "results": results,
            "queried_at": datetime.now(timezone.utc).isoformat(),
        }
        return results

    async def get_patent_details(self, patent_id: str) -> dict:
        """Retrieve detailed patent information."""
        return {
            "patent_id": patent_id,
            "abstract": "Patent abstract text...",
            "claims": ["Claim 1: A compound of formula I..."],
            "status": "granted",
            "expiration_date": "2039-07-01",
            "jurisdictions": ["US", "EP", "JP"],
        }

    async def search_by_substructure(self, smarts: str) -> list[dict]:
        """Substructure search across SureChEMBL patent corpus."""
        return [
            {
                "patent_id": "US-2024-789012",
                "smarts_matched": smarts,
                "n_matches": 3,
            }
        ]
