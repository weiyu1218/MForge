"""UniProt grounding tool."""
from __future__ import annotations

import os

from cig_compiler_svc.domain.tools.http_json import get_json

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"


async def query_uniprot_entry(query: str, limit: int = 5) -> list[dict]:
    data, timestamp = await get_json(
        os.environ.get("UNIPROT_SEARCH_URL", UNIPROT_SEARCH_URL),
        params={
            "query": f"gene_exact:{query} AND organism_id:9606",
            "fields": "accession,protein_name,gene_names,xref_pdb",
            "format": "json",
            "size": limit,
        },
    )

    records = []
    for result in data.get("results", []):
        accession = result.get("primaryAccession")
        if not accession:
            continue
        records.append(
            {
                "accession": accession,
                "protein_name": _protein_name(result),
                "gene_name": _gene_name(result),
                "organism": result.get("organism", {}).get("scientificName"),
                "pdb_ids": _pdb_ids(result),
                "source_timestamp": timestamp,
            }
        )
    return records


def _protein_name(result: dict) -> str:
    description = result.get("proteinDescription", {})
    recommended = description.get("recommendedName", {})
    full_name = recommended.get("fullName", {})
    return full_name.get("value", "")


def _gene_name(result: dict) -> str:
    genes = result.get("genes", [])
    if not genes:
        return ""
    gene_name = genes[0].get("geneName", {})
    return gene_name.get("value", "")


def _pdb_ids(result: dict) -> list[str]:
    references = result.get("uniProtKBCrossReferences", [])
    return [
        reference["id"]
        for reference in references
        if reference.get("database") == "PDB" and reference.get("id")
    ]
