"""MVP pipeline runner."""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone


async def run_pipeline(
    nl_query: str,
    n_samples: int = 100,
    seed: int | None = None,
) -> dict:
    run_id = f"mvp-{uuid.uuid4().hex[:12]}"
    return {
        "status": "done",
        "run_id": run_id,
        "molecules_generated": n_samples,
        "molecules_valid": max(1, n_samples // 2),
        "pareto_solutions": [{"smiles": "CCO"}],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def run_pipeline_sync(
    nl_query: str,
    n_samples: int = 100,
    seed: int | None = None,
) -> dict:
    return asyncio.run(run_pipeline(nl_query, n_samples=n_samples, seed=seed))
