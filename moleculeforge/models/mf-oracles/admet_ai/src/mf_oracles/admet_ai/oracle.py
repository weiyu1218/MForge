"""ADMET-AI oracle: L1 ADMET property prediction."""
from __future__ import annotations

import inspect
import os
from collections.abc import Callable
from typing import Any

from mf_core.plugins.oracle import OraclePlugin


class ADMETAIOracle(OraclePlugin):
    def __init__(self, runner=None):
        self.runner = runner

    async def evaluate(
        self,
        molecules: list[str],
        properties: list[str],
    ) -> dict[str, dict[str, float]]:
        if self.runner is None:
            raise RuntimeError("ADMET_AI_RUNNER is required")
        result = self.runner.evaluate(_rdkit_descriptor_rows(molecules), properties)
        if inspect.isawaitable(result):
            return await result
        return result

    async def predict_with_uncertainty(self, molecules, properties):
        if self.runner is None or not hasattr(self.runner, "predict_with_uncertainty"):
            raise RuntimeError("ADMET uncertainty runner is required")
        result = self.runner.predict_with_uncertainty(
            _rdkit_descriptor_rows(molecules),
            properties,
        )
        if inspect.isawaitable(result):
            return await result
        return result

    def oracle_level(self) -> int:
        return 0


class ADMETHTTPRunner:
    """HTTP runner for the external Chemprop ADMET microservice."""

    def __init__(
        self,
        service_url: str,
        targets: list[str],
        batch_size: int = 64,
        timeout: float = 120.0,
        post_json: Callable[[str, dict, float], dict[str, Any]] | None = None,
    ) -> None:
        if not service_url:
            raise RuntimeError("ADMET_SERVICE_URL is required")
        if not targets:
            raise RuntimeError("ADMET_TARGETS is required")
        self.service_url = service_url.rstrip("/")
        self.targets = list(targets)
        self.batch_size = int(batch_size)
        self.timeout = float(timeout)
        self._post_json = post_json or _httpx_post_json

    @classmethod
    def from_env(cls) -> "ADMETHTTPRunner":
        targets = [
            item.strip()
            for item in os.environ.get("ADMET_TARGETS", "").split(",")
            if item.strip()
        ]
        return cls(
            service_url=os.environ.get("ADMET_SERVICE_URL", ""),
            targets=targets,
            batch_size=int(os.environ.get("ADMET_BATCH_SIZE", "64")),
        )

    def evaluate(
        self,
        descriptor_rows: list[dict[str, float | int | str]],
        properties: list[str],
    ) -> dict[str, dict[str, float]]:
        requested = list(properties or self.targets)
        if not requested:
            raise RuntimeError("ADMET prediction requires at least one target")
        smiles = [str(row["smiles"]) for row in descriptor_rows]
        payload = {
            "smiles": smiles,
            "endpoints": requested,
            "batch_size": self.batch_size,
        }
        data = self._post_json(f"{self.service_url}/predict", payload, self.timeout)
        return _predictions_by_smiles(data, requested)

    def predict_with_uncertainty(self, descriptor_rows, properties):
        raise RuntimeError("ADMET uncertainty runner is required")


def _httpx_post_json(url: str, payload: dict, timeout: float) -> dict[str, Any]:
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("httpx is required for ADMET HTTP runner") from exc
    response = httpx.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _predictions_by_smiles(data: dict[str, Any], requested: list[str]) -> dict[str, dict[str, float]]:
    results = data.get("results")
    if not isinstance(results, list):
        raise RuntimeError("ADMET service response must contain a results list")
    output: dict[str, dict[str, float]] = {}
    for item in results:
        if not isinstance(item, dict):
            raise RuntimeError("ADMET service result entries must be objects")
        smiles = item.get("smiles")
        predictions = item.get("predictions")
        if not isinstance(smiles, str) or not isinstance(predictions, dict):
            raise RuntimeError("ADMET service result requires smiles and predictions")
        missing = [name for name in requested if name not in predictions]
        if missing:
            raise RuntimeError(
                "ADMET service response missing targets: " + ", ".join(missing)
            )
        output[smiles] = {name: float(predictions[name]) for name in requested}
    return output


def _rdkit_descriptor_rows(molecules: list[str]) -> list[dict[str, float | int | str]]:
    try:
        from rdkit import Chem
        from rdkit.Chem import QED, Descriptors, Lipinski
    except ImportError as exc:
        raise ImportError("RDKit is required for ADMET descriptor generation") from exc

    rows: list[dict[str, float | int | str]] = []
    for smiles in molecules:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES for ADMET descriptor generation: {smiles}")
        row = {
            "smiles": Chem.MolToSmiles(mol),
            "mol_wt": float(Descriptors.MolWt(mol)),
            "logp": float(Descriptors.MolLogP(mol)),
            "tpsa": float(Descriptors.TPSA(mol)),
            "hbd": int(Lipinski.NumHDonors(mol)),
            "hba": int(Lipinski.NumHAcceptors(mol)),
            "rotatable_bonds": int(Lipinski.NumRotatableBonds(mol)),
            "ring_count": int(Descriptors.RingCount(mol)),
            "qed": float(QED.qed(mol)),
        }
        row["lipinski_violations"] = int(_lipinski_violations(row))
        rows.append(row)
    return rows


def _lipinski_violations(row: dict[str, float | int | str]) -> int:
    violations = 0
    if float(row["mol_wt"]) > 500.0:
        violations += 1
    if int(row["hbd"]) > 5:
        violations += 1
    if int(row["hba"]) > 10:
        violations += 1
    if float(row["logp"]) > 5.0:
        violations += 1
    return violations
