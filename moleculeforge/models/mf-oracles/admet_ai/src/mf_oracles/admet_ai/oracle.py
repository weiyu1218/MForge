"""ADMET-AI oracle: L1 ADMET property prediction."""

from __future__ import annotations

import asyncio
import inspect
import math
import os
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

from mf_core.plugins.oracle import (
    OracleDataError,
    OraclePlugin,
    OracleUnavailableError,
)


class ADMETPredictionResult(dict):
    def __init__(
        self,
        values: dict,
        *,
        model_version: str,
        artifact_name: str,
        artifact_checksum: str,
    ) -> None:
        super().__init__(values)
        self.model_version = model_version
        self.artifact_name = artifact_name
        self.artifact_checksum = artifact_checksum


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
        descriptor_rows = await asyncio.to_thread(_rdkit_descriptor_rows, molecules)
        evaluate = self.runner.evaluate
        if inspect.iscoroutinefunction(evaluate):
            result = await evaluate(descriptor_rows, properties)
        else:
            result = await asyncio.to_thread(evaluate, descriptor_rows, properties)
        if inspect.isawaitable(result):
            result = await result
        return _restore_input_smiles(result, molecules, descriptor_rows)

    async def predict_with_uncertainty(self, molecules, properties):
        if self.runner is None or not hasattr(self.runner, "predict_with_uncertainty"):
            raise RuntimeError("ADMET uncertainty runner is required")
        descriptor_rows = await asyncio.to_thread(_rdkit_descriptor_rows, molecules)
        predict = self.runner.predict_with_uncertainty
        if inspect.iscoroutinefunction(predict):
            result = await predict(descriptor_rows, properties)
        else:
            result = await asyncio.to_thread(
                predict,
                descriptor_rows,
                properties,
            )
        if inspect.isawaitable(result):
            result = await result
        return _restore_input_smiles(result, molecules, descriptor_rows)

    def oracle_level(self) -> int:
        return 1


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
        parsed_url = urlparse(service_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise RuntimeError("ADMET_SERVICE_URL must be an absolute http(s) URL")
        normalized_targets = [str(target).strip() for target in targets]
        if any(not target for target in normalized_targets):
            raise RuntimeError("ADMET_TARGETS must contain non-empty names")
        if len(set(normalized_targets)) != len(normalized_targets):
            raise RuntimeError("ADMET_TARGETS must not contain duplicates")
        try:
            parsed_batch_size = int(batch_size)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("ADMET batch_size must be a positive integer") from exc
        if isinstance(batch_size, bool) or parsed_batch_size <= 0:
            raise RuntimeError("ADMET batch_size must be a positive integer")
        try:
            parsed_timeout = float(timeout)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("ADMET timeout must be a finite positive number") from exc
        if isinstance(timeout, bool) or not math.isfinite(parsed_timeout) or parsed_timeout <= 0:
            raise RuntimeError("ADMET timeout must be a finite positive number")
        self.service_url = service_url.rstrip("/")
        self.targets = normalized_targets
        self.batch_size = parsed_batch_size
        self.timeout = parsed_timeout
        self._post_json = post_json or _httpx_post_json

    @classmethod
    def from_env(cls) -> ADMETHTTPRunner:
        targets = [
            item.strip() for item in os.environ.get("ADMET_TARGETS", "").split(",") if item.strip()
        ]
        try:
            batch_size = int(os.environ.get("ADMET_BATCH_SIZE", "64"))
        except ValueError as exc:
            raise RuntimeError("ADMET_BATCH_SIZE must be a positive integer") from exc
        return cls(
            service_url=os.environ.get("ADMET_SERVICE_URL", ""),
            targets=targets,
            batch_size=batch_size,
            timeout=os.environ.get("ADMET_ORACLE_TIMEOUT_SECONDS", "120"),
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
        return _predictions_by_smiles(data, requested, smiles)

    def predict_with_uncertainty(self, descriptor_rows, properties):
        requested = list(properties or self.targets)
        if not requested:
            raise RuntimeError("ADMET uncertainty prediction requires at least one target")
        smiles = [str(row["smiles"]) for row in descriptor_rows]
        payload = {
            "smiles": smiles,
            "endpoints": requested,
            "batch_size": self.batch_size,
            "return_uncertainty": True,
        }
        data = self._post_json(f"{self.service_url}/predict", payload, self.timeout)
        return _predictions_and_uncertainties_by_smiles(data, requested, smiles)


def _httpx_post_json(url: str, payload: dict, timeout: float) -> dict[str, Any]:
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("httpx is required for ADMET HTTP runner") from exc
    try:
        response = httpx.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise TimeoutError("ADMET service request timed out") from exc
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code
        if status_code in {408, 504}:
            raise TimeoutError(f"ADMET service returned HTTP {status_code}") from exc
        if status_code == 429 or status_code >= 500:
            raise OracleUnavailableError(f"ADMET service returned HTTP {status_code}") from exc
        raise OracleDataError(
            f"ADMET service rejected prediction request with HTTP {status_code}"
        ) from exc
    except httpx.RequestError as exc:
        raise OracleUnavailableError("ADMET service request failed") from exc
    try:
        data = response.json()
    except ValueError as exc:
        raise OracleDataError("ADMET service returned invalid JSON") from exc
    if not isinstance(data, dict):
        raise OracleDataError("ADMET service response must be an object")
    return data


def _predictions_by_smiles(
    data: dict[str, Any],
    requested: list[str],
    expected_smiles: list[str],
) -> ADMETPredictionResult:
    results, metadata = _validated_response(data, expected_smiles)
    output: dict[str, dict[str, float]] = {}
    for item in results:
        if not isinstance(item, dict):
            raise OracleDataError("ADMET service result entries must be objects")
        smiles = item.get("smiles")
        predictions = item.get("predictions")
        if not isinstance(smiles, str) or not isinstance(predictions, dict):
            raise OracleDataError("ADMET service result requires smiles and predictions")
        missing = [name for name in requested if name not in predictions]
        if missing:
            raise OracleDataError("ADMET service response missing targets: " + ", ".join(missing))
        output[smiles] = {
            name: _finite_metric(predictions[name], f"predictions[{name}]") for name in requested
        }
    return ADMETPredictionResult(output, **metadata)


def _predictions_and_uncertainties_by_smiles(
    data: dict[str, Any],
    requested: list[str],
    expected_smiles: list[str],
) -> ADMETPredictionResult:
    results, metadata = _validated_response(data, expected_smiles)
    output: dict[str, tuple[dict[str, float], dict[str, float]]] = {}
    for item in results:
        if not isinstance(item, dict):
            raise OracleDataError("ADMET service result entries must be objects")
        smiles = item.get("smiles")
        predictions = item.get("predictions")
        uncertainties = item.get("uncertainties")
        if (
            not isinstance(smiles, str)
            or not isinstance(predictions, dict)
            or not isinstance(uncertainties, dict)
        ):
            raise OracleDataError(
                "ADMET service uncertainty result requires smiles, predictions, and uncertainties"
            )
        missing_predictions = [name for name in requested if name not in predictions]
        missing_uncertainties = [name for name in requested if name not in uncertainties]
        if missing_predictions:
            raise OracleDataError(
                "ADMET service response missing targets: " + ", ".join(missing_predictions)
            )
        if missing_uncertainties:
            raise OracleDataError(
                "ADMET service response missing uncertainties: " + ", ".join(missing_uncertainties)
            )
        output[smiles] = (
            {name: _finite_metric(predictions[name], f"predictions[{name}]") for name in requested},
            {
                name: _non_negative_metric(uncertainties[name], f"uncertainties[{name}]")
                for name in requested
            },
        )
    return ADMETPredictionResult(output, **metadata)


def _validated_response(
    data: object,
    expected_smiles: list[str],
) -> tuple[list[dict], dict[str, str]]:
    if not isinstance(data, dict):
        raise OracleDataError("ADMET service response must be an object")
    results = data.get("results")
    if not isinstance(results, list):
        raise OracleDataError("ADMET service response must contain a results list")
    actual_smiles = [item.get("smiles") if isinstance(item, dict) else None for item in results]
    if actual_smiles != expected_smiles:
        raise OracleDataError("ADMET service result count or molecule order does not match request")
    model_version = data.get("model_version")
    if not isinstance(model_version, str) or not model_version.strip():
        raise OracleDataError("ADMET service response requires model_version")
    artifact = data.get("artifact")
    if not isinstance(artifact, dict):
        raise OracleDataError("ADMET service response requires artifact metadata")
    artifact_name = artifact.get("name")
    if not isinstance(artifact_name, str) or not artifact_name.strip():
        raise OracleDataError("ADMET service artifact requires name")
    checksum = artifact.get("sha256")
    if not isinstance(checksum, str):
        raise OracleDataError("ADMET service artifact requires sha256")
    digest = checksum.removeprefix("sha256:").lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise OracleDataError("ADMET service artifact sha256 must contain 64 hexadecimal digits")
    return results, {
        "model_version": model_version.strip(),
        "artifact_name": artifact_name.strip(),
        "artifact_checksum": f"sha256:{digest}",
    }


def _finite_metric(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise OracleDataError(f"ADMET service {field_name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise OracleDataError(f"ADMET service {field_name} must be finite")
    return number


def _non_negative_metric(value: object, field_name: str) -> float:
    number = _finite_metric(value, field_name)
    if number < 0:
        raise OracleDataError(f"ADMET service {field_name} must be non-negative")
    return number


def _restore_input_smiles(
    result: object,
    molecules: list[str],
    descriptor_rows: list[dict[str, float | int | str]],
):
    if not isinstance(result, dict):
        raise OracleDataError("ADMET runner result must be an object")
    canonical_smiles = [str(row["smiles"]) for row in descriptor_rows]
    if set(result) != set(canonical_smiles):
        raise OracleDataError(
            "ADMET runner result molecules do not match canonical request molecules"
        )
    remapped = {
        original: result[canonical]
        for original, canonical in zip(molecules, canonical_smiles, strict=True)
    }
    if isinstance(result, ADMETPredictionResult):
        return ADMETPredictionResult(
            remapped,
            model_version=result.model_version,
            artifact_name=result.artifact_name,
            artifact_checksum=result.artifact_checksum,
        )
    return remapped


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
