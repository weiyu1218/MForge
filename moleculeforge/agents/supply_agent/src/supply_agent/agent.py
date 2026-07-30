"""Supply Agent - Building block accessibility scoring (Agent-6)."""

import asyncio
import inspect
import os
from typing import Any

from mf_agents.base.agent import (
    BaseAgent,
    agent_health_check_timeout_seconds,
    close_owned_channel,
    ensure_default_event_loop,
    run_health_probe_in_daemon,
)
from mf_agents.crg.graph import ChemicalReasoningGraph
from mf_core.db.repositories import build_shared_crg_repository_from_env

_SYNTHETIC_VALIDATION_MARKER = "synthetic_pipeline_validation_only"


class SupplyOracleGrpcClient:
    def __init__(self, target: str):
        import grpc

        self.target = target
        ensure_default_event_loop()
        self.channel = grpc.aio.insecure_channel(target)
        self.stub = None
        self._closed = False

    async def check_availability(
        self,
        smiles: str,
        *,
        request_id: str | None = None,
        project_id: str | None = None,
        candidate_id: str | None = None,
        candidate_index: int | None = None,
        canonical_smiles: str | None = None,
    ) -> dict:
        from mf_core.proto_gen.moleculeforge.v1.oracle import supply_pb2

        correlation = {
            key: value
            for key, value in {
                "request_id": request_id,
                "project_id": project_id,
                "candidate_id": candidate_id,
                "candidate_index": candidate_index,
                "canonical_smiles": canonical_smiles,
            }.items()
            if value is not None
        }
        response = await self._stub().CheckAvailability(
            supply_pb2.AvailabilityRequest(
                smiles=smiles,
                **correlation,
            )
        )
        return {
            "smiles": response.smiles,
            "available": response.available,
            "catalog_id": response.catalog_id,
            "source": response.catalog_source,
            "source_timestamp": response.source_timestamp,
            "price": response.price if response.HasField("price") else None,
            "currency": response.currency,
            "lead_time_days": (
                response.lead_time_days if response.HasField("lead_time_days") else None
            ),
            "evidence_id": response.evidence_id,
            "catalog_version": response.catalog_version,
            "catalog_checksum": response.catalog_checksum,
            "request_id": response.request_id,
            "project_id": response.project_id,
            "candidate_id": response.candidate_id,
            "candidate_index": (
                response.candidate_index if response.HasField("candidate_index") else None
            ),
            "canonical_smiles": response.canonical_smiles,
        }

    async def batch_check(
        self,
        smiles_list: list[str],
        *,
        request_id: str,
        project_id: str,
        candidate_id: str,
        candidate_index: int,
        canonical_smiles: str,
    ) -> dict:
        from mf_core.proto_gen.moleculeforge.v1.oracle import supply_pb2

        identity = {
            "request_id": request_id,
            "project_id": project_id,
            "candidate_id": candidate_id,
            "candidate_index": candidate_index,
            "canonical_smiles": canonical_smiles,
        }
        request = supply_pb2.BatchAvailabilityRequest(
            **identity,
            requests=[
                supply_pb2.AvailabilityRequest(
                    smiles=smiles,
                    request_id=f"{request_id}:supply:{index}",
                    project_id=project_id,
                    candidate_id=candidate_id,
                    candidate_index=candidate_index,
                    canonical_smiles=canonical_smiles,
                )
                for index, smiles in enumerate(smiles_list)
            ],
        )
        response = await self._stub().BatchCheck(request)
        for field, expected in identity.items():
            actual = getattr(response, field)
            if actual != expected:
                raise RuntimeError(f"supply batch response {field} does not match request")
        if len(response.results) != len(smiles_list):
            raise RuntimeError("supply batch response result count does not match request")
        return {
            **identity,
            "results": [_availability_response_dict(result) for result in response.results],
        }

    async def health_check(self) -> dict[str, bool]:
        from mf_core.proto_gen.moleculeforge.v1.oracle import supply_pb2

        response = await self._stub().CheckAvailability(
            supply_pb2.AvailabilityRequest(smiles="C", request_id="supply-health"),
            timeout=agent_health_check_timeout_seconds(),
        )
        return {"healthy": bool(response.smiles)}

    def _stub(self):
        if self.stub is None:
            from mf_core.proto_gen.moleculeforge.v1.oracle import supply_pb2_grpc

            self.stub = supply_pb2_grpc.SupplyOracleServiceStub(self.channel)
        return self.stub

    async def close(self) -> None:
        await close_owned_channel(self, self.channel)


class _SupplyClientTarget:
    def __init__(self, client: object) -> None:
        self.client = client

    @property
    def _close_target(self) -> object:
        return self.client

    async def health_check(self) -> dict[str, bool]:
        result = await run_health_probe_in_daemon(lambda: _run_supply_health_probe(self.client))
        return {"healthy": isinstance(result, dict) and str(result.get("smiles") or "") == "C"}

    async def close(self) -> None:
        close = getattr(self.client, "close", None)
        if not callable(close):
            return
        result = close()
        if inspect.isawaitable(result):
            await result


def _run_supply_health_probe(client: object) -> object:
    result = client.check_availability("C")
    if inspect.isawaitable(result):
        return asyncio.run(result)
    return result


class SupplyAgent(BaseAgent):
    def __init__(
        self,
        message_bus=None,
        supply_client=None,
        supply_target: str | None = None,
        crg_repository: Any = None,
    ):
        super().__init__("supply_agent", message_bus)
        self._subscription_subjects = ["agent.supply.request", "orchestrator.supply.check"]
        self.crg = ChemicalReasoningGraph()
        self.supply_client = supply_client or _build_supply_client(supply_target)
        if crg_repository is None:
            self.crg_repository = build_shared_crg_repository_from_env()
            self._owns_crg_repository = self.crg_repository is not None
        else:
            self.crg_repository = crg_repository
            self._owns_crg_repository = False

    def runtime_targets(self) -> dict[str, object | None]:
        target = self.supply_client
        if (
            target is not None
            and not callable(getattr(target, "health_check", None))
            and callable(getattr(target, "check_availability", None))
        ):
            target = _SupplyClientTarget(target)
        targets: dict[str, object | None] = {"supply_oracle": target}
        if self._owns_crg_repository:
            targets["crg_repository"] = self.crg_repository
        return targets

    async def process(self, data):
        """Evaluate supply chain feasibility for building blocks.

        Checks building block availability, lead times, pricing, and
        supplier diversity across major chemical catalogs.
        """
        if not isinstance(data, dict):
            raise TypeError("supply request must be a dictionary")
        strict_identity = data.get("workflow_scope") == "full" or any(
            field in data
            for field in (
                "request_id",
                "project_id",
                "candidate_id",
                "candidate_index",
                "canonical_smiles",
            )
        )
        identity = _supply_identity(data) if strict_identity else None
        smiles = (
            identity["canonical_smiles"]
            if identity is not None
            else _required_string(data.get("smiles"), "smiles")
        )
        building_blocks = data.get("building_blocks")
        if building_blocks is None:
            raise ValueError("building_blocks is required for supply assessment")
        if not isinstance(building_blocks, list):
            raise TypeError("building_blocks must be a list")
        if building_blocks and self.supply_client is None:
            raise RuntimeError("SUPPLY_ORACLE_TARGET or supply_client is required")

        block_smiles_list = [
            _building_block_smiles(building_block) for building_block in building_blocks
        ]
        batch_check = getattr(self.supply_client, "batch_check", None)
        if block_smiles_list and identity is not None and callable(batch_check):
            batch_response = await batch_check(block_smiles_list, **identity)
            block_assessments = _batch_availability_records(
                block_smiles_list,
                batch_response,
                identity=identity,
            )
        else:
            block_assessments = []
            for index, block_smiles in enumerate(block_smiles_list):
                if identity is None:
                    record = await self.supply_client.check_availability(block_smiles)
                    block_assessments.append(
                        _legacy_availability_record(block_smiles, record)
                    )
                    continue
                query_identity = {
                    **identity,
                    "request_id": f"{identity['request_id']}:supply:{index}",
                }
                record = await self.supply_client.check_availability(
                    block_smiles,
                    **query_identity,
                )
                block_assessments.append(
                    _availability_record(
                        block_smiles,
                        record,
                        expected_identity=query_identity,
                    )
                )

        assessment = _supply_assessment(block_assessments)
        belief = self.crg.add_belief(
            subject=smiles or "route",
            predicate="supply_feasibility",
            obj=assessment["overall_feasibility"],
            confidence=(
                assessment["commercially_available"] / assessment["total_blocks"]
                if assessment["total_blocks"]
                else 0.0
            ),
            source_agent=self.name,
            evidence_ids=[
                str(record.get("evidence_id") or record.get("catalog_id"))
                for record in block_assessments
                if record.get("evidence_id") or record.get("catalog_id")
            ],
        )
        await self._persist_belief(
            belief,
            project_id=str(data.get("project_id") or ""),
            run_id=str(data.get("run_id") or data.get("request_id") or ""),
        )
        result = {
            "agent": self.name,
            "status": "assessed",
            "smiles": smiles,
            "supply_assessment": assessment,
            "block_assessments": block_assessments,
        }
        if identity is not None:
            result.update(
                {
                    "project_id": identity["project_id"],
                    "candidate_id": identity["candidate_id"],
                    "candidate_index": identity["candidate_index"],
                    "canonical_smiles": identity["canonical_smiles"],
                }
            )
        if data.get("workflow_scope") == "full":
            result["route_id"] = _required_string(data.get("route_id"), "route_id")
        return result

    async def _persist_belief(self, belief, project_id: str, run_id: str) -> None:
        if self.crg_repository is None:
            return
        write_belief = getattr(self.crg_repository, "write_workflow_belief", None)
        if not callable(write_belief):
            raise TypeError("crg_repository must expose write_workflow_belief(**kwargs)")
        result = write_belief(
            project_id=project_id,
            run_id=run_id or belief.subject,
            belief_id=belief.id,
            subject=belief.subject,
            predicate=belief.predicate,
            object_value=belief.object,
            confidence=belief.confidence,
            source_agent=belief.source_agent,
            timestamp_ns=belief.timestamp_ns,
            evidence_ids=list(belief.evidence_ids),
        )
        if inspect.isawaitable(result):
            await result


def _build_supply_client(supply_target: str | None):
    target = supply_target or os.environ.get("SUPPLY_ORACLE_TARGET", "")
    if target:
        return SupplyOracleGrpcClient(target)
    if os.environ.get("SUPPLY_CATALOG_URI"):
        try:
            from supply_oracle_svc.main import _build_catalog_client
        except ImportError as exc:
            raise RuntimeError(
                "SUPPLY_CATALOG_URI is configured, but supply_oracle_svc is not importable"
            ) from exc
        return _build_catalog_client()
    return None


def _building_block_smiles(building_block: object) -> str:
    if isinstance(building_block, str):
        smiles = building_block
    elif isinstance(building_block, dict):
        smiles = str(
            building_block.get("smiles") or building_block.get("building_block_smiles") or ""
        )
    else:
        raise TypeError("building block entries must be strings or dictionaries")
    if not smiles:
        raise ValueError("building block smiles is required")
    return smiles


def _supply_identity(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise TypeError("supply request must be a dictionary")
    identity = {
        field: _required_string(data.get(field), field)
        for field in ("request_id", "project_id", "candidate_id", "canonical_smiles")
    }
    smiles = _required_string(data.get("smiles"), "smiles")
    if smiles != identity["canonical_smiles"]:
        raise ValueError("smiles must match canonical_smiles")
    candidate_index = data.get("candidate_index")
    if (
        isinstance(candidate_index, bool)
        or not isinstance(candidate_index, int)
        or candidate_index < 0
    ):
        raise ValueError("candidate_index must be a non-negative integer")
    identity["candidate_index"] = candidate_index
    return identity


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} is required")
    if value != value.strip():
        raise ValueError(f"{field} must not contain surrounding whitespace")
    return value


def _availability_record(
    smiles: str,
    record: Any,
    *,
    expected_identity: dict[str, Any],
) -> dict:
    _reject_synthetic_validation_catalog(record)
    response_smiles = _field(record, "smiles", None)
    if response_smiles != smiles:
        raise RuntimeError("supply response smiles does not match request")
    available = _field(record, "available", None)
    if not isinstance(available, bool):
        raise RuntimeError("supply response available must be a boolean")
    for field, expected in expected_identity.items():
        if _field(record, field, None) != expected:
            raise RuntimeError(f"supply response {field} does not match request")
    evidence_id = _required_response_string(record, "evidence_id")
    catalog_version = _required_response_string(record, "catalog_version")
    catalog_checksum = _required_response_string(record, "catalog_checksum")
    if (
        not catalog_checksum.startswith("sha256:")
        or len(catalog_checksum) != len("sha256:") + 64
        or any(character not in "0123456789abcdef" for character in catalog_checksum[7:])
    ):
        raise RuntimeError("supply response catalog_checksum is invalid")
    if available:
        for field in ("catalog_id", "source", "source_timestamp"):
            _required_response_string(record, field)
    return {
        "smiles": smiles,
        "available": available,
        "catalog_id": _field(record, "catalog_id", None),
        "catalog_source": _field(record, "catalog_source", _field(record, "source", None)),
        "source_timestamp": _field(record, "source_timestamp", None),
        "price": _field(record, "price", None),
        "currency": _field(record, "currency", None),
        "lead_time_days": _field(record, "lead_time_days", None),
        "evidence_id": evidence_id,
        "catalog_version": catalog_version,
        "catalog_checksum": catalog_checksum,
        **expected_identity,
    }


def _legacy_availability_record(smiles: str, record: Any) -> dict:
    _reject_synthetic_validation_catalog(record)
    return {
        "smiles": str(_field(record, "smiles", smiles) or smiles),
        "available": bool(_field(record, "available", False)),
        "catalog_id": _field(record, "catalog_id", None),
        "catalog_source": _field(record, "catalog_source", _field(record, "source", None)),
        "source_timestamp": _field(record, "source_timestamp", None),
        "price": _field(record, "price", None),
        "currency": _field(record, "currency", None),
        "lead_time_days": _field(record, "lead_time_days", None),
        "evidence_id": _field(record, "evidence_id", None),
        "catalog_version": _field(record, "catalog_version", None),
        "catalog_checksum": _field(record, "catalog_checksum", None),
    }


def _reject_synthetic_validation_catalog(record: Any) -> None:
    catalog_source = _field(record, "catalog_source", _field(record, "source", None))
    if _SYNTHETIC_VALIDATION_MARKER in (
        _field(record, "catalog_version", None),
        catalog_source,
        _field(record, "validation_marker", None),
    ):
        raise RuntimeError(
            "synthetic validation catalog result cannot satisfy a business request"
        )


def _availability_response_dict(response: Any) -> dict[str, Any]:
    return {
        "smiles": response.smiles,
        "available": response.available,
        "catalog_id": response.catalog_id,
        "source": response.catalog_source,
        "source_timestamp": response.source_timestamp,
        "price": response.price if response.HasField("price") else None,
        "currency": response.currency,
        "lead_time_days": (
            response.lead_time_days if response.HasField("lead_time_days") else None
        ),
        "evidence_id": response.evidence_id,
        "catalog_version": response.catalog_version,
        "catalog_checksum": response.catalog_checksum,
        "request_id": response.request_id,
        "project_id": response.project_id,
        "candidate_id": response.candidate_id,
        "candidate_index": (
            response.candidate_index if response.HasField("candidate_index") else None
        ),
        "canonical_smiles": response.canonical_smiles,
    }


def _batch_availability_records(
    smiles_list: list[str],
    response: Any,
    *,
    identity: dict[str, Any],
) -> list[dict]:
    for field, expected in identity.items():
        if _field(response, field, None) != expected:
            raise RuntimeError(f"supply batch response {field} does not match request")
    results = _field(response, "results", None)
    if not isinstance(results, list) or len(results) != len(smiles_list):
        raise RuntimeError("supply batch response result count does not match request")
    return [
        _availability_record(
            smiles,
            record,
            expected_identity={
                **identity,
                "request_id": f"{identity['request_id']}:supply:{index}",
            },
        )
        for index, (smiles, record) in enumerate(zip(smiles_list, results, strict=True))
    ]


def _required_response_string(record: Any, field: str) -> str:
    value = _field(record, field, None)
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise RuntimeError(f"supply response {field} must be a non-empty string")
    return value


def _field(record: Any, name: str, fallback: Any) -> Any:
    if isinstance(record, dict):
        return record.get(name, fallback)
    return getattr(record, name, fallback)


def _supply_assessment(records: list[dict]) -> dict:
    total_blocks = len(records)
    commercially_available = sum(1 for record in records if record["available"])
    prices = [
        _numeric(record["price"]) for record in records if _numeric(record["price"]) is not None
    ]
    lead_times = [
        _numeric(record["lead_time_days"])
        for record in records
        if _numeric(record["lead_time_days"]) is not None
    ]
    suppliers = {
        str(record["catalog_source"])
        for record in records
        if record["available"] and record.get("catalog_source")
    }
    return {
        "total_blocks": total_blocks,
        "commercially_available": commercially_available,
        "avg_price_per_gram": _mean(prices),
        "avg_lead_time_days": _mean(lead_times),
        "supplier_diversity": len(suppliers),
        "overall_feasibility": _overall_feasibility(total_blocks, commercially_available),
    }


def _numeric(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _overall_feasibility(total_blocks: int, commercially_available: int) -> str:
    if total_blocks == 0:
        return "unknown"
    if commercially_available == total_blocks:
        return "available"
    if commercially_available > 0:
        return "partial"
    return "unavailable"
