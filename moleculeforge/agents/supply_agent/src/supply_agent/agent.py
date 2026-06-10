"""Supply Agent - Building block accessibility scoring (Agent-6)."""
import inspect
import json
import os
from typing import Any

from mf_agents.base.agent import BaseAgent, ensure_default_event_loop
from mf_agents.crg.graph import ChemicalReasoningGraph
from mf_core.db.repositories import build_shared_crg_repository_from_env
from mf_core.proto_gen.moleculeforge.v1.oracle import supply_pb2, supply_pb2_grpc


class SupplyOracleGrpcClient:
    def __init__(self, target: str):
        import grpc

        ensure_default_event_loop()
        self.channel = grpc.aio.insecure_channel(target)
        self.stub = supply_pb2_grpc.SupplyOracleServiceStub(self.channel)

    async def check_availability(self, smiles: str) -> dict:
        response = await self.stub.CheckAvailability(supply_pb2.AvailabilityRequest(smiles=smiles))
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
        }


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
        self.crg_repository = (
            crg_repository
            if crg_repository is not None
            else build_shared_crg_repository_from_env()
        )

    async def handle_message(self, subject, payload, reply_to=""):
        data = json.loads(payload) if isinstance(payload, bytes) else {"raw": payload}
        result = await self.process(data)
        if reply_to:
            await self.publish(reply_to, json.dumps(result).encode())

    async def process(self, data):
        """Evaluate supply chain feasibility for building blocks.

        Checks building block availability, lead times, pricing, and
        supplier diversity across major chemical catalogs.
        """
        smiles = data.get("smiles", "")
        building_blocks = data.get("building_blocks")
        if building_blocks is None:
            raise ValueError("building_blocks is required for supply assessment")
        if not isinstance(building_blocks, list):
            raise TypeError("building_blocks must be a list")
        run_id = str(data.get("run_id") or data.get("request_id") or "")
        cached_feasibility = await self._existing_supply_feasibility(smiles, run_id)
        if cached_feasibility:
            block_assessments = _crg_supply_records(building_blocks, cached_feasibility)
            assessment = _supply_assessment(block_assessments)
            assessment["overall_feasibility"] = cached_feasibility
            return {
                "agent": self.name,
                "status": "assessed",
                "smiles": smiles,
                "cache_source": "shared_crg",
                "supply_assessment": assessment,
                "block_assessments": block_assessments,
            }
        if await self._has_zero_retrosyn_routes(smiles, run_id):
            block_assessments = [
                _crg_unavailable_record(_building_block_smiles(building_block))
                for building_block in building_blocks
            ]
            assessment = _supply_assessment(block_assessments)
            belief = self.crg.add_belief(
                subject=smiles or "route",
                predicate="supply_feasibility",
                obj=assessment["overall_feasibility"],
                confidence=0.0,
                source_agent=self.name,
                evidence_ids=["crg_retrosyn_routes"],
            )
            await self._persist_belief(
                belief,
                project_id=str(data.get("project_id") or ""),
                run_id=run_id,
            )
            return {
                "agent": self.name,
                "status": "assessed",
                "smiles": smiles,
                "supply_assessment": assessment,
                "block_assessments": block_assessments,
            }
        if building_blocks and self.supply_client is None:
            raise RuntimeError("SUPPLY_ORACLE_TARGET or supply_client is required")

        block_assessments = []
        for building_block in building_blocks:
            block_smiles = _building_block_smiles(building_block)
            record = await self.supply_client.check_availability(block_smiles)
            block_assessments.append(_availability_record(block_smiles, record))

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
                str(record["catalog_id"])
                for record in block_assessments
                if record.get("catalog_id")
            ],
        )
        await self._persist_belief(
            belief,
            project_id=str(data.get("project_id") or ""),
            run_id=str(data.get("run_id") or data.get("request_id") or ""),
        )
        return {
            "agent": self.name,
            "status": "assessed",
            "smiles": smiles,
            "supply_assessment": assessment,
            "block_assessments": block_assessments,
        }

    async def _existing_supply_feasibility(self, smiles: str, run_id: str) -> str:
        if not run_id or self.crg_repository is None:
            return ""
        if not callable(getattr(self.crg_repository, "get_run_crg", None)):
            return ""
        crg = await self.read_shared_crg(run_id)
        for belief in crg.get("beliefs", []):
            if not isinstance(belief, dict):
                continue
            if str(belief.get("subject") or "") != str(smiles or "route"):
                continue
            predicate = str(belief.get("predicate") or "")
            object_value = str(
                belief.get("object_value", belief.get("object", ""))
            ).lower()
            if predicate == "supply_feasibility" and object_value in {
                "available",
                "partial",
                "unavailable",
                "unknown",
            }:
                return object_value
        return ""

    async def _has_zero_retrosyn_routes(self, smiles: str, run_id: str) -> bool:
        if not run_id or self.crg_repository is None:
            return False
        if not callable(getattr(self.crg_repository, "get_run_crg", None)):
            return False
        crg = await self.read_shared_crg(run_id)
        for belief in crg.get("beliefs", []):
            if not isinstance(belief, dict):
                continue
            if str(belief.get("subject") or "") != str(smiles or "route"):
                continue
            predicate = str(belief.get("predicate") or "")
            object_value = str(
                belief.get("object_value", belief.get("object", ""))
            )
            if predicate == "retrosyn_routes" and object_value == "0":
                return True
        return False

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
            building_block.get("smiles")
            or building_block.get("building_block_smiles")
            or ""
        )
    else:
        raise TypeError("building block entries must be strings or dictionaries")
    if not smiles:
        raise ValueError("building block smiles is required")
    return smiles


def _availability_record(smiles: str, record: Any) -> dict:
    return {
        "smiles": str(_field(record, "smiles", smiles) or smiles),
        "available": bool(_field(record, "available", False)),
        "catalog_id": _field(record, "catalog_id", None),
        "catalog_source": _field(record, "catalog_source", _field(record, "source", None)),
        "source_timestamp": _field(record, "source_timestamp", None),
        "price": _field(record, "price", None),
        "currency": _field(record, "currency", None),
        "lead_time_days": _field(record, "lead_time_days", None),
    }


def _crg_unavailable_record(smiles: str) -> dict:
    return {
        "smiles": smiles,
        "available": False,
        "catalog_id": "crg_retrosyn_routes",
        "catalog_source": "shared_crg",
        "source_timestamp": None,
        "price": None,
        "currency": None,
        "lead_time_days": None,
    }


def _crg_supply_records(building_blocks: list, feasibility: str) -> list[dict]:
    total_blocks = len(building_blocks)
    if feasibility == "available":
        available_count = total_blocks
    elif feasibility == "partial":
        available_count = max(1, total_blocks - 1) if total_blocks else 0
    else:
        available_count = 0
    records = []
    for index, building_block in enumerate(building_blocks):
        records.append(
            _crg_supply_record(
                _building_block_smiles(building_block),
                available=index < available_count,
            )
        )
    return records


def _crg_supply_record(smiles: str, *, available: bool) -> dict:
    return {
        "smiles": smiles,
        "available": available,
        "catalog_id": "crg_supply_feasibility",
        "catalog_source": "shared_crg",
        "source_timestamp": None,
        "price": None,
        "currency": None,
        "lead_time_days": None,
    }


def _field(record: Any, name: str, fallback: Any) -> Any:
    if isinstance(record, dict):
        return record.get(name, fallback)
    return getattr(record, name, fallback)


def _supply_assessment(records: list[dict]) -> dict:
    total_blocks = len(records)
    commercially_available = sum(1 for record in records if record["available"])
    prices = [
        _numeric(record["price"])
        for record in records
        if _numeric(record["price"]) is not None
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
