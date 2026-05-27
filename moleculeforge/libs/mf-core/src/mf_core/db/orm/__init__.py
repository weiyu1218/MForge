"""SQLAlchemy ORM models for MoleculeForge core tables."""
import uuid
from datetime import UTC, datetime

from sqlalchemy import Float, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class MoleculeORM(Base):
    __tablename__ = "molecules"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    smiles: Mapped[str] = mapped_column(Text, nullable=False)
    inchikey: Mapped[str | None] = mapped_column(String(27), unique=True)
    mw: Mapped[float | None] = mapped_column(Float, nullable=True)
    logp: Mapped[float | None] = mapped_column(Float, nullable=True)
    extra: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=_utcnow, onupdate=_utcnow)


class RunORM(Base):
    __tablename__ = "runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    cig_id: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), server_default="pending")
    hv_pareto: Mapped[float | None] = mapped_column(Float, nullable=True)
    n_oracle_l1: Mapped[int] = mapped_column(Integer, server_default="0")
    n_oracle_l2: Mapped[int] = mapped_column(Integer, server_default="0")
    n_oracle_l3: Mapped[int] = mapped_column(Integer, server_default="0")
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=_utcnow, onupdate=_utcnow)


class AgentLogORM(Base):
    __tablename__ = "agent_logs"

    msg_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    trace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    run_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    from_agent: Mapped[str] = mapped_column(String(128), nullable=False)
    to_agent: Mapped[str] = mapped_column(String(128), nullable=False)
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    logged_at: Mapped[datetime] = mapped_column(default=_utcnow)


class OracleCallORM(Base):
    __tablename__ = "oracle_calls"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    oracle_name: Mapped[str] = mapped_column(String(64), nullable=False)
    smiles: Mapped[str] = mapped_column(Text, nullable=False)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    extra: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)


class ParetoFrontORM(Base):
    __tablename__ = "pareto_fronts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    smiles: Mapped[str] = mapped_column(Text, nullable=False)
    objectives: Mapped[dict] = mapped_column(JSONB, nullable=False)
    hv_contribution: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)
