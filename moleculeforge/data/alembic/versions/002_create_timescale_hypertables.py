"""Create TimescaleDB hypertables: oracle_calls, pareto_fronts.

Revision ID: 002
Revises: 001
Create Date: 2026-05-04
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── oracle_calls ─────────────────────────────────────────────────────
    op.create_table(
        "oracle_calls",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "mol_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("molecules.id"),
            nullable=False,
        ),
        sa.Column("oracle", sa.String(128), nullable=False),
        sa.Column("value", sa.Float, nullable=True),
        sa.Column("uncertainty", sa.Float, nullable=True),
        sa.Column("cost_s", sa.Float, nullable=True),
        sa.Column("called_at", sa.TIMESTAMP(timezone=True), nullable=False),
    )
    op.create_index("ix_oracle_calls_mol_id", "oracle_calls", ["mol_id"])
    op.create_index("ix_oracle_calls_oracle", "oracle_calls", ["oracle"])

    # ── pareto_fronts ───────────────────────────────────────────────────
    op.create_table(
        "pareto_fronts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("runs.id"),
            nullable=False,
        ),
        sa.Column("timestamp", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("front_json", postgresql.JSONB, nullable=True),
        sa.Column("hv_value", sa.Float, nullable=True),
    )
    op.create_index("ix_pareto_fronts_run_id", "pareto_fronts", ["run_id"])

    # Convert to TimescaleDB hypertables when extension is available
    conn = op.get_bind()
    result = conn.execute(
        sa.text("SELECT EXISTS(SELECT 1 FROM pg_proc WHERE proname = 'create_hypertable')")
    )
    has_timescale = result.scalar()
    if has_timescale:
        op.execute(
            "SELECT create_hypertable('oracle_calls', 'called_at', if_not_exists => TRUE)"
        )
        op.execute(
            "SELECT create_hypertable('pareto_fronts', 'timestamp', if_not_exists => TRUE)"
        )


def downgrade() -> None:
    op.drop_table("pareto_fronts")
    op.drop_table("oracle_calls")
