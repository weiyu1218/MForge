"""Create core tables: molecules, runs, agent_logs.

Revision ID: 001
Revises:
Create Date: 2026-05-04
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── molecules ────────────────────────────────────────────────────────
    op.create_table(
        "molecules",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("smiles", sa.Text, nullable=False),
        sa.Column("inchikey", sa.String(27), unique=True),
        sa.Column("mw", sa.Float, nullable=True),
        sa.Column("logp", sa.Float, nullable=True),
        sa.Column("extra", postgresql.JSONB, nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False),
    )
    op.create_index("ix_molecules_inchikey", "molecules", ["inchikey"])

    # ── runs ─────────────────────────────────────────────────────────────
    op.create_table(
        "runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("cig_id", sa.String(128), nullable=False),
        sa.Column("status", sa.String(32), server_default="pending"),
        sa.Column("hv_pareto", sa.Float, nullable=True),
        sa.Column("n_oracle_l1", sa.Integer, server_default="0"),
        sa.Column("n_oracle_l2", sa.Integer, server_default="0"),
        sa.Column("n_oracle_l3", sa.Integer, server_default="0"),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False),
    )
    op.create_index("ix_runs_cig_id", "runs", ["cig_id"])

    # ── agent_logs ───────────────────────────────────────────────────────
    op.create_table(
        "agent_logs",
        sa.Column("msg_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("trace_id", sa.String(128), nullable=False),
        sa.Column("run_id", sa.String(128), nullable=True),
        sa.Column("from_agent", sa.String(128), nullable=False),
        sa.Column("to_agent", sa.String(128), nullable=False),
        sa.Column("payload_hash", sa.String(64), nullable=False),
        sa.Column("logged_at", sa.TIMESTAMP(timezone=True), nullable=False),
    )
    op.create_index("ix_agent_logs_trace_id", "agent_logs", ["trace_id"])
    op.create_index("ix_agent_logs_run_id", "agent_logs", ["run_id"])


def downgrade() -> None:
    op.drop_table("agent_logs")
    op.drop_table("runs")
    op.drop_table("molecules")
