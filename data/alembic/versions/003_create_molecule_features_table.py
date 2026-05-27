"""Create molecule_computed_features table for Feast FeatureView.

Revision ID: 003
Revises: 002
Create Date: 2026-05-04
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "molecule_computed_features",
        sa.Column(
            "mol_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("molecules.id"),
            primary_key=True,
        ),
        sa.Column("ecfp4_1024", postgresql.ARRAY(sa.BigInteger), nullable=True),
        sa.Column("rdkit_desc_200", postgresql.ARRAY(sa.Float), nullable=True),
        sa.Column("humu_z_128", postgresql.ARRAY(sa.Float), nullable=True),
        sa.Column("admet_values", postgresql.ARRAY(sa.Float), nullable=True),
        sa.Column("admet_uncertainty", postgresql.ARRAY(sa.Float), nullable=True),
        sa.Column("computed_at", sa.TIMESTAMP(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_molecule_computed_features_mol_id",
        "molecule_computed_features",
        ["mol_id"],
    )


def downgrade() -> None:
    op.drop_table("molecule_computed_features")
