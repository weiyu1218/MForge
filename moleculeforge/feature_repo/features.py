from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, String
from feast.value_type import ValueType

FEATURE_REPO_DIR = Path(__file__).resolve().parent

molecule = Entity(
    name="entity_id",
    join_keys=["entity_id"],
    value_type=ValueType.STRING,
    description="Molecule entity keyed by MoleculeForge molecule id.",
)

molecule_source = FileSource(
    name="molecule_feature_source",
    path=str(FEATURE_REPO_DIR / "molecule_features.parquet"),
    timestamp_field="event_timestamp",
)

molecule_features = FeatureView(
    name="molecule_features",
    entities=[molecule],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="smiles", dtype=String),
        Field(name="source", dtype=String),
        Field(name="mw", dtype=Float64),
        Field(name="logp", dtype=Float64),
        Field(name="qed", dtype=Float64),
        Field(name="tpsa", dtype=Float64),
    ],
    source=molecule_source,
    online=True,
)
