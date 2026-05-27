"""CIC end-to-end integration tests.

Tests the full pipeline: Stage1 -> Stage1b -> Stage2 -> Stage2b -> Stage3
with realistic inputs.  No external network calls — all API tools are
mocked so the suite runs offline and in CI.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
import torch
from mf_core.types.cig import ChemicalIntentGraph, ObjectiveNode, ObjectiveType

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


KRAS_INPUT = (
    "Design a selective KRAS G12C covalent inhibitor "
    "with IC50 below 100 nM, oral bioavailability, "
    "no CYP3A4 interaction, avoiding Mirati patent US11186593, "
    "synthesizable in 5 steps from Enamine REAL building blocks."
)

SIMPLE_INPUT = "Design a drug-like soluble molecule with high potency"


# ──────────────────────────────────────────────────────────────────────
# Stage 1: Semantic extraction
# ──────────────────────────────────────────────────────────────────────

class TestStage1SemanticExtraction:
    """Test heuristic entity extraction from NL input."""

    def test_kras_g12c_target_extraction(self):
        from cig_compiler_svc.domain.stages.stage1_semantic import _heuristic_extract

        result = _heuristic_extract(KRAS_INPUT)

        # Target detected
        assert len(result["targets"]) >= 1
        target_names = [t["name"] for t in result["targets"]]
        assert any("KRAS" in n for n in target_names)

        # Activity type detected
        assert result["activity"]["type"] == "IC50"
        assert result["activity"]["direction"] == "minimize"
        assert result["activity"]["target_value"] == 100.0

        # ADMET constraints
        assert result["admet_constraints"]["oral_bioavailability_min"] is not None
        assert result["admet_constraints"]["cyp3a4_ic50_min"] == 10.0

        # Patent IDs
        assert "US11186593" in result["ip_constraints"]["blocked_patent_ids"]
        assert result["ip_constraints"]["fto_required"] is True

        # Synthetic constraints
        assert result["synthetic_constraints"]["max_synthetic_steps"] == 5

    def test_simple_input_extraction(self):
        from cig_compiler_svc.domain.stages.stage1_semantic import _heuristic_extract

        result = _heuristic_extract(SIMPLE_INPUT)

        assert len(result["properties"]) >= 1
        assert result["constraints"] is not None

    def test_empty_input_fallback(self):
        from cig_compiler_svc.domain.stages.stage1_semantic import _heuristic_extract

        result = _heuristic_extract("")
        assert "properties" in result
        assert isinstance(result["properties"], list)  # fallback returns empty list

    def test_patent_regex(self):
        from cig_compiler_svc.domain.stages.stage1_semantic import _heuristic_extract

        result = _heuristic_extract(
            "Avoid patents US11291420 and EP1234567B1 in the design"
        )
        pids = result["ip_constraints"]["blocked_patent_ids"]
        assert "US11291420" in pids
        assert "EP1234567B1" in pids

    def test_binding_mode_detection(self):
        from cig_compiler_svc.domain.stages.stage1_semantic import _heuristic_extract

        result = _heuristic_extract(
            "Design a covalent irreversible inhibitor for EGFR"
        )
        assert result["targets"][0]["binding_mode"] == "covalent_irreversible"


# ──────────────────────────────────────────────────────────────────────
# Stage 2: CIG construction
# ──────────────────────────────────────────────────────────────────────

class TestStage2CIGConstruction:
    """Test CIG building from extracted intent."""

    def test_build_cig_from_kras(self):
        from cig_compiler_svc.domain.stages.stage1_semantic import _heuristic_extract
        from cig_compiler_svc.domain.stages.stage2_cig_build import build_cig

        extracted = _heuristic_extract(KRAS_INPUT)
        cig = build_cig(extracted, KRAS_INPUT)

        assert cig.intent_id.startswith("cig-")
        assert cig.source_user_input == KRAS_INPUT

        # Should have multiple objective types
        obj_ids = [o.id for o in cig.objective_nodes]
        assert any("affinity" in oid for oid in obj_ids), "Missing affinity objective"
        assert any("admet" in oid for oid in obj_ids), "Missing ADMET objective"
        assert any("fto" in oid for oid in obj_ids), "Missing FTO objective"

        # Target context should have the target
        assert len(cig.target_context.get("uniprot_ids", [])) >= 1

    def test_build_cig_weights_normalized(self):
        from cig_compiler_svc.domain.stages.stage1_semantic import _heuristic_extract
        from cig_compiler_svc.domain.stages.stage2_cig_build import build_cig

        extracted = _heuristic_extract(KRAS_INPUT)
        cig = build_cig(extracted, KRAS_INPUT)

        total_weight = sum(o.weight for o in cig.objective_nodes)
        assert 0.95 <= total_weight <= 1.05, f"Weights sum to {total_weight}"

    def test_build_cig_admet_bundle_constraints(self):
        from cig_compiler_svc.domain.stages.stage1_semantic import _heuristic_extract
        from cig_compiler_svc.domain.stages.stage2_cig_build import build_cig

        extracted = _heuristic_extract(KRAS_INPUT)
        cig = build_cig(extracted, KRAS_INPUT)

        admet_nodes = [o for o in cig.objective_nodes if "admet" in o.id]
        assert len(admet_nodes) == 1
        admet = admet_nodes[0]
        assert admet.type == ObjectiveType.MULTI_CONSTRAINT_SATISFY
        assert admet.constraints is not None
        assert "CYP3A4_IC50" in admet.constraints

    def test_build_cig_fto_objective(self):
        from cig_compiler_svc.domain.stages.stage1_semantic import _heuristic_extract
        from cig_compiler_svc.domain.stages.stage2_cig_build import build_cig

        extracted = _heuristic_extract(KRAS_INPUT)
        cig = build_cig(extracted, KRAS_INPUT)

        fto_nodes = [o for o in cig.objective_nodes if "fto" in o.id]
        assert len(fto_nodes) == 1
        assert fto_nodes[0].pareto_tier == 1  # FTO is a hard constraint

    def test_build_cig_legacy_input(self):
        """Ensure backward compatibility with legacy input format."""
        from cig_compiler_svc.domain.stages.stage2_cig_build import build_cig

        extracted = {
            "properties": [
                {"name": "qed", "direction": "maximize", "priority": 1},
            ],
            "constraints": {"max_mw": 400},
        }
        cig = build_cig(extracted, "legacy test")
        assert cig.intent_id.startswith("cig-")
        assert len(cig.objective_nodes) >= 1
        assert isinstance(cig.generative_priors, dict)


# ──────────────────────────────────────────────────────────────────────
# Stage 2b: CIG validation
# ──────────────────────────────────────────────────────────────────────

class TestStage2bValidation:
    """Test CIG consistency validation."""

    def test_valid_cig_passes(self):
        from cig_compiler_svc.domain.stages.stage1_semantic import _heuristic_extract
        from cig_compiler_svc.domain.stages.stage2_cig_build import build_cig
        from cig_compiler_svc.domain.stages.validation import validate_cig

        extracted = _heuristic_extract(KRAS_INPUT)
        cig = build_cig(extracted, KRAS_INPUT)
        report = validate_cig(cig)

        assert report.is_valid, f"Validation failed: {report}"

    def test_empty_cig_fails(self):
        from cig_compiler_svc.domain.stages.validation import validate_cig
        from mf_core.types.cig import TargetContext

        empty_cig = ChemicalIntentGraph(
            intent_id="test-empty",
            target_context=TargetContext(),
            objective_nodes=[],
            source_user_input="test",
        )
        report = validate_cig(empty_cig)
        assert not report.is_valid
        assert any("No objective" in e for e in report.errors)

    def test_weight_warning(self):
        from cig_compiler_svc.domain.stages.validation import validate_cig
        from mf_core.types.cig import TargetContext

        cig = ChemicalIntentGraph(
            intent_id="test-weights",
            target_context=TargetContext(),
            objective_nodes=[
                ObjectiveNode(
                    id="obj1",
                    type=ObjectiveType.CONTINUOUS_MAXIMIZE,
                    oracle="test",
                    weight=0.9,
                ),
                ObjectiveNode(
                    id="obj2",
                    type=ObjectiveType.CONTINUOUS_MAXIMIZE,
                    oracle="test",
                    weight=0.9,
                ),
            ],
            source_user_input="test",
        )
        report = validate_cig(cig)
        assert any("sum to" in w for w in report.warnings)


# ──────────────────────────────────────────────────────────────────────
# Stage 3: HCIV encoding
# ──────────────────────────────────────────────────────────────────────

class TestStage3HCIVEncoding:
    """Test HCIV encoding (hash + learned)."""

    def _make_cig(self) -> ChemicalIntentGraph:
        from cig_compiler_svc.domain.stages.stage1_semantic import _heuristic_extract
        from cig_compiler_svc.domain.stages.stage2_cig_build import build_cig

        extracted = _heuristic_extract(KRAS_INPUT)
        return build_cig(extracted, KRAS_INPUT)

    def test_hash_encoding_reproducible(self):
        from cig_compiler_svc.domain.hciv_encoder import hash_encode_hciv

        cig = self._make_cig()
        hciv1 = hash_encode_hciv(cig, dim=16, seed=42)
        hciv2 = hash_encode_hciv(cig, dim=16, seed=42)

        assert hciv1.coordinates == hciv2.coordinates

    def test_hash_encoding_different_seeds(self):
        from cig_compiler_svc.domain.hciv_encoder import hash_encode_hciv

        cig = self._make_cig()
        hciv1 = hash_encode_hciv(cig, dim=16, seed=1)
        hciv2 = hash_encode_hciv(cig, dim=16, seed=2)

        assert hciv1.coordinates != hciv2.coordinates

    def test_hash_encoding_lorentz_constraint(self):
        """HCIV must satisfy <x,x>_L = -1 for Lorentz model."""
        from cig_compiler_svc.domain.hciv_encoder import hash_encode_hciv

        cig = self._make_cig()
        hciv = hash_encode_hciv(cig, dim=16, seed=42)

        coords = torch.tensor(hciv.coordinates)
        # Lorentz inner product: -x0^2 + sum(xi^2)
        lorentz_inner = -coords[0] ** 2 + (coords[1:] ** 2).sum()
        assert abs(lorentz_inner.item() - (-1.0)) < 0.01, (
            f"Lorentz constraint violated: <x,x>_L = {lorentz_inner.item()}"
        )

    def test_hash_encoding_time_positive(self):
        """Time component x_0 must be > 0 on Lorentz manifold."""
        from cig_compiler_svc.domain.hciv_encoder import hash_encode_hciv

        cig = self._make_cig()
        hciv = hash_encode_hciv(cig, dim=16, seed=42)
        assert hciv.coordinates[0] > 0, f"x_0 = {hciv.coordinates[0]} must be > 0"

    def test_learned_encoder_output_shape(self):
        from cig_compiler_svc.domain.hciv_encoder import HCIVEncoder

        encoder = HCIVEncoder(dim=32, curvature=1.0)
        cig = self._make_cig()
        hciv, cone = encoder.encode(cig)

        assert len(hciv.coordinates) == 33  # dim + 1 for Lorentz
        assert hciv.coordinates[0] > 0  # time component positive
        assert cone.angle_radians > 0
        assert cone.length > 0

    def test_learned_encoder_lorentz_constraint(self):
        from cig_compiler_svc.domain.hciv_encoder import HCIVEncoder

        encoder = HCIVEncoder(dim=32, curvature=1.0)
        cig = self._make_cig()
        hciv, cone = encoder.encode(cig)

        coords = torch.tensor(hciv.coordinates)
        lorentz_inner = -coords[0] ** 2 + (coords[1:] ** 2).sum()
        assert abs(lorentz_inner.item() - (-1.0)) < 0.05, (
            f"Lorentz constraint violated: <x,x>_L = {lorentz_inner.item()}"
        )

    def test_cig_feature_extraction(self):
        from cig_compiler_svc.domain.hciv_encoder import cig_to_features

        cig = self._make_cig()
        features = cig_to_features(cig)

        assert features.shape == (64,)
        # Should have non-zero entries for the constraint flags
        assert features[28] == 1.0  # has_fto
        assert features[29] == 1.0  # has_admet


# ──────────────────────────────────────────────────────────────────────
# Full pipeline: Stage1 -> Stage2 -> Stage2b -> Stage3
# ──────────────────────────────────────────────────────────────────────

class TestFullCICPipeline:
    """End-to-end test through all CIC stages."""

    def test_hash_mode_full_pipeline(self):
        from cig_compiler_svc.domain.compiler import CIGCompiler

        compiler = CIGCompiler(
            mode="local_demo",
            encoding_mode="hash",
            hciv_dim=16,
        )
        cig, hciv, cone = _run(compiler.compile(KRAS_INPUT, seed=42))

        # CIG checks
        assert cig.intent_id.startswith("cig-")
        assert len(cig.objective_nodes) >= 2
        assert sum(o.weight for o in cig.objective_nodes) > 0

        # HCIV checks
        assert len(hciv.coordinates) == 17  # dim + 1
        assert hciv.manifold_type == "lorentz"

        # Cone checks
        assert cone.apex is not None
        assert cone.angle_radians > 0

    def test_learned_mode_full_pipeline(self):
        from cig_compiler_svc.domain.compiler import CIGCompiler
        from cig_compiler_svc.domain.hciv_encoder import HCIVEncoder

        encoder = HCIVEncoder(dim=16, curvature=1.0)
        compiler = CIGCompiler(
            mode="local_demo",
            encoding_mode="learned",
            hciv_dim=16,
            learned_encoder=encoder,
        )
        cig, hciv, cone = _run(compiler.compile(KRAS_INPUT, seed=42))

        assert cig.intent_id.startswith("cig-")
        assert len(hciv.coordinates) == 17
        assert cone.apex is not None

    def test_random_mode_full_pipeline(self):
        from cig_compiler_svc.domain.compiler import CIGCompiler

        compiler = CIGCompiler(
            mode="local_demo",
            encoding_mode="random",
            hciv_dim=8,
        )
        cig, hciv, cone = _run(compiler.compile(SIMPLE_INPUT, seed=42))

        assert cig.intent_id.startswith("cig-")
        assert len(hciv.coordinates) == 9

    @pytest.mark.asyncio
    async def test_grounding_mode_with_mocks(self):
        """Test knowledge grounding with mocked external API calls."""
        mock_uniprot_results = [
            {
                "accession": "P01116",
                "protein_name": "GTPase KRas",
                "gene_name": "KRAS",
                "organism": "Homo sapiens",
                "pdb_ids": ["4DSO", "5VBA"],
                "source_timestamp": "2026-05-15T00:00:00Z",
            }
        ]

        with patch(
            "cig_compiler_svc.domain.tools.uniprot_tool.query_uniprot_entry",
            new_callable=AsyncMock,
            return_value=mock_uniprot_results,
        ):
            from cig_compiler_svc.domain.stages.stage1_semantic import _heuristic_extract
            from cig_compiler_svc.domain.stages.stage1b_grounding import ground_knowledge

            extracted = _heuristic_extract("Design a KRAS G12C inhibitor")
            enriched = await ground_knowledge(extracted, enable_pdb=False)

            assert len(enriched.get("_grounded_pdb_ids", [])) >= 1
            assert "P01116" in enriched.get("_grounded_uniprot_ids", [])
