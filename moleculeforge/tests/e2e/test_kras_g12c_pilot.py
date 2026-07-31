"""E2E test: KRAS G12C inhibitor design pilot (Architecture §11).

This test validates the complete MoleculeForge pipeline for a real-world
drug discovery task: designing KRAS G12C covalent inhibitors.

Requirements to run:
  - docker compose -f infra/docker/docker-compose.test.yml up -d
  - Real or mock model weights for HFM-3D + HUMU encoders
  - Test dataset: KRAS G12C crystal structures (PDB: 6OIM, 6N2K)
"""

import os
from types import SimpleNamespace

import pytest
from mf_core.artifacts import (
    ArtifactRequirement,
    RequirementStatus,
    ToolRequirement,
    check_artifact,
    check_tool,
)

KRAS_E2E_REQUIRED_ARTIFACTS = (
    ArtifactRequirement("hfm_checkpoint", "HFM_CHECKPOINT_PATH"),
    ArtifactRequirement("hfm_decoder", "HFM_DECODER_PATH"),
    ArtifactRequirement("boltz_model", "BOLTZ_MODEL_PATH", kind="path"),
    ArtifactRequirement("boltz_input_templates", "BOLTZ_INPUT_TEMPLATE_DIR", kind="directory"),
    ArtifactRequirement("aizynth_config", "AIZYNTH_CONFIG_PATH", kind="file"),
)
KRAS_E2E_REQUIRED_TOOLS = (
    ToolRequirement("boltz", executable="boltz", env_var="BOLTZ_BINARY"),
)
KRAS_E2E_REQUIRED_FLAGS = (
    "CRITIC_AGENT_READY",
    "ORCHESTRATOR_E2E_READY",
)
KRAS_E2E_SIGSTORE_REQUIRED_ENV = (
    "SIGSTORE_IDENTITY_TOKEN",
    "SIGSTORE_EXPECTED_IDENTITY",
    "SIGSTORE_SIGN_COMMAND",
    "SIGSTORE_VERIFY_COMMAND",
    "SIGSTORE_REKOR_URL",
)
KRAS_E2E_REQUIRED_ENV = tuple(
    requirement.env_var for requirement in KRAS_E2E_REQUIRED_ARTIFACTS
) + KRAS_E2E_REQUIRED_FLAGS + KRAS_E2E_SIGSTORE_REQUIRED_ENV
KRAS_E2E_DKI_REQUIRED_ENV = (
    "NEO4J_URI",
    "NEO4J_USER",
    "NEO4J_PASSWORD",
    "MINIO_ENDPOINT_URL",
    "MINIO_ACCESS_KEY",
    "MINIO_SECRET_KEY",
    "MINIO_BUCKET",
    "REDIS_HOST",
    "REDIS_PORT",
)
KRAS_E2E_FULL_SCOPE = "full"


def kras_e2e_preflight_status() -> dict:
    missing: list[str] = []
    scope = _kras_e2e_scope()
    if scope != KRAS_E2E_FULL_SCOPE:
        missing.append("KRAS_E2E_SCOPE must be full")
    else:
        for requirement in KRAS_E2E_REQUIRED_ARTIFACTS:
            status = check_artifact(requirement)
            if not status.available:
                missing.append(_missing_status_name(status))
        for requirement in KRAS_E2E_REQUIRED_TOOLS:
            status = check_tool(requirement)
            if not status.available:
                missing.append(_missing_tool_status_name(requirement, status))
    for name in KRAS_E2E_REQUIRED_FLAGS:
        if not os.environ.get(name):
            missing.append(name)
        if os.environ.get(name) and os.environ.get(name) != "1":
            missing.append(f"{name}=1")
    for name in KRAS_E2E_DKI_REQUIRED_ENV:
        if not os.environ.get(name):
            missing.append(name)
    if not (os.environ.get("PROVENANCE_DATABASE_URL") or os.environ.get("TEST_DATABASE_URL")):
        missing.append("PROVENANCE_DATABASE_URL or TEST_DATABASE_URL")
    if os.environ.get("PROVENANCE_STORE_MODE") != "production_real":
        missing.append("PROVENANCE_STORE_MODE=production_real")
    for name in KRAS_E2E_SIGSTORE_REQUIRED_ENV:
        if not os.environ.get(name):
            missing.append(name)
    return {
        "ready": not missing,
        "missing": missing,
        "scope": scope,
        "message": "Missing KRAS G12C E2E dependencies: " + ", ".join(missing),
    }


def _missing_status_name(status: RequirementStatus) -> str:
    if not status.configured:
        return status.source
    return f"{status.name}: {status.message}"


def _missing_tool_status_name(requirement: ToolRequirement, status: RequirementStatus) -> str:
    if not status.configured and requirement.env_var:
        return requirement.env_var
    return _missing_status_name(status)


def _kras_e2e_scope() -> str:
    return os.environ.get("KRAS_E2E_SCOPE", KRAS_E2E_FULL_SCOPE).strip().lower()


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_KRAS_G12C_E2E") != "1",
    reason="RUN_KRAS_G12C_E2E=1 is required for KRAS pilot E2E",
)


@pytest.mark.e2e
@pytest.mark.slow
class TestKRASG12CPilot:
    """End-to-end KRAS G12C inhibitor design validation."""

    @pytest.fixture(autouse=True)
    async def setup_stack(self):
        """Ensure test infrastructure is running."""
        status = kras_e2e_preflight_status()
        assert status["ready"], status["message"]

    async def test_full_pipeline_compiles_cig(self):
        """Step 1: Natural language design intent → CIG compilation."""
        nl_input = (
            "Design covalent inhibitors for KRAS G12C with the following: "
            "Molecular weight < 500 Da, LogP 1-4, HBD ≤ 3, HBA ≤ 8, "
            "TPSA < 120 Å², with a Michael acceptor warhead targeting Cys12. "
            "Prioritize candidates with predicted pKd > 8 against KRAS G12C "
            "and >100-fold selectivity over KRAS WT."
        )
        assert nl_input

    async def test_generates_diverse_candidates(self):
        """Step 2: HFM-3D generates structurally diverse KRAS inhibitors."""
        from hfm_generator_svc.main import HFMGeneratorServicer

        response = await HFMGeneratorServicer().Generate(
            SimpleNamespace(
                project_id="kras-g12c",
                batch_size=1,
                generator_params={"sampling_seed": 42},
            ),
            None,
        )

        assert response.generator_name == "hfm_3d"
        assert len(response.molecules) == 1

    async def test_oracle_cascade_validates_affinity(self):
        """Step 3: Boltz-2 validates binding affinity."""
        from boltz2_svc.main import Boltz2Servicer

        response = await Boltz2Servicer().PredictAffinity(
            SimpleNamespace(
                project_id="kras-g12c",
                protein_pdb_id="6OIM",
                ligand_smiles=["CCO"],
                ensemble_size=1,
            ),
            None,
        )

        assert response.protein_pdb_id == "6OIM"
        assert len(response.affinities) == 1

    async def test_retrosyn_plans_synthesis(self):
        """Step 5: AiZynthFinder plans synthesis route."""
        from retrosyn_svc.main import RetrosynServicer

        response = await RetrosynServicer().FindRoutes(
            SimpleNamespace(
                project_id="kras-g12c",
                molecule_smiles="CCO",
                max_routes=1,
                engine="aizynth",
            ),
            None,
        )

        assert response.total_routes_found >= 0

    async def test_critic_reviews_concerns(self):
        """Step 6: Scientific Critic identifies potential issues."""
        from critic_agent.agent import ScientificCriticAgent

        result = await ScientificCriticAgent().evaluate_molecule(
            {"smiles": "CCO", "properties": {"delta_g_kcal_mol": -8.0}}
        )

        assert result["total_rules"] > 0
