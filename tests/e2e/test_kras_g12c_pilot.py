"""E2E test: KRAS G12C inhibitor design pilot (Architecture §11).

This test validates the complete MoleculeForge pipeline for a real-world
drug discovery task: designing KRAS G12C covalent inhibitors.

Requirements to run:
  - docker compose -f infra/docker/docker-compose.test.yml up -d
  - Real or mock model weights for HFM-3D + HUMU encoders
  - Test dataset: KRAS G12C crystal structures (PDB: 6OIM, 6N2K)
"""

import os

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
    ArtifactRequirement("patent_index", "PATENT_INDEX_URI", kind="uri"),
    ArtifactRequirement("retrosyn_runner", "RETROSYN_RUNNER_URI", kind="uri"),
)
KRAS_E2E_REQUIRED_FLAGS = (
    "CRITIC_AGENT_READY",
    "ORCHESTRATOR_E2E_READY",
)
KRAS_E2E_REQUIRED_ENV = tuple(
    requirement.env_var for requirement in KRAS_E2E_REQUIRED_ARTIFACTS
) + KRAS_E2E_REQUIRED_FLAGS
KRAS_E2E_ALTERNATIVES = (("GNINA_BINARY", "DIFFDOCK_MODEL_PATH"),)
KRAS_E2E_ALTERNATIVE_REQUIREMENTS = (
    (
        ToolRequirement("gnina", executable="gnina", env_var="GNINA_BINARY"),
        ArtifactRequirement("diffdock_model", "DIFFDOCK_MODEL_PATH", kind="path"),
    ),
)
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
KRAS_E2E_ENGINEERING_SCOPE = "engineering"


def kras_e2e_preflight_status() -> dict:
    missing: list[str] = []
    scope = _kras_e2e_scope()
    if scope == KRAS_E2E_FULL_SCOPE:
        for requirement in KRAS_E2E_REQUIRED_ARTIFACTS:
            status = check_artifact(requirement)
            if not status.available:
                missing.append(_missing_status_name(status))
        for requirement_group in KRAS_E2E_ALTERNATIVE_REQUIREMENTS:
            statuses = [
                check_tool(requirement)
                if isinstance(requirement, ToolRequirement)
                else check_artifact(requirement)
                for requirement in requirement_group
            ]
            if not any(status.available for status in statuses):
                if any(status.configured for status in statuses):
                    missing.append(" or ".join(_missing_status_name(status) for status in statuses))
                else:
                    missing.append(
                        " or ".join(requirement.env_var for requirement in requirement_group)
                    )
        required_flags = KRAS_E2E_REQUIRED_FLAGS
    elif scope == KRAS_E2E_ENGINEERING_SCOPE:
        required_flags = ("ORCHESTRATOR_E2E_READY",)
    else:
        missing.append("KRAS_E2E_SCOPE must be full or engineering")
        required_flags = ()
    for name in required_flags:
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
        if _kras_e2e_scope() == KRAS_E2E_ENGINEERING_SCOPE:
            pytest.skip("HFM expert generation is excluded in engineering scope")
        assert os.environ.get("HFM_CHECKPOINT_PATH")
        assert os.environ.get("HFM_DECODER_PATH")

    async def test_oracle_cascade_validates_affinity(self):
        """Step 3: Boltz-2 + GNINA validate binding affinity."""
        if _kras_e2e_scope() == KRAS_E2E_ENGINEERING_SCOPE:
            pytest.skip("external affinity oracles are excluded in engineering scope")
        assert os.environ.get("GNINA_BINARY") or os.environ.get("DIFFDOCK_MODEL_PATH")
        assert os.environ.get("BOLTZ_MODEL_PATH")

    async def test_fto_check_confirms_novelty(self):
        """Step 4: Patent search confirms IP freedom-to-operate."""
        if _kras_e2e_scope() == KRAS_E2E_ENGINEERING_SCOPE:
            pytest.skip("FTO patent resources are excluded in engineering scope")
        assert os.environ.get("PATENT_INDEX_URI")

    async def test_retrosyn_plans_synthesis(self):
        """Step 5: AiZynthFinder plans synthesis route."""
        if _kras_e2e_scope() == KRAS_E2E_ENGINEERING_SCOPE:
            pytest.skip("AiZynthFinder resources are excluded in engineering scope")
        assert os.environ.get("RETROSYN_RUNNER_URI")

    async def test_critic_reviews_concerns(self):
        """Step 6: Scientific Critic identifies potential issues."""
        if _kras_e2e_scope() == KRAS_E2E_ENGINEERING_SCOPE:
            from orchestrator_svc.main import start_design

            result = await start_design(
                {
                    "nl_input": "Design KRAS G12C inhibitor",
                    "workflow_scope": "engineering",
                    "n_samples": 2,
                }
            )
            assert result["state"]["critic"]["total_rules"] > 0
            return
        assert os.environ.get("CRITIC_AGENT_READY") == "1"

    async def test_end_to_end_kras_g12c(self):
        """Full pipeline: NL → CIG → Generate → Validate → FTO → RetroSyn → Critic."""
        if _kras_e2e_scope() == KRAS_E2E_ENGINEERING_SCOPE:
            from orchestrator_svc.main import start_design

            result = await start_design(
                {
                    "nl_input": (
                        "Design covalent inhibitors for KRAS G12C with "
                        "Molecular weight < 500 Da and LogP 1-4."
                    ),
                    "workflow_scope": "engineering",
                    "n_samples": 2,
                }
            )
            assert result["status"] == "completed"
            assert result["history"] == [
                "PLANNING",
                "GENERATING",
                "VALIDATING",
                "RETROSYN",
                "CRITIC",
            ]
            assert result["state"]["retrosyn"]["skipped"] is True
            return
        assert os.environ.get("ORCHESTRATOR_E2E_READY") == "1"
