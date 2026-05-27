"""Artifact and tool requirement checks."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_artifact_requirement_reports_missing_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from mf_core.artifacts import ArtifactRequirement, check_artifact

    monkeypatch.delenv("HUMU_CHECKPOINT_PATH", raising=False)

    status = check_artifact(ArtifactRequirement("humu_checkpoint", "HUMU_CHECKPOINT_PATH"))

    assert status.name == "humu_checkpoint"
    assert status.configured is False
    assert status.available is False
    assert "HUMU_CHECKPOINT_PATH" in status.message


def test_artifact_requirement_accepts_existing_file(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mf_core.artifacts import ArtifactRequirement, check_artifact

    checkpoint = tmp_path / "humu.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setenv("HUMU_CHECKPOINT_PATH", str(checkpoint))

    status = check_artifact(ArtifactRequirement("humu_checkpoint", "HUMU_CHECKPOINT_PATH"))

    assert status.configured is True
    assert status.available is True
    assert status.path == str(checkpoint)


def test_artifact_requirement_accepts_configured_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    from mf_core.artifacts import ArtifactRequirement, check_artifact

    monkeypatch.setenv("PATENT_INDEX_URI", "s3://mf-indexes/patents")

    status = check_artifact(ArtifactRequirement("patent_index", "PATENT_INDEX_URI", kind="uri"))

    assert status.configured is True
    assert status.available is True
    assert status.path == "s3://mf-indexes/patents"


def test_tool_requirement_resolves_path_tool(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mf_core.artifacts import ToolRequirement, check_tool

    tool = tmp_path / "gnina"
    tool.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    tool.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))

    status = check_tool(ToolRequirement("gnina", executable="gnina"))

    assert status.configured is True
    assert status.available is True
    assert status.path == str(tool)


def test_manifest_loads_artifact_and_tool_requirements(tmp_path) -> None:
    from mf_core.artifacts import load_artifact_manifest

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "artifacts": [
                    {
                        "name": "humu_checkpoint",
                        "env_var": "HUMU_CHECKPOINT_PATH",
                        "kind": "file",
                    }
                ],
                "tools": [
                    {
                        "name": "gnina",
                        "env_var": "GNINA_BINARY",
                        "executable": "gnina",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = load_artifact_manifest(manifest_path)

    assert manifest.schema_version == "1.0"
    assert manifest.artifacts[0].env_var == "HUMU_CHECKPOINT_PATH"
    assert manifest.tools[0].executable == "gnina"


def test_python_package_requirement_detects_importable_package() -> None:
    from mf_core.artifacts import PythonPackageRequirement, check_python_package

    status = check_python_package(PythonPackageRequirement("json", module="json"))

    assert status.configured is True
    assert status.available is True
    assert status.path == "json"


def test_require_available_raises_with_missing_names(monkeypatch: pytest.MonkeyPatch) -> None:
    from mf_core.artifacts import ArtifactRequirement, check_artifact, require_available

    monkeypatch.delenv("ADMET_MODEL_PATH", raising=False)
    status = check_artifact(ArtifactRequirement("admet_model", "ADMET_MODEL_PATH"))

    with pytest.raises(RuntimeError, match="admet_model"):
        require_available([status])


def test_project_artifact_manifest_records_core_dependencies() -> None:
    from mf_core.artifacts import load_artifact_manifest

    manifest = load_artifact_manifest(ROOT / "models/artifacts/manifest.json")

    artifact_names = {requirement.name for requirement in manifest.artifacts}
    tool_names = {requirement.name for requirement in manifest.tools}
    package_names = {requirement.name for requirement in manifest.python_packages}

    assert {
        "humu_checkpoint",
        "hfm_checkpoint",
        "hfm_decoder",
        "fragfm_vocabulary",
        "admet_model",
        "boltz_model",
        "feast_repo",
    }.issubset(artifact_names)
    assert {"gnina", "boltz", "openfe_runner", "openbabel"}.issubset(tool_names)
    assert {"rdkit", "openfe"}.issubset(package_names)
