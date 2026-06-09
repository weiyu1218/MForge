"""Focused tests for H9 CIG LLM parser/refiner wrappers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_deepseek_semantic_parser_posts_text_and_returns_extracted_intent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "h9_deepseek_semantic_parser_test",
        ROOT / "tools/cig/deepseek_semantic_parser.py",
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://deepseek.example/v1")
    monkeypatch.setenv("CIG_DEEPSEEK_MODEL", "deepseek-v4-flash")
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(
            {
                "url": request.full_url,
                "headers": dict(request.header_items()),
                "body": json.loads(request.data.decode("utf-8")),
                "timeout": timeout,
            }
        )
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "properties": [
                                        {
                                            "name": "solubility",
                                            "direction": "maximize",
                                            "priority": 1,
                                        }
                                    ],
                                    "targets": [{"name": "KRAS G12C"}],
                                    "constraints": {"max_mw": 500},
                                }
                            )
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)

    parsed = module.parse_semantic_text("Design a soluble KRAS G12C inhibitor")

    assert parsed["properties"][0]["name"] == "solubility"
    assert parsed["targets"] == [{"name": "KRAS G12C"}]
    assert calls[0]["url"] == "https://deepseek.example/v1/chat/completions"
    assert calls[0]["headers"]["Authorization"] == "Bearer test-key"
    assert calls[0]["body"]["model"] == "deepseek-v4-flash"
    assert calls[0]["body"]["response_format"] == {"type": "json_object"}
    assert calls[0]["body"]["messages"][-1]["content"].endswith(
        "Design a soluble KRAS G12C inhibitor"
    )


def test_hciv_teacher_projection_returns_valid_lorentz_coordinates() -> None:
    module = _load_module(
        "h9_hciv_teacher_dataset_test",
        ROOT / "tools/cig/build_hciv_teacher_dataset.py",
    )
    from mf_core.geometry import normalize_lorentz_embedding

    coordinates = module.lorentz_coordinates_from_embedding(
        [0.5, -1.0, 0.25],
        dim=8,
        curvature=1.0,
    )

    assert len(coordinates) == 9
    assert normalize_lorentz_embedding(
        coordinates,
        expected_dim=9,
        curvature=1.0,
    ) is not None


def test_hciv_teacher_supports_sklearn_hashing_text_embedder() -> None:
    module = _load_module(
        "h9_hciv_teacher_hashing_test",
        ROOT / "tools/cig/build_hciv_teacher_dataset.py",
    )

    embed = module.load_public_text_embedder("sklearn-hashing")
    vector = embed(
        "target=KRAS G12C; objective=binding_affinity maximize; solubility high"
    )

    assert len(vector) == 384
    assert any(value != 0.0 for value in vector)


def test_deepseek_refiner_uses_checkpoint_to_return_hciv_and_cone(tmp_path) -> None:
    module = _load_module(
        "h9_deepseek_refiner_test",
        ROOT / "tools/cig/deepseek_refiner.py",
    )
    import torch
    from cig_compiler_svc.domain.hciv_encoder import HCIVEncoder

    checkpoint_path = tmp_path / "hciv.pt"
    torch.save(
        {
            "schema": "moleculeforge.cig_compiler.hciv_encoder.v1",
            "state_dict": HCIVEncoder(dim=8).state_dict(),
            "dim": 8,
            "curvature": 1.0,
        },
        checkpoint_path,
    )

    payload = {
        "cig": {"project_id": "cig-original"},
        "feedback": "add solubility constraint",
        "context": {"run_id": "run-1"},
    }

    response = module.refine_payload(
        payload,
        semantic_refiner=lambda _: {
            "properties": [
                {
                    "name": "solubility",
                    "direction": "maximize",
                    "priority": 1,
                }
            ],
            "targets": [{"name": "KRAS G12C"}],
            "constraints": {"max_mw": 500},
        },
        hciv_checkpoint_path=str(checkpoint_path),
        dim=8,
    )

    assert response["cig"]["intent_id"].startswith("cig-")
    assert len(response["hciv"]["coordinates"]) == 9
    assert len(response["intent_cone"]["axis"]) == 9
    assert response["ambiguities"] == []


def test_deepseek_refiner_accepts_nested_cig_llm_output(tmp_path) -> None:
    module = _load_module(
        "h9_deepseek_refiner_nested_test",
        ROOT / "tools/cig/deepseek_refiner.py",
    )
    import torch
    from cig_compiler_svc.domain.hciv_encoder import HCIVEncoder

    checkpoint_path = tmp_path / "hciv.pt"
    torch.save(
        {
            "schema": "moleculeforge.cig_compiler.hciv_encoder.v1",
            "state_dict": HCIVEncoder(dim=8).state_dict(),
            "dim": 8,
            "curvature": 1.0,
        },
        checkpoint_path,
    )

    response = module.refine_payload(
        {
            "cig": {"project_id": "cig-original"},
            "feedback": "add solubility and hERG constraints",
            "context": {},
        },
        semantic_refiner=lambda _: {
            "cig": {
                "properties": ["aqueous_solubility"],
                "targets": [{"name": "KRAS G12C"}],
                "constraints": [{"name": "max_mw", "operator": "less_than", "value": 500}],
                "admet_constraints": [
                    {"name": "hERG", "operator": "less_than", "value": "low"}
                ],
            }
        },
        hciv_checkpoint_path=str(checkpoint_path),
        dim=8,
    )

    assert response["cig"]["intent_id"].startswith("cig-")
    assert len(response["cig"]["objective_nodes"]) >= 2
    assert len(response["hciv"]["coordinates"]) == 9


def test_deepseek_refiner_accepts_target_id_and_property_object(tmp_path) -> None:
    module = _load_module(
        "h9_deepseek_refiner_target_id_test",
        ROOT / "tools/cig/deepseek_refiner.py",
    )
    import torch
    from cig_compiler_svc.domain.hciv_encoder import HCIVEncoder

    checkpoint_path = tmp_path / "hciv.pt"
    torch.save(
        {
            "schema": "moleculeforge.cig_compiler.hciv_encoder.v1",
            "state_dict": HCIVEncoder(dim=8).state_dict(),
            "dim": 8,
            "curvature": 1.0,
        },
        checkpoint_path,
    )

    response = module.refine_payload(
        {
            "cig": {"project_id": "cig-original"},
            "feedback": "preserve KRAS potency",
            "context": {},
        },
        semantic_refiner=lambda _: {
            "properties": {},
            "targets": [
                {
                    "id": "KRAS G12C",
                    "constraint": {"type": "potency", "operator": "<=", "value": 100},
                }
            ],
            "constraints": [],
            "admet_constraints": [
                {"type": "solubility", "description": "aqueous solubility >= 10 uM"}
            ],
            "synthetic_constraints": [],
        },
        hciv_checkpoint_path=str(checkpoint_path),
        dim=8,
    )

    assert response["cig"]["target_context"]["uniprot_ids"] == ["KRAS G12C"]
    assert len(response["hciv"]["coordinates"]) == 9


def test_deepseek_refiner_uses_activity_target_when_targets_missing(tmp_path) -> None:
    module = _load_module(
        "h9_deepseek_refiner_activity_target_test",
        ROOT / "tools/cig/deepseek_refiner.py",
    )
    import torch
    from cig_compiler_svc.domain.hciv_encoder import HCIVEncoder

    checkpoint_path = tmp_path / "hciv.pt"
    torch.save(
        {
            "schema": "moleculeforge.cig_compiler.hciv_encoder.v1",
            "state_dict": HCIVEncoder(dim=8).state_dict(),
            "dim": 8,
            "curvature": 1.0,
        },
        checkpoint_path,
    )

    response = module.refine_payload(
        {
            "cig": {"project_id": "cig-original"},
            "feedback": "preserve KRAS potency",
            "context": {},
        },
        semantic_refiner=lambda _: {
            "activity": {"target": "KRAS G12C", "type": "IC50", "value": 100},
            "properties": {},
            "constraints": [],
            "admet_constraints": [],
            "synthetic_constraints": [],
        },
        hciv_checkpoint_path=str(checkpoint_path),
        dim=8,
    )

    assert response["cig"]["target_context"]["uniprot_ids"] == ["KRAS G12C"]
    assert len(response["hciv"]["coordinates"]) == 9


def test_deepseek_refiner_drops_null_constraints(tmp_path) -> None:
    module = _load_module(
        "h9_deepseek_refiner_null_constraint_test",
        ROOT / "tools/cig/deepseek_refiner.py",
    )
    import torch
    from cig_compiler_svc.domain.hciv_encoder import HCIVEncoder

    checkpoint_path = tmp_path / "hciv.pt"
    torch.save(
        {
            "schema": "moleculeforge.cig_compiler.hciv_encoder.v1",
            "state_dict": HCIVEncoder(dim=8).state_dict(),
            "dim": 8,
            "curvature": 1.0,
        },
        checkpoint_path,
    )

    response = module.refine_payload(
        {
            "cig": {"project_id": "cig-original"},
            "feedback": "preserve KRAS potency",
            "context": {},
        },
        semantic_refiner=lambda _: {
            "properties": [{"name": "solubility"}],
            "targets": [{"name": "KRAS G12C"}],
            "constraints": {"max_mw": None},
            "admet_constraints": {},
            "synthetic_constraints": {},
        },
        hciv_checkpoint_path=str(checkpoint_path),
        dim=8,
    )

    assert "mw_range" not in response["cig"].get("generative_priors", {})
    assert len(response["hciv"]["coordinates"]) == 9
