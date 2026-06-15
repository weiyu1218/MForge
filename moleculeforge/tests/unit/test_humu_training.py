"""Smoke tests for HUMU pretraining pipeline (training loop + checkpointing)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
for rel_path in (
    "libs/mf-core/src",
    "libs/mf-humu/src",
    "models/mf-encoders/humu_mol_encoder/src",
    "models/mf-encoders/humu_pocket_encoder/src",
    "models/mf-encoders/humu_route_encoder/src",
    "pipelines/humu_pretrain/src",
    "models/mf-generators/hfm_3d/src",
):
    sys.path.insert(0, str(ROOT / rel_path))


def _write_minimal_humu_sources(tmp_path: Path) -> dict[str, Path]:
    sources = {
        "mol": tmp_path / "mol",
        "pocket": tmp_path / "pocket",
        "route": tmp_path / "route",
        "joint": tmp_path / "joint",
        "activity": tmp_path / "activity",
        "protacpedia": tmp_path / "protacpedia",
        "protacdb": tmp_path / "protacdb",
        "route_eval": tmp_path / "route_eval",
        "retropath_templates": tmp_path / "retropath_templates",
        "protac8k": tmp_path / "protac8k",
        "rcsb_mmcif": tmp_path / "rcsb_mmcif",
        "interface_skempi2": tmp_path / "interface_skempi2",
        "pdcdb": tmp_path / "pdcdb",
    }
    for directory in sources.values():
        directory.mkdir()

    (sources["mol"] / "manifest.json").write_text(
        json.dumps({"shards": ["shard_0000.jsonl"], "n_records": 2}),
        encoding="utf-8",
    )
    (sources["mol"] / "shard_0000.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"smiles": "CCO", "inchikey": "mol-1"}),
                json.dumps({"smiles": "CCN", "inchikey": "mol-2"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (sources["pocket"] / "index.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "index": 0,
                        "pdb_id": "1ABC_A",
                        "pocket_path": "pocket_000000.json",
                        "ligand_smiles": "CCO",
                    }
                ),
                json.dumps(
                    {
                        "index": 1,
                        "pdb_id": "2ABC_A",
                        "pocket_path": "pocket_000001.json",
                        "ligand_smiles": "CCN",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    for index in range(2):
        (sources["pocket"] / f"pocket_{index:06d}.json").write_text(
            json.dumps(
                {
                    "pocket_atoms": [
                        {"x": 0.0, "y": 0.0, "z": 0.0, "element": "C", "residue": "ALA"}
                    ],
                    "protein_sequence": "AAAA",
                }
            ),
            encoding="utf-8",
        )

    (sources["route"] / "routes.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "route-1",
                        "root_smiles": "CCC",
                        "reaction_smiles": "CCO>>CCC",
                        "source_split": "train",
                    }
                ),
                json.dumps(
                    {
                        "id": "route-2",
                        "root_smiles": "CCCl",
                        "reaction_smiles": "CCO>>CCCl",
                        "source_split": "train",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (sources["joint"] / "joint.jsonl").write_text(
        json.dumps(
            {
                "id": "joint-1",
                "pdb_id": "1ABC_A",
                "pocket_path": "pocket_joint.json",
                "ligand_smiles": "CCO",
                "route_id": "route-joint-1",
                "reactions": ["CCN>>CCO"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (sources["joint"] / "pocket_joint.json").write_text(
        json.dumps(
            {
                "pocket_atoms": [
                    {"x": 0.0, "y": 0.0, "z": 0.0, "element": "C", "residue": "ALA"}
                ],
                "protein_sequence": "AAAA",
            }
        ),
        encoding="utf-8",
    )

    (sources["activity"] / "activity.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "ligand_smiles": "CCO",
                        "target_id": "target-1",
                        "activity_value": 8.0,
                    }
                ),
                json.dumps(
                    {
                        "ligand_smiles": "CCN",
                        "target_id": "target-1",
                        "activity_value": 6.0,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (sources["protacpedia"] / "protacpedia.jsonl").write_text(
        json.dumps(
            {
                "protacdb_id": "p1",
                "protac_canonical_smiles": "CCCOCCN",
                "e3_binder_canonical_smiles": "CCO",
                "ligand_canonical_smiles": "CCN",
                "linker_canonical_smiles": "COC",
                "source": "PROTACpedia",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (sources["protacdb"] / "e3_ligand.jsonl").write_text(
        json.dumps(
            {
                "record_id": "protacdb_e3_ligand_1",
                "canonical_smiles": "NC1=CC=CC=C1",
                "smiles": "NC1=CC=CC=C1",
                "component": "e3_ligand",
                "smiles_valid": True,
                "source": "PROTAC-DB",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (sources["route_eval"] / "routes_valid.jsonl").write_text(
        json.dumps(
            {
                "id": "route-eval-1",
                "root_smiles": "CCO",
                "reaction_smiles": "CCN>>CCO",
                "source_split": "valid",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (sources["retropath_templates"] / "templates.jsonl").write_text(
        json.dumps(
            {
                "template_id": "template-1",
                "template": "[C:1]>>[C:1]O",
                "source_dataset": "fixture",
                "valid": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (sources["protac8k"] / "manifest.json").write_text(
        json.dumps({"format": "protac8k_archive_index", "n_files": 3}),
        encoding="utf-8",
    )
    pocket_payload = {
        "coords": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        "elements": ["C", "N"],
        "residue_types": ["ALA", "LYS"],
        "protein_sequence": "AK",
    }
    (sources["protac8k"] / "protac8k.jsonl").write_text(
        json.dumps(
            {
                "record_id": "protac8k-1",
                "protac_smiles": "CCCOCCN",
                "target_ligand_smiles": "CCO",
                "e3_ligand_smiles": "CCN",
                "target_pocket": pocket_payload,
                "e3_pocket": pocket_payload,
                "source": "PROTAC-8K",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (sources["rcsb_mmcif"] / "structures.jsonl").write_text(
        json.dumps(
            {
                "pdb_id": "1abc",
                "interface": pocket_payload,
                "source_tags": ["fixture"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (sources["interface_skempi2"] / "skempi2.jsonl").write_text(
        json.dumps(
            {
                "id": "SKEMPI2:1",
                "pdb_complex": "1ABC_A_B",
                "mutations_cleaned": "AA1G",
                "affinity_mut_m": "1e-8",
                "affinity_wt_m": "1e-9",
                "wt_interface": pocket_payload,
                "mut_interface": {
                    **pocket_payload,
                    "residue_types": ["GLY", "LYS"],
                    "protein_sequence": "GK",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (sources["pdcdb"] / "pdc.jsonl").write_text(
        json.dumps(
            {
                "PDC_ID": "PDC_1",
                "Peptide_Sequence": "CCKIGLFRWR",
                "Linker_ID": "LIN1",
                "Payload_Name": "Doxorubicin",
                "source": "PDCdb",
                "record_type": "pdc",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (sources["pdcdb"] / "pdc_components.jsonl").write_text(
        json.dumps(
            {
                "record_id": "PDC_1",
                "peptide_sequence": "CCKIGLFRWR",
                "peptide_pocket": pocket_payload,
                "linker_smiles": "O=C(O)CC(=O)O",
                "payload_smiles": "COC",
                "source": "PDCdb",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return sources


def test_validate_config_defaults():
    from humu_pretrain.pipeline import _validate_config
    cfg = _validate_config({})
    assert cfg["batch_size"] == 64
    assert cfg["epochs"] == 100
    assert cfg["learning_rate"] == 1e-4
    assert cfg["embed_dim"] == 129
    assert cfg["use_amp"] is False


def test_amp_dtype_from_config_accepts_bfloat16_aliases():
    from humu_pretrain.pipeline import _amp_dtype_from_config

    assert _amp_dtype_from_config({"amp_dtype": "bfloat16"}) is torch.bfloat16
    assert _amp_dtype_from_config({"amp_dtype": "bf16"}) is torch.bfloat16


def test_amp_dtype_from_config_accepts_float16_aliases():
    from humu_pretrain.pipeline import _amp_dtype_from_config

    assert _amp_dtype_from_config({"amp_dtype": "float16"}) is torch.float16
    assert _amp_dtype_from_config({"amp_dtype": "fp16"}) is torch.float16
    assert _amp_dtype_from_config({"amp_dtype": "half"}) is torch.float16
    assert _amp_dtype_from_config({}) is torch.float16


def test_amp_dtype_from_config_rejects_invalid_dtype():
    from humu_pretrain.pipeline import _amp_dtype_from_config

    with pytest.raises(ValueError, match="amp_dtype"):
        _amp_dtype_from_config({"amp_dtype": "float32"})


def test_cuda_sdp_backend_config_defaults_disable_cudnn_sdp():
    from humu_pretrain.pipeline import _cuda_sdp_backend_config

    assert _cuda_sdp_backend_config({}) == {"enable_cudnn_sdp": False}


def test_cuda_sdp_backend_config_accepts_explicit_cudnn_sdp_toggle():
    from humu_pretrain.pipeline import _cuda_sdp_backend_config

    assert _cuda_sdp_backend_config({"cuda_backends": {"enable_cudnn_sdp": True}}) == {
        "enable_cudnn_sdp": True
    }
    assert _cuda_sdp_backend_config({"cuda_backends": {"enable_cudnn_sdp": False}}) == {
        "enable_cudnn_sdp": False
    }


def test_validate_config_override():
    from humu_pretrain.pipeline import _validate_config
    cfg = _validate_config({"batch_size": 32, "learning_rate": 5e-3})
    assert cfg["batch_size"] == 32
    assert cfg["learning_rate"] == 5e-3
    assert cfg["epochs"] == 100  # default preserved


def test_build_encoders():
    from humu_pretrain.pipeline import _build_encoders
    cfg = {"embed_dim": 129, "curvature": 1.0, "encoders": {}}
    device = torch.device("cpu")
    encoders = _build_encoders(cfg, device)
    assert "mol" in encoders
    assert "pocket" in encoders
    assert "route" in encoders
    assert "intent" not in encoders
    for _name, model in encoders.items():
        assert isinstance(model, torch.nn.Module)


def test_validate_config_rejects_pretrain_intent_residuals():
    from humu_pretrain.pipeline import _validate_config

    with pytest.raises(ValueError, match="loss_weights.intent"):
        _validate_config({"loss_weights": {"intent": 1.0}})

    with pytest.raises(ValueError, match="data.intent_source"):
        _validate_config({"data": {"intent_source": "/tmp/intent"}})

    with pytest.raises(ValueError, match="data.joint_oversample_factor"):
        _validate_config({"data": {"joint_oversample_factor": 4}})


def test_preflight_rejects_pretrain_intent_residuals(tmp_path):
    from humu_pretrain.data_loader import preflight_humu_data_contract

    pocket_dir = tmp_path / "pocket"
    route_dir = tmp_path / "route"
    intent_dir = tmp_path / "intent"
    for directory in (pocket_dir, route_dir, intent_dir):
        directory.mkdir()

    with pytest.raises(ValueError, match="loss_weights.intent"):
        preflight_humu_data_contract(
            {
                "loss_weights": {"intent": 1.0},
                "data": {
                    "pocket_source": str(pocket_dir),
                    "route_source": str(route_dir),
                },
            }
        )

    with pytest.raises(ValueError, match="data.intent_source"):
        preflight_humu_data_contract(
            {
                "loss_weights": {},
                "data": {
                    "pocket_source": str(pocket_dir),
                    "route_source": str(route_dir),
                    "intent_source": str(intent_dir),
                },
            }
        )


def test_contrastive_loss_no_none():
    from humu_pretrain.pipeline import _contrastive_loss
    from mf_humu.manifold.lorentz import LorentzManifold
    manifold = LorentzManifold(curvature=1.0)
    # Create two random points on the Lorentz manifold
    v1 = torch.randn(2, 128)
    x1 = torch.cat([torch.sqrt(1 + v1.pow(2).sum(dim=-1, keepdim=True)), v1], dim=-1)
    v2 = torch.randn(2, 128)
    x2 = torch.cat([torch.sqrt(1 + v2.pow(2).sum(dim=-1, keepdim=True)), v2], dim=-1)
    loss = _contrastive_loss(x1, x2, manifold)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0


def test_encoder_wrapper_forward():
    from humu_pretrain.pipeline import _build_encoders
    cfg = {"embed_dim": 129, "curvature": 1.0, "encoders": {}}
    device = torch.device("cpu")
    encoders = _build_encoders(cfg, device)
    mol_enc = encoders["mol"]
    emb = mol_enc.encode_batch(["CCO", "c1ccccc1"])
    assert emb.shape == (2, 129)
    # Should be approximately on the Lorentz manifold: x0^2 - sum(xi^2) = 1
    for i in range(2):
        norm = -emb[i, 0].item() ** 2 + emb[i, 1:].pow(2).sum().item()
        assert abs(norm + 1.0) < 1.1, f"Off manifold: norm={norm}"


def test_molecule_encoder_uses_rdkit_graph_features_without_sampling(monkeypatch):
    from mf_encoders.humu_mol.encoder import HUMUMoleculeEncoder

    def forbid_randn(*args, **kwargs):
        raise AssertionError("molecule encoder must not sample graph embeddings")

    monkeypatch.setattr(torch, "randn", forbid_randn)
    encoder = HUMUMoleculeEncoder(dim=128, curvature=1.0)

    emb = encoder.encode("CCO")

    assert emb.shape == (1, 129)
    assert torch.isfinite(emb).all()
    with pytest.raises(ValueError, match="valid SMILES"):
        encoder.encode("not_a_smiles")


def test_molecule_encoder_encodes_valence_outlier_smiles():
    from mf_encoders.humu_mol.encoder import HUMUMoleculeEncoder

    encoder = HUMUMoleculeEncoder(dim=8, curvature=1.0)

    embedding = encoder.encode("B(F)(F)(F)F")

    assert embedding.shape == (1, 9)
    assert torch.isfinite(embedding).all()


def test_molecule_encoder_uses_3d_geometry_invariant_features():
    from mf_encoders.humu_mol.encoder import HUMUMoleculeEncoder

    encoder = HUMUMoleculeEncoder(dim=8, curvature=1.0)
    base = {
        "smiles": "CCO",
        "coords": [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [2.1, 0.8, 0.0]],
    }
    rotated_translated = {
        "smiles": "CCO",
        "coords": [[2.0, -1.0, 0.5], [2.0, 0.4, 0.5], [1.2, 1.1, 0.5]],
    }
    stretched = {
        "smiles": "CCO",
        "coords": [[0.0, 0.0, 0.0], [2.2, 0.0, 0.0], [3.4, 1.3, 0.0]],
    }

    base_embedding = encoder.encode(base)
    rotated_embedding = encoder.encode(rotated_translated)
    stretched_embedding = encoder.encode(stretched)

    assert torch.allclose(base_embedding, rotated_embedding, atol=1e-5)
    assert not torch.allclose(base_embedding, stretched_embedding)


def test_molecule_encoder_keeps_legacy_device_attribute_compatible():
    from mf_encoders.humu_mol.encoder import HUMUMoleculeEncoder

    encoder = HUMUMoleculeEncoder(dim=128, curvature=1.0)
    encoder._device = torch.device("cpu")

    emb = encoder.encode("CCO")

    assert emb.shape == (1, 129)


def test_pocket_encoder_requires_coordinates_without_sampling(monkeypatch):
    from mf_encoders.humu_pocket.encoder import HUMUPocketEncoder

    def forbid_randn(*args, **kwargs):
        raise AssertionError("pocket encoder must not sample missing coordinates")

    monkeypatch.setattr(torch, "randn", forbid_randn)
    encoder = HUMUPocketEncoder(dim=128, curvature=1.0)

    with pytest.raises(ValueError, match="coords"):
        encoder.encode({"pdb_id": "1ABC"})

    emb = encoder.encode(
        {
            "pdb_id": "1ABC",
            "coords": [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]],
            "elements": ["C", "N", "O"],
            "residue_types": ["ALA", "LYS", "ASP"],
        }
    )

    assert emb.shape == (1, 129)
    assert torch.isfinite(emb).all()


def test_pocket_encoder_uses_e3_invariant_geometry_features():
    from mf_encoders.humu_pocket.encoder import HUMUPocketEncoder

    encoder = HUMUPocketEncoder(dim=8, curvature=1.0)
    base = {
        "coords": [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]],
        "elements": ["C", "N", "O"],
        "residue_types": ["ALA", "LYS", "ASP"],
    }
    rotated_translated = {
        "coords": [[2.0, -1.0, 0.5], [2.0, 0.5, 0.5], [0.5, -1.0, 0.5]],
        "elements": ["C", "N", "O"],
        "residue_types": ["ALA", "LYS", "ASP"],
    }
    stretched = {
        "coords": [[0.0, 0.0, 0.0], [2.4, 0.0, 0.0], [0.0, 1.5, 0.0]],
        "elements": ["C", "N", "O"],
        "residue_types": ["ALA", "LYS", "ASP"],
    }

    base_embedding = encoder.encode(base)
    rotated_embedding = encoder.encode(rotated_translated)
    stretched_embedding = encoder.encode(stretched)

    assert torch.allclose(base_embedding, rotated_embedding, atol=1e-5)
    assert not torch.allclose(base_embedding, stretched_embedding)


def test_pocket_encoder_uses_precomputed_esm2_embedding():
    from mf_encoders.humu_pocket.encoder import HUMUPocketEncoder

    pocket = {
        "pdb_id": "1ABC",
        "coords": [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]],
        "elements": ["C", "N", "O"],
        "residue_types": ["ALA", "LYS", "ASP"],
    }
    encoder = HUMUPocketEncoder(
        dim=8,
        curvature=1.0,
        use_esm2=True,
        esm2_dim=4,
    )

    emb_a = encoder.encode({**pocket, "esm2_embedding": [1.0, 0.0, 0.0, 0.0]})
    emb_b = encoder.encode({**pocket, "esm2_embedding": [0.0, 1.0, 0.0, 0.0]})

    assert emb_a.shape == (1, 9)
    assert emb_b.shape == (1, 9)
    assert not torch.allclose(emb_a, emb_b)


def test_pocket_encoder_requires_esm2_input_when_enabled():
    from mf_encoders.humu_pocket.encoder import HUMUPocketEncoder

    encoder = HUMUPocketEncoder(
        dim=8,
        curvature=1.0,
        use_esm2=True,
        esm2_dim=4,
        esm2_required_sources=["pocket"],
    )

    with pytest.raises(ValueError, match="ESM-2"):
        encoder.encode(
            {
                "pdb_id": "1ABC",
                "source_name": "pocket",
                "coords": [[0.0, 0.0, 0.0]],
                "elements": ["C"],
                "residue_types": ["ALA"],
            }
        )


def test_pocket_encoder_allows_geometry_only_for_structure_source_when_esm2_enabled():
    from mf_encoders.humu_pocket.encoder import HUMUPocketEncoder

    encoder = HUMUPocketEncoder(
        dim=8,
        curvature=1.0,
        use_esm2=True,
        esm2_dim=4,
        esm2_required_sources=["pocket", "joint"],
    )

    embedding = encoder.encode(
        {
            "pdb_id": "1ABC",
            "source_name": "rcsb_mmcif",
            "coords": [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
            "elements": ["C", "N"],
            "residue_types": ["ALA", "LYS"],
        }
    )

    assert embedding.shape == (1, 9)
    assert torch.isfinite(embedding).all()


def test_pocket_encoder_reuses_sequence_esm2_embedding(monkeypatch):
    from mf_encoders.humu_pocket.encoder import HUMUPocketEncoder

    pocket = {
        "pdb_id": "1ABC",
        "coords": [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
        "elements": ["C", "N"],
        "residue_types": ["ALA", "LYS"],
        "protein_sequence": "ACDE",
    }
    encoder = HUMUPocketEncoder(dim=8, curvature=1.0, use_esm2=True, esm2_dim=4)
    calls = []

    def fake_compute(sequence: str):
        calls.append(sequence)
        return torch.ones(4)

    monkeypatch.setattr(encoder, "_compute_esm2_embedding", fake_compute)

    encoder.encode(pocket)
    encoder.encode(dict(pocket))

    assert calls == ["ACDE"]


def test_pocket_encoder_rejects_overlong_esm2_sequence_before_loading_model():
    from mf_encoders.humu_pocket.encoder import HUMUPocketEncoder

    encoder = HUMUPocketEncoder(
        dim=8,
        use_esm2=True,
        esm2_checkpoint="missing.pt",
        esm2_max_sequence_length=3,
    )

    with pytest.raises(ValueError, match="ESM-2 sequence length"):
        encoder.encode(
            {
                "coords": [[0.0, 0.0, 0.0]],
                "elements": ["C"],
                "residue_types": ["ALA"],
                "protein_sequence": "AAAA",
            }
        )


def test_pocket_encoder_batches_sequence_esm2_embeddings(monkeypatch):
    from mf_encoders.humu_pocket.encoder import HUMUPocketEncoder

    pockets = [
        {
            "pdb_id": "1ABC",
            "coords": [[0.0, 0.0, 0.0]],
            "elements": ["C"],
            "residue_types": ["ALA"],
            "protein_sequence": "ACDE",
        },
        {
            "pdb_id": "2ABC",
            "coords": [[1.0, 0.0, 0.0]],
            "elements": ["N"],
            "residue_types": ["LYS"],
            "protein_sequence": "FGHI",
        },
    ]
    encoder = HUMUPocketEncoder(dim=8, curvature=1.0, use_esm2=True, esm2_dim=4)
    calls = []

    def fake_batch_compute(sequences: list[str]):
        calls.append(sequences)
        return torch.stack(
            [
                torch.full((4,), float(index + 1))
                for index, _ in enumerate(sequences)
            ]
        )

    def forbid_single_compute(sequence: str):
        raise AssertionError("encode_batch must batch ESM-2 sequence embeddings")

    monkeypatch.setattr(encoder, "_compute_esm2_batch_embeddings", fake_batch_compute)
    monkeypatch.setattr(encoder, "_compute_esm2_embedding", forbid_single_compute)

    embedding = encoder.encode_batch(pockets)

    assert embedding.shape == (2, 9)
    assert calls == [["ACDE", "FGHI"]]


def test_build_encoders_passes_pocket_esm2_config():
    from humu_pretrain.pipeline import _build_encoders

    cfg = {
        "embed_dim": 9,
        "curvature": 1.0,
        "encoders": {
            "pocket": {
                "use_esm2": True,
                "esm2_checkpoint": "models/esm2/esm2_t33_650M_UR50D.pt",
                "esm2_dim": 4,
                "esm2_layer": 33,
                "esm2_required_sources": ["pocket", "joint"],
            }
        },
    }

    encoders = _build_encoders(cfg, torch.device("cpu"))

    assert encoders["pocket"].inner.use_esm2 is True
    assert encoders["pocket"].inner.esm2_checkpoint == "models/esm2/esm2_t33_650M_UR50D.pt"
    assert encoders["pocket"].inner.esm2_dim == 4
    assert encoders["pocket"].inner.esm2_layer == 33
    assert encoders["pocket"].inner.esm2_required_sources == {"pocket", "joint"}


def test_setup_distributed_uses_configured_timeout(monkeypatch):
    from datetime import timedelta
    import humu_pretrain.pipeline as pipeline
    from humu_pretrain.pipeline import DistributedContext

    calls = {}

    def fake_init_process_group(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr(pipeline.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(pipeline.dist, "init_process_group", fake_init_process_group)

    pipeline._setup_distributed(
        DistributedContext(enabled=True, rank=1, world_size=4, local_rank=1),
        torch.device("cuda", 1),
        {
            "distributed_backend": "nccl",
            "distributed_timeout_seconds": 3600,
        },
    )

    assert calls["backend"] == "nccl"
    assert calls["rank"] == 1
    assert calls["world_size"] == 4
    assert calls["timeout"] == timedelta(seconds=3600)


def test_h200_runner_exports_nccl_heartbeat_timeout():
    env_file = (ROOT / "pipelines/humu_pretrain/.env").read_text(encoding="utf-8")
    script = (ROOT / "pipelines/humu_pretrain/run_humu_4h200_background.sh").read_text(
        encoding="utf-8"
    )

    assert (
        'TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"'
        in env_file
    )
    assert (
        'TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-1}"'
        in env_file
    )
    assert "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC" in script.partition("export ")[2]


def test_build_encoders_consumes_architecture_config():
    from humu_pretrain.pipeline import _build_encoders

    cfg = {
        "embed_dim": 17,
        "curvature": 1.0,
        "encoders": {
            "mol": {
                "hidden_dim": 32,
                "n_layers": 3,
                "n_heads": 2,
                "dropout": 0.25,
                "use_3d_geometry": True,
            },
            "pocket": {
                "hidden_dim": 24,
                "n_layers": 2,
                "n_heads": 4,
                "dropout": 0.2,
                "radius_angstrom": 4.5,
                "max_neighbors": 3,
            },
            "route": {
                "hidden_dim": 28,
                "n_layers": 2,
                "n_heads": 4,
                "dropout": 0.15,
                "use_tree_pooling": True,
            },
        },
    }

    encoders = _build_encoders(cfg, torch.device("cpu"))

    mol = encoders["mol"].inner
    pocket = encoders["pocket"].inner
    route = encoders["route"].inner
    assert mol.hidden_dim == 32
    assert mol.n_layers == 3
    assert mol.n_heads == 2
    assert mol.dropout_p == pytest.approx(0.25)
    assert mol.use_3d_geometry is True
    assert pocket.hidden_dim == 24
    assert pocket.n_layers == 2
    assert pocket.n_heads == 4
    assert pocket.dropout_p == pytest.approx(0.2)
    assert pocket.radius_angstrom == pytest.approx(4.5)
    assert pocket.max_neighbors == 3
    assert route.hidden_dim == 28
    assert route.n_layers == 2
    assert route.n_heads == 4
    assert route.dropout_p == pytest.approx(0.15)
    assert route.use_tree_pooling is True


def test_route_encoder_requires_reaction_graph_without_sampling(monkeypatch):
    from mf_encoders.humu_route.encoder import HUMURouteEncoder

    def forbid_randn(*args, **kwargs):
        raise AssertionError("route encoder must not sample route embeddings")

    monkeypatch.setattr(torch, "randn", forbid_randn)
    encoder = HUMURouteEncoder(dim=128, curvature=1.0)

    with pytest.raises(ValueError, match="reactions"):
        encoder.encode({"root_smiles": "CCO"})

    emb = encoder.encode(
        {
            "reactions": ["CCO>>CC=O"],
            "steps": 1,
            "score": 0.8,
            "intermediates": ["CCO", "CC=O"],
        }
    )

    assert emb.shape == (1, 129)
    assert torch.isfinite(emb).all()


def test_route_encoder_uses_reaction_tree_topology():
    from mf_encoders.humu_route.encoder import HUMURouteEncoder

    reactions = [
        "CCO>>CC=O",
        "CC=O>>CC(=O)O",
        "CC(=O)O>>CC(=O)Cl",
    ]
    linear_route = {
        "reactions": reactions,
        "steps": [
            {"step_id": "s1", "reaction": reactions[0]},
            {"step_id": "s2", "reaction": reactions[1], "parent_step_id": "s1"},
            {"step_id": "s3", "reaction": reactions[2], "parent_step_id": "s2"},
        ],
    }
    branched_route = {
        "reactions": reactions,
        "steps": [
            {"step_id": "s1", "reaction": reactions[0]},
            {"step_id": "s2", "reaction": reactions[1], "parent_step_id": "s1"},
            {"step_id": "s3", "reaction": reactions[2], "parent_step_id": "s1"},
        ],
    }
    encoder = HUMURouteEncoder(dim=32, curvature=1.0)

    linear_embedding = encoder.encode(linear_route)
    branched_embedding = encoder.encode(branched_route)

    assert linear_embedding.shape == (1, 33)
    assert branched_embedding.shape == (1, 33)
    assert not torch.allclose(linear_embedding, branched_embedding)


@pytest.mark.asyncio
async def test_pretrain_encoder_wrappers_do_not_claim_training_success():
    from humu_pretrain.pipeline import pretrain_molecule_encoder

    with pytest.raises(RuntimeError, match="run"):
        await pretrain_molecule_encoder({})


@pytest.mark.asyncio
async def test_training_loop_applies_warmup_lr_before_optimizer_step(monkeypatch, tmp_path):
    import torch.optim as optim
    import humu_pretrain.data_loader as data_loader_module
    import humu_pretrain.pipeline as pipeline_module

    step_lrs: list[float] = []

    class RecordingAdamW(optim.AdamW):
        def step(self, *args, **kwargs):
            step_lrs.append(float(self.param_groups[0]["lr"]))
            return super().step(*args, **kwargs)

    def build_encoders(_cfg, _device):
        return {"mol": torch.nn.Linear(1, 1, bias=False)}

    def create_dataloaders(_cfg):
        return {"paired": [{}, {}, {}, {}]}

    def forward_batch(encoders, _batch, _cfg):
        loss = sum(param.pow(2).sum() for model in encoders.values() for param in model.parameters())
        return {"total": loss}

    monkeypatch.setattr(pipeline_module, "AdamW", RecordingAdamW)
    monkeypatch.setattr(pipeline_module, "_build_encoders", build_encoders)
    monkeypatch.setattr(data_loader_module, "create_dataloaders", create_dataloaders)
    monkeypatch.setattr(pipeline_module, "_forward_paired_batch", forward_batch)
    monkeypatch.setattr(pipeline_module, "_log_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline_module, "_save_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline_module, "_clear_progress_line", lambda: None)

    await pipeline_module.run(
        {
            "embed_dim": 2,
            "curvature": 1.0,
            "learning_rate": 3.0e-4,
            "weight_decay": 0.0,
            "epochs": 1,
            "warmup_steps": 4,
            "gradient_clip_norm": 0.0,
            "use_amp": False,
            "device": "cpu",
            "output_dir": str(tmp_path),
            "save_every_n_epochs": 100,
            "save_every_n_steps": 0,
            "logging": {"log_every_n_steps": 1000},
            "eval": {"every_n_epochs": 0},
            "data": {
                "pocket_source": str(tmp_path),
                "route_source": str(tmp_path),
            },
        }
    )

    assert step_lrs == pytest.approx([7.5e-5, 1.5e-4, 2.25e-4, 3.0e-4])


def test_apply_lr_schedule_only_steps_cosine_scheduler():
    from humu_pretrain.pipeline import _apply_lr_schedule

    class RecordingScheduler:
        def __init__(self):
            self.calls: list[float] = []

        def step(self, value):
            self.calls.append(float(value))

    scheduler = RecordingScheduler()

    _apply_lr_schedule(scheduler, epoch=1, step=5, n_batches=20)

    assert scheduler.calls == pytest.approx([1.25])


def test_torchrun_distributed_context_from_environment(monkeypatch):
    from humu_pretrain.pipeline import _distributed_context_from_env

    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "2")

    context = _distributed_context_from_env()

    assert context.enabled is True
    assert context.rank == 2
    assert context.world_size == 4
    assert context.local_rank == 2


def test_distributed_wrap_allows_unused_tower_parameters(monkeypatch):
    from humu_pretrain import pipeline
    from humu_pretrain.pipeline import DistributedContext

    calls = []

    class FakeDDP(torch.nn.Module):
        def __init__(
            self,
            model,
            device_ids=None,
            find_unused_parameters=False,
        ):
            super().__init__()
            self.model = model
            calls.append(
                {
                    "device_ids": device_ids,
                    "find_unused_parameters": find_unused_parameters,
                }
            )

    monkeypatch.setattr(pipeline, "DistributedDataParallel", FakeDDP)
    encoders = {
        "mol": torch.nn.Linear(1, 1),
        "pocket": torch.nn.Linear(1, 1),
        "route": torch.nn.Linear(1, 1),
    }

    pipeline._wrap_distributed(
        encoders,
        DistributedContext(enabled=True, rank=0, world_size=4, local_rank=0),
        torch.device("cuda", 0),
    )

    assert calls
    assert all(call["find_unused_parameters"] is False for call in calls)


def test_distributed_wrap_can_enable_unused_parameter_detection(monkeypatch):
    from humu_pretrain import pipeline
    from humu_pretrain.pipeline import DistributedContext

    calls = []

    class FakeDDP(torch.nn.Module):
        def __init__(
            self,
            model,
            device_ids=None,
            find_unused_parameters=False,
        ):
            super().__init__()
            self.model = model
            calls.append(find_unused_parameters)

    monkeypatch.setattr(pipeline, "DistributedDataParallel", FakeDDP)
    encoders = {"mol": torch.nn.Linear(1, 1)}

    pipeline._wrap_distributed(
        encoders,
        DistributedContext(enabled=True, rank=0, world_size=4, local_rank=0),
        torch.device("cuda", 0),
        find_unused_parameters=True,
    )

    assert calls == [True]


def test_distributed_batch_failure_syncs_across_ranks(monkeypatch):
    from humu_pretrain import pipeline
    from humu_pretrain.pipeline import DistributedContext, _distributed_batch_failed

    def fake_all_reduce(tensor, op=None):
        tensor.fill_(1)

    monkeypatch.setattr(pipeline.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(pipeline.dist, "all_reduce", fake_all_reduce)

    failed = _distributed_batch_failed(
        DistributedContext(enabled=True, rank=0, world_size=2, local_rank=0),
        torch.device("cpu"),
        local_failed=False,
    )

    assert failed is True


def test_encode_items_at_indices_calls_tower_once_for_selected_records():
    from humu_pretrain.pipeline import _encode_items_at_indices

    class FakeTower(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, items):
            self.calls.append(items)
            return torch.ones(len(items), 129)

    tower = FakeTower()
    items = [{"id": "skip"}, {"id": "a"}, {"id": "b"}]

    embeddings, zero_loss = _encode_items_at_indices(tower, items, [1, 2], {"id": "dummy"})

    assert embeddings.shape == (2, 129)
    assert zero_loss is None
    assert tower.calls == [[{"id": "a"}, {"id": "b"}]]


def test_encode_items_at_indices_keeps_ddp_path_for_empty_tower_batch():
    from humu_pretrain.pipeline import _encode_items_at_indices

    class FakeTower(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.calls = []

        def forward(self, items):
            self.calls.append(items)
            return self.weight.expand(len(items), 129)

    tower = FakeTower()
    dummy = {"id": "dummy"}

    embeddings, zero_loss = _encode_items_at_indices(tower, [], [], dummy)
    zero_loss.backward()

    assert embeddings is None
    assert tower.calls == [[dummy]]
    assert zero_loss.item() == 0.0
    assert tower.weight.grad is not None
    assert tower.weight.grad.item() == 0.0


def test_prepare_distributed_loaders_preserves_non_shuffled_loader():
    from humu_pretrain.pipeline import DistributedContext, _prepare_distributed_loaders
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    loader = DataLoader([0, 1, 2, 3], batch_size=2, shuffle=False)
    prepared = _prepare_distributed_loaders(
        {"paired": loader},
        DistributedContext(enabled=True, rank=0, world_size=2, local_rank=0),
    )

    sampler = prepared["paired"].sampler
    assert isinstance(sampler, DistributedSampler)
    assert sampler.shuffle is False


def test_next_or_restart_cycles_exhausted_loader():
    from humu_pretrain import pipeline
    from torch.utils.data import DataLoader

    next_or_restart = getattr(pipeline, "_next_or_restart", None)
    assert callable(next_or_restart)

    loader = DataLoader([0, 1], batch_size=1, shuffle=False)
    iterator = iter(loader)

    batch, iterator = next_or_restart(loader, iterator)
    values = [int(batch.item())]
    batch, iterator = next_or_restart(loader, iterator)
    values.append(int(batch.item()))
    batch, iterator = next_or_restart(loader, iterator)
    values.append(int(batch.item()))

    assert values == [0, 1, 0]


def test_paired_dataset_builds_real_mol_pocket_and_mol_route_contract(tmp_path):
    from humu_pretrain.data_loader import PairedHUMUDataset

    pocket_dir = tmp_path / "pocket"
    route_dir = tmp_path / "route"
    pocket_dir.mkdir()
    route_dir.mkdir()

    (pocket_dir / "index.jsonl").write_text(
        json.dumps(
            {
                "index": 0,
                "pdb_id": "1ABC_A",
                "pocket_path": "pocket_000000.json",
                "ligand_smiles": "CCO",
                "source_receptor_pdb": "target/1abc.pdb",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (pocket_dir / "pocket_000000.json").write_text(
        json.dumps(
            {
                "pdb_id": "1ABC_A",
                "pocket_atoms": [
                    {"x": 0.0, "y": 0.0, "z": 0.0, "element": "C", "residue": "ALA"},
                    {"x": 1.0, "y": 0.0, "z": 0.0, "element": "N", "residue": "LYS"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (route_dir / "routes.jsonl").write_text(
        json.dumps(
            {
                "id": "USPTO-MIT-train-000001",
                "root_smiles": "CCN",
                "reaction_smiles": "CCO>>CCN",
                "reactions": ["CCO>>CCN"],
                "source_split": "train",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = PairedHUMUDataset(str(pocket_dir), str(route_dir))

    assert len(dataset) == 2
    mol_pocket = dataset[0]
    mol_route = dataset[1]
    assert mol_pocket["pair_type"] == "mol_pocket"
    assert mol_pocket["mol_id"] == "pocket_ligand:1ABC_A"
    assert mol_pocket["pocket_id"] == "1ABC_A"
    assert mol_pocket["route_id"] is None
    assert mol_pocket["ligand_smiles"] == "CCO"
    assert mol_pocket["target_id"] == "target/1abc.pdb"
    assert mol_pocket["source_dataset"] == "crossdocked"
    assert mol_pocket["split"] == "train"
    assert mol_pocket["pocket"]["coords"] == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    assert mol_route["pair_type"] == "mol_route"
    assert mol_route["mol_id"] == "route_product:USPTO-MIT-train-000001"
    assert mol_route["pocket_id"] is None
    assert mol_route["route_id"] == "USPTO-MIT-train-000001"
    assert mol_route["ligand_smiles"] == "CCN"
    assert mol_route["source_dataset"] == "uspto_mit"
    assert mol_route["split"] == "train"


def test_paired_dataset_builds_joint_pocket_route_contract(tmp_path):
    from humu_pretrain.data_loader import PairedHUMUDataset

    pocket_dir = tmp_path / "pocket"
    route_dir = tmp_path / "route"
    joint_dir = tmp_path / "joint"
    for directory in (pocket_dir, route_dir, joint_dir):
        directory.mkdir()

    (pocket_dir / "index.jsonl").write_text("", encoding="utf-8")
    (route_dir / "routes.jsonl").write_text("", encoding="utf-8")
    (joint_dir / "joint.jsonl").write_text(
        json.dumps(
            {
                "id": "joint-1",
                "pdb_id": "1ABC_A",
                "pocket_path": "pocket_joint.json",
                "ligand_smiles": "CCO",
                "route_id": "route-1",
                "reactions": ["CCBr>>CCO"],
                "target_id": "KRAS_G12C",
                "source_dataset": "joint_fixture",
                "split": "train",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (joint_dir / "pocket_joint.json").write_text(
        json.dumps(
            {
                "pdb_id": "1ABC_A",
                "pocket_atoms": [
                    {"x": 0.0, "y": 0.0, "z": 0.0, "element": "C", "residue": "ALA"}
                ],
            }
        ),
        encoding="utf-8",
    )
    dataset = PairedHUMUDataset(
        str(pocket_dir),
        str(route_dir),
        joint_dir=str(joint_dir),
        require_pocket_route=True,
    )

    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["pair_type"] == "mol_pocket_route"
    assert sample["mol_id"] == "joint:joint-1"
    assert sample["pocket_id"] == "1ABC_A"
    assert sample["route_id"] == "route-1"
    assert sample["target_id"] == "KRAS_G12C"
    assert sample["pocket"]["coords"] == [[0.0, 0.0, 0.0]]
    assert sample["route"]["reactions"] == ["CCBr>>CCO"]


def test_paired_dataset_builds_protac_component_contract(tmp_path):
    from humu_pretrain.data_loader import PairedHUMUDataset

    pocket_dir = tmp_path / "pocket"
    route_dir = tmp_path / "route"
    protac_dir = tmp_path / "protacpedia"
    for directory in (pocket_dir, route_dir, protac_dir):
        directory.mkdir()

    (pocket_dir / "index.jsonl").write_text("", encoding="utf-8")
    (route_dir / "routes.jsonl").write_text("", encoding="utf-8")
    (protac_dir / "protacpedia.jsonl").write_text(
        json.dumps(
            {
                "protacdb_id": "p1",
                "protac_canonical_smiles": "CCCOCCN",
                "e3_binder_canonical_smiles": "CCO",
                "ligand_canonical_smiles": "CCN",
                "linker_canonical_smiles": "COC",
                "source": "PROTACpedia",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = PairedHUMUDataset(
        str(pocket_dir),
        str(route_dir),
        protac_dirs=[str(protac_dir)],
    )

    assert len(dataset) == 3
    assert [dataset[index]["pair_type"] for index in range(len(dataset))] == [
        "protac_component",
        "protac_component",
        "protac_component",
    ]
    sample = dataset[0]
    assert sample["mol_id"] == "protac_component:p1:e3_binder"
    assert sample["ligand_smiles"] == "CCCOCCN"
    assert sample["component_smiles"] == "CCO"
    assert sample["component_type"] == "e3_binder"
    assert sample["source_dataset"] == "PROTACpedia"
    assert sample["pocket"] is None
    assert sample["route"] is None


def test_paired_dataset_skips_invalid_optional_protac_component(tmp_path):
    from humu_pretrain.data_loader import PairedHUMUDataset

    pocket_dir = tmp_path / "pocket"
    route_dir = tmp_path / "route"
    protac_dir = tmp_path / "protacpedia"
    for directory in (pocket_dir, route_dir, protac_dir):
        directory.mkdir()

    (pocket_dir / "index.jsonl").write_text("", encoding="utf-8")
    (route_dir / "routes.jsonl").write_text("", encoding="utf-8")
    (protac_dir / "protacpedia.jsonl").write_text(
        json.dumps(
            {
                "protacdb_id": "260",
                "protac_canonical_smiles": "CCCOCCN",
                "e3_binder_canonical_smiles": "CCO",
                "ligand_canonical_smiles": "CCN",
                "linker_smiles": "None",
                "linker_smiles_valid": False,
                "source": "PROTACpedia",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = PairedHUMUDataset(
        str(pocket_dir),
        str(route_dir),
        protac_dirs=[str(protac_dir)],
    )

    assert [dataset[index]["component_type"] for index in range(len(dataset))] == [
        "e3_binder",
        "target_ligand",
    ]


def test_preflight_reports_joint_contract(tmp_path):
    from humu_pretrain.data_loader import preflight_humu_data_contract

    pocket_dir = tmp_path / "pocket"
    route_dir = tmp_path / "route"
    joint_dir = tmp_path / "joint"
    for directory in (pocket_dir, route_dir, joint_dir):
        directory.mkdir()

    (joint_dir / "joint.jsonl").write_text(
        json.dumps(
            {
                "id": "joint-1",
                "pdb_id": "1ABC_A",
                "pocket_path": "pocket_joint.json",
                "ligand_smiles": "CCO",
                "route_id": "route-1",
                "reactions": ["CCBr>>CCO"],
                "target_id": "KRAS_G12C",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (joint_dir / "pocket_joint.json").write_text(
        json.dumps(
            {
                "pdb_id": "1ABC_A",
                "pocket_atoms": [
                    {"x": 0.0, "y": 0.0, "z": 0.0, "element": "C", "residue": "ALA"}
                ],
            }
        ),
        encoding="utf-8",
    )
    report = preflight_humu_data_contract(
        {
            "loss_weights": {"pocket_route": 1.0},
            "data": {
                "pocket_source": str(pocket_dir),
                "route_source": str(route_dir),
                "joint_source": str(joint_dir),
            },
        }
    )

    assert report["required"]["joint_source"] is True
    assert report["sources"]["joint_source"]["records"] == 1


def test_preflight_uses_pocket_manifest_for_esm2_record_count(tmp_path):
    from humu_pretrain.data_loader import preflight_humu_data_contract

    pocket_dir = tmp_path / "pocket"
    route_dir = tmp_path / "route"
    pocket_dir.mkdir()
    route_dir.mkdir()
    (pocket_dir / "manifest.json").write_text(
        json.dumps(
            {
                "format": "jsonl_index_with_json_sidecars",
                "n_records": 43421,
                "esm2_input": "protein_sequence extracted from receptor PDB ATOM records",
            }
        ),
        encoding="utf-8",
    )
    (pocket_dir / "index.jsonl").write_text(
        json.dumps(
            {
                "index": 0,
                "pdb_id": "1ABC_A",
                "pocket_path": "pocket_000000.json",
                "ligand_smiles": "CCO",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (pocket_dir / "pocket_000000.json").write_text(
        json.dumps(
            {
                "pocket_atoms": [
                    {"x": 0.0, "y": 0.0, "z": 0.0, "element": "C", "residue": "ALA"}
                ],
                "protein_sequence": "AAAA",
            }
        ),
        encoding="utf-8",
    )

    report = preflight_humu_data_contract(
        {
            "data": {
                "pocket_source": str(pocket_dir),
                "route_source": str(route_dir),
            },
            "encoders": {
                "pocket": {
                    "use_esm2": True,
                    "esm2_dim": 1280,
                }
            },
        }
    )

    assert report["sources"]["pocket_source"]["esm2_records"] == 43421


def test_preflight_rejects_enabled_joint_loss_without_joint_records(tmp_path):
    from humu_pretrain.data_loader import preflight_humu_data_contract

    pocket_dir = tmp_path / "pocket"
    route_dir = tmp_path / "route"
    joint_dir = tmp_path / "joint"
    for directory in (pocket_dir, route_dir, joint_dir):
        directory.mkdir()

    with pytest.raises(ValueError, match="data.joint_source contains no joint records"):
        preflight_humu_data_contract(
            {
                "loss_weights": {"pocket_route": 1.0},
                "data": {
                    "pocket_source": str(pocket_dir),
                    "route_source": str(route_dir),
                    "joint_source": str(joint_dir),
                },
            }
        )


def test_preflight_reports_activity_source_contract(tmp_path):
    from humu_pretrain.data_loader import preflight_humu_data_contract

    pocket_dir = tmp_path / "pocket"
    route_dir = tmp_path / "route"
    activity_dir = tmp_path / "activity"
    for directory in (pocket_dir, route_dir, activity_dir):
        directory.mkdir()
    (activity_dir / "activity.jsonl").write_text(
        json.dumps(
            {
                "ligand_smiles": "CCO",
                "target_id": "CHEMBL_TARGET_1",
                "activity_value": 7.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = preflight_humu_data_contract(
        {
            "data": {
                "pocket_source": str(pocket_dir),
                "route_source": str(route_dir),
                "activity_source": str(activity_dir),
            },
        }
    )

    assert report["sources"]["activity_source"]["records"] == 1


def test_preflight_reports_activity_sources_contract(tmp_path):
    from humu_pretrain.data_loader import preflight_humu_data_contract

    pocket_dir = tmp_path / "pocket"
    route_dir = tmp_path / "route"
    chembl_dir = tmp_path / "activity"
    bindingdb_dir = tmp_path / "bindingdb_activity"
    for directory in (pocket_dir, route_dir, chembl_dir, bindingdb_dir):
        directory.mkdir()
    (chembl_dir / "activity.jsonl").write_text(
        json.dumps(
            {
                "ligand_smiles": "CCO",
                "target_id": "CHEMBL_TARGET_1",
                "activity_value": 7.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (bindingdb_dir / "activity.jsonl").write_text(
        json.dumps(
            {
                "ligand_smiles": "CCN",
                "target_id": "P03367",
                "activity_value": 8.1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = preflight_humu_data_contract(
        {
            "data": {
                "pocket_source": str(pocket_dir),
                "route_source": str(route_dir),
                "activity_source": str(chembl_dir),
                "activity_sources": [str(bindingdb_dir)],
            },
        }
    )

    assert report["sources"]["activity_sources"]["records"] == 2
    assert [source["records"] for source in report["sources"]["activity_sources"]["sources"]] == [
        1,
        1,
    ]


def test_preflight_uses_activity_manifest_record_count(tmp_path):
    from humu_pretrain.data_loader import preflight_humu_data_contract

    pocket_dir = tmp_path / "pocket"
    route_dir = tmp_path / "route"
    activity_dir = tmp_path / "bindingdb_activity"
    for directory in (pocket_dir, route_dir, activity_dir):
        directory.mkdir()
    (activity_dir / "manifest.json").write_text(
        json.dumps(
            {
                "format": "humu_activity_jsonl_candidate",
                "n_records": 3145942,
                "output": str(activity_dir / "activity.jsonl"),
            }
        ),
        encoding="utf-8",
    )
    (activity_dir / "activity.jsonl").write_text(
        json.dumps(
            {
                "ligand_smiles": "CCO",
                "target_id": "P03367",
                "activity_value": 8.1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = preflight_humu_data_contract(
        {
            "data": {
                "pocket_source": str(pocket_dir),
                "route_source": str(route_dir),
                "activity_sources": [str(activity_dir)],
            },
        }
    )

    assert report["sources"]["activity_sources"]["records"] == 3145942
    assert report["sources"]["activity_sources"]["sources"][0]["records"] == 3145942


def test_activity_pair_iterator_uses_processed_contract_without_rdkit_validation(
    monkeypatch,
    tmp_path,
):
    import humu_pretrain.data_loader as data_loader_module

    activity_dir = tmp_path / "activity"
    activity_dir.mkdir()
    (activity_dir / "activity.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "ligand_smiles": "CCO",
                        "target_id": "CHEMBL_TARGET_1",
                        "activity_value": 7.0,
                    }
                ),
                json.dumps(
                    {
                        "ligand_smiles": "CCN",
                        "target_id": "CHEMBL_TARGET_1",
                        "activity_value": 8.0,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def fail_if_called(_smiles):
        raise AssertionError("activity iterator must not RDKit-validate every record")

    monkeypatch.setattr(data_loader_module, "_is_valid_smiles", fail_if_called)

    records = list(data_loader_module._iter_activity_pair_records(str(activity_dir)))

    assert len(records) == 1
    assert records[0]["ligand_smiles"] == "CCO"
    assert records[0]["positive_smiles"] == "CCN"


def test_preflight_reports_protac_source_contract(tmp_path):
    from humu_pretrain.data_loader import preflight_humu_data_contract

    pocket_dir = tmp_path / "pocket"
    route_dir = tmp_path / "route"
    protac_dir = tmp_path / "protacpedia"
    for directory in (pocket_dir, route_dir, protac_dir):
        directory.mkdir()
    (protac_dir / "protacpedia.jsonl").write_text(
        json.dumps(
            {
                "protac_canonical_smiles": "CCCOCCN",
                "e3_binder_canonical_smiles": "CCO",
                "ligand_canonical_smiles": "CCN",
                "linker_canonical_smiles": "COC",
                "protacdb_id": "p1",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = preflight_humu_data_contract(
        {
            "loss_weights": {"protac_component": 0.2},
            "data": {
                "pocket_source": str(pocket_dir),
                "route_source": str(route_dir),
                "protac_sources": [str(protac_dir)],
            },
        }
    )

    assert report["sources"]["protac_sources"]["records"] == 3
    assert report["sources"]["protac_sources"]["sources"][0]["records"] == 3


def test_paired_dataset_preserves_records_and_collate_rejects_invalid_smiles(tmp_path):
    from humu_pretrain.data_loader import PairedHUMUDataset, _record_collate

    pocket_dir = tmp_path / "pocket"
    route_dir = tmp_path / "route"
    pocket_dir.mkdir()
    route_dir.mkdir()

    (pocket_dir / "index.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "index": 0,
                        "pdb_id": "valid-pocket",
                        "pocket_path": "pocket_000000.json",
                        "ligand_smiles": "CCO",
                    }
                ),
                json.dumps(
                    {
                        "index": 1,
                        "pdb_id": "invalid-pocket",
                        "pocket_path": "pocket_000001.json",
                        "ligand_smiles": "C1CC",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    for idx in range(2):
        (pocket_dir / f"pocket_{idx:06d}.json").write_text(
            json.dumps(
                {
                    "pocket_atoms": [
                        {"x": 0.0, "y": 0.0, "z": 0.0, "element": "C", "residue": "ALA"}
                    ]
                }
            ),
            encoding="utf-8",
        )
    (route_dir / "routes.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "valid-route",
                        "root_smiles": "CCN",
                        "reaction_smiles": "CCO>>CCN",
                    }
                ),
                json.dumps(
                    {
                        "id": "invalid-route",
                        "root_smiles": "C1CC",
                        "reaction_smiles": "CCO>>C1CC",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = PairedHUMUDataset(str(pocket_dir), str(route_dir))
    assert [dataset.samples[idx]["ligand_smiles"] for idx in range(len(dataset))] == [
        "CCO",
        "C1CC",
        "CCN",
        "C1CC",
    ]

    with pytest.raises(ValueError, match="invalid ligand_smiles"):
        _record_collate([dataset[idx] for idx in range(len(dataset))])

    batch = _record_collate([dataset[0], dataset[2]])

    assert batch["ligand_smiles"] == ["CCO", "CCN"]
    assert batch["pair_type"] == ["mol_pocket", "mol_route"]


def test_record_collate_rejects_invalid_smiles_without_dropping_records():
    from humu_pretrain.data_loader import _record_collate

    with pytest.raises(ValueError, match="invalid ligand_smiles"):
        _record_collate(
            [
                {"mol_id": "valid", "ligand_smiles": "CCO", "pair_type": "mol_route"},
                {"mol_id": "invalid", "ligand_smiles": "C1CC", "pair_type": "mol_route"},
            ]
        )


def test_record_collate_accepts_valence_outlier_smiles():
    from humu_pretrain.data_loader import _record_collate

    batch = _record_collate(
        [{"mol_id": "boron", "ligand_smiles": "B(F)(F)(F)F", "pair_type": "mol_route"}]
    )

    assert batch["ligand_smiles"] == ["B(F)(F)(F)F"]


def test_record_collate_rejects_invalid_component_smiles():
    from humu_pretrain.data_loader import _record_collate

    with pytest.raises(ValueError, match="invalid component_smiles"):
        _record_collate(
            [
                {
                    "mol_id": "protac-component",
                    "ligand_smiles": "CCO",
                    "component_smiles": "C1CC",
                    "pair_type": "protac_component",
                }
            ]
        )


def test_forward_route_only_batch_keeps_pocket_encoder_ddp_path_with_esm2(monkeypatch):
    from humu_pretrain.pipeline import _forward_paired_batch, _wrap_as_module
    from mf_encoders.humu_pocket.encoder import HUMUPocketEncoder

    device = torch.device("cpu")
    pocket = HUMUPocketEncoder(dim=8, use_esm2=True, esm2_checkpoint="unused")

    def fake_esm2_batch_embeddings(self, sequences):
        assert sequences == ["A"]
        return torch.zeros(len(sequences), self.esm2_dim, device=self._param_device())

    monkeypatch.setattr(
        HUMUPocketEncoder,
        "_compute_esm2_batch_embeddings",
        fake_esm2_batch_embeddings,
    )

    class ConstantEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(9, 9)

        def forward(self, items):
            size = len(items) if isinstance(items, list) else 1
            spatial = torch.zeros(size, 8)
            time_coord = torch.ones(size, 1)
            return self.proj(torch.cat([time_coord, spatial], dim=-1))

    encoders = {
        "mol": ConstantEncoder(),
        "pocket": _wrap_as_module(pocket, 8, device),
        "route": ConstantEncoder(),
    }
    batch = {
        "ligand_smiles": ["CCO"],
        "pair_type": ["mol_route"],
        "pocket": [None],
        "route": [{"id": "r1", "reactions": ["CCO>>CCN"]}],
    }

    losses = _forward_paired_batch(encoders, batch, {"loss_weights": {"mol_route": 1.0}})

    assert torch.isfinite(losses["total"])


def test_forward_paired_batch_computes_protac_component_loss():
    from humu_pretrain.pipeline import _forward_paired_batch

    class SmilesEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, items):
            if not isinstance(items, list):
                items = [items]
            spatial = torch.zeros(len(items), 128) * self.weight
            for index, item in enumerate(items):
                smiles = item if isinstance(item, str) else item.get("smiles", "")
                spatial[index, index % 4] = float(len(smiles)) / 100.0
            time_coord = torch.sqrt(1 + spatial.pow(2).sum(dim=-1, keepdim=True))
            return torch.cat([time_coord, spatial], dim=-1)

    class EmptyTower(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, items):
            if not isinstance(items, list):
                items = [items]
            spatial = torch.zeros(len(items), 128) * self.weight
            time_coord = torch.ones(len(items), 1)
            return torch.cat([time_coord, spatial], dim=-1)

    encoders = {
        "mol": SmilesEncoder(),
        "pocket": EmptyTower(),
        "route": EmptyTower(),
    }
    batch = {
        "ligand_smiles": ["CCCOCCN", "CCCCCCN"],
        "component_smiles": ["CCO", "CCN"],
        "pair_type": ["protac_component", "protac_component"],
        "pocket": [None, None],
        "route": [None, None],
    }

    losses = _forward_paired_batch(
        encoders,
        batch,
        {
            "loss_weights": {
                "mol_pocket": 0.0,
                "mol_route": 0.0,
                "protac_component": 1.0,
            }
        },
    )

    assert "l_protac_component" in losses
    assert torch.isfinite(losses["l_protac_component"])
    assert losses["total"] == losses["l_protac_component"]


def test_forward_paired_batch_computes_protac_component_library_loss():
    from humu_pretrain.pipeline import _forward_paired_batch

    class SmilesEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, items):
            if not isinstance(items, list):
                items = [items]
            spatial = torch.zeros(len(items), 128) * self.weight
            for index, item in enumerate(items):
                smiles = item if isinstance(item, str) else item.get("smiles", "")
                spatial[index, index % 4] = float(len(smiles)) / 100.0
            time_coord = torch.sqrt(1 + spatial.pow(2).sum(dim=-1, keepdim=True))
            return torch.cat([time_coord, spatial], dim=-1)

    class EmptyTower(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, items):
            if not isinstance(items, list):
                items = [items]
            spatial = torch.zeros(len(items), 128) * self.weight
            time_coord = torch.ones(len(items), 1)
            return torch.cat([time_coord, spatial], dim=-1)

    encoders = {
        "mol": SmilesEncoder(),
        "pocket": EmptyTower(),
        "route": EmptyTower(),
    }
    batch = {
        "ligand_smiles": [None, None],
        "component_smiles": ["NC1=CC=CC=C1", "O=C(NCC)C1=CC=CC=C1"],
        "pair_type": ["protac_component_library", "protac_component_library"],
        "pocket": [None, None],
        "route": [None, None],
    }

    losses = _forward_paired_batch(
        encoders,
        batch,
        {
            "loss_weights": {
                "mol_pocket": 0.0,
                "mol_route": 0.0,
                "protac_component_library": 1.0,
            }
        },
    )

    assert "l_protac_component_library" in losses
    assert torch.isfinite(losses["l_protac_component_library"])
    assert losses["total"] == losses["l_protac_component_library"]


def test_paired_dataset_loads_pocket_coordinates_lazily(tmp_path):
    from humu_pretrain.data_loader import PairedHUMUDataset

    pocket_dir = tmp_path / "pocket"
    route_dir = tmp_path / "route"
    pocket_dir.mkdir()
    route_dir.mkdir()

    (pocket_dir / "index.jsonl").write_text(
        json.dumps(
            {
                "index": 0,
                "pdb_id": "lazy-pocket",
                "pocket_path": "missing_pocket.json",
                "ligand_smiles": "CCO",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (route_dir / "routes.jsonl").write_text(
        json.dumps(
            {
                "id": "route-1",
                "root_smiles": "CCN",
                "reaction_smiles": "CCO>>CCN",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = PairedHUMUDataset(str(pocket_dir), str(route_dir))

    assert len(dataset) == 2
    with pytest.raises(FileNotFoundError, match="missing_pocket.json"):
        dataset[0]


def test_create_dataloaders_filters_overlong_esm2_pocket_sequences(tmp_path):
    from humu_pretrain.data_loader import create_dataloaders

    pocket_dir = tmp_path / "pocket"
    route_dir = tmp_path / "route"
    pocket_dir.mkdir()
    route_dir.mkdir()

    (pocket_dir / "index.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "index": 0,
                        "pdb_id": "long-pocket",
                        "pocket_path": "pocket_000000.json",
                        "ligand_smiles": "CCO",
                    }
                ),
                json.dumps(
                    {
                        "index": 1,
                        "pdb_id": "kept-pocket",
                        "pocket_path": "pocket_000001.json",
                        "ligand_smiles": "CCN",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    for path, sequence in (
        ("pocket_000000.json", "AAAA"),
        ("pocket_000001.json", "AAA"),
    ):
        (pocket_dir / path).write_text(
            json.dumps(
                {
                    "pocket_atoms": [
                        {"x": 0.0, "y": 0.0, "z": 0.0, "element": "C", "residue": "ALA"}
                    ],
                    "protein_sequence": sequence,
                }
            ),
            encoding="utf-8",
        )
    (route_dir / "routes.jsonl").write_text(
        json.dumps(
            {
                "id": "route-1",
                "root_smiles": "CCC",
                "reaction_smiles": "CCO>>CCC",
                "source_split": "train",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    loaders = create_dataloaders(
        {
            "batch_size": 8,
            "data": {
                "pocket_source": str(pocket_dir),
                "route_source": str(route_dir),
                "shuffle": False,
                "num_workers": 0,
            },
            "encoders": {
                "pocket": {
                    "use_esm2": True,
                    "esm2_max_sequence_length": 3,
                }
            },
        }
    )

    dataset = loaders["paired"].dataset
    assert dataset.filtered_counts["pocket_esm2_sequence"] == 1
    pocket_ids = [
        sample["pocket_id"]
        for sample in dataset.samples
        if sample["pair_type"] == "mol_pocket"
    ]
    assert pocket_ids == ["kept-pocket"]


def test_create_dataloaders_limits_pocket_point_clouds(tmp_path):
    from humu_pretrain.data_loader import create_dataloaders

    pocket_dir = tmp_path / "pocket"
    route_dir = tmp_path / "route"
    joint_dir = tmp_path / "joint"
    for directory in (pocket_dir, route_dir, joint_dir):
        directory.mkdir()

    def atom(index: int) -> dict:
        return {
            "x": float(index),
            "y": 0.0,
            "z": 0.0,
            "element": f"E{index}",
            "residue": f"R{index}",
        }

    (pocket_dir / "index.jsonl").write_text(
        json.dumps(
            {
                "index": 0,
                "pdb_id": "large-pocket",
                "pocket_path": "pocket_000000.json",
                "ligand_smiles": "CCO",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (pocket_dir / "pocket_000000.json").write_text(
        json.dumps({"pocket_atoms": [atom(index) for index in range(5)]}),
        encoding="utf-8",
    )
    (route_dir / "routes.jsonl").write_text(
        json.dumps(
            {
                "id": "route-1",
                "root_smiles": "CCC",
                "reaction_smiles": "CCO>>CCC",
                "source_split": "train",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (joint_dir / "joint.jsonl").write_text(
        json.dumps(
            {
                "id": "joint-1",
                "pdb_id": "joint-pocket",
                "pocket_path": "pocket_joint.json",
                "ligand_smiles": "CCN",
                "route_id": "route-1",
                "reactions": ["CCO>>CCN"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (joint_dir / "pocket_joint.json").write_text(
        json.dumps({"pocket_atoms": [atom(index) for index in range(5, 10)]}),
        encoding="utf-8",
    )

    loaders = create_dataloaders(
        {
            "batch_size": 8,
            "data": {
                "pocket_source": str(pocket_dir),
                "route_source": str(route_dir),
                "joint_source": str(joint_dir),
                "max_pocket_points": 3,
                "shuffle": False,
                "num_workers": 0,
            },
        }
    )

    dataset = loaders["paired"].dataset
    pocket_sample = next(
        dataset[index]
        for index, sample in enumerate(dataset.samples)
        if sample["pair_type"] == "mol_pocket"
    )
    joint_sample = next(
        dataset[index]
        for index, sample in enumerate(dataset.samples)
        if sample["pair_type"] == "mol_pocket_route"
    )

    assert pocket_sample["pocket"]["coords"] == [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
    ]
    assert pocket_sample["pocket"]["elements"] == ["E0", "E2", "E4"]
    assert joint_sample["pocket"]["coords"] == [
        [5.0, 0.0, 0.0],
        [7.0, 0.0, 0.0],
        [9.0, 0.0, 0.0],
    ]
    assert joint_sample["pocket"]["residue_types"] == ["R5", "R7", "R9"]


def test_create_dataloaders_uses_single_paired_loader(tmp_path):
    from humu_pretrain.data_loader import create_dataloaders

    pocket_dir = tmp_path / "pocket"
    route_dir = tmp_path / "route"
    mol_dir = tmp_path / "mol"
    pocket_dir.mkdir()
    route_dir.mkdir()
    mol_dir.mkdir()
    (mol_dir / "manifest.json").write_text(json.dumps({"shards": []}), encoding="utf-8")
    (pocket_dir / "index.jsonl").write_text(
        json.dumps(
            {
                "index": 0,
                "pdb_id": "1ABC_A",
                "pocket_path": "pocket_000000.json",
                "ligand_smiles": "CCO",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (pocket_dir / "pocket_000000.json").write_text(
        json.dumps(
            {
                "pocket_atoms": [
                    {"x": 0.0, "y": 0.0, "z": 0.0, "element": "C", "residue": "ALA"}
                ]
            }
        ),
        encoding="utf-8",
    )
    (route_dir / "routes.jsonl").write_text(
        json.dumps(
            {
                "id": "route-1",
                "root_smiles": "CCN",
                "reaction_smiles": "CCO>>CCN",
                "source_split": "train",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    loaders = create_dataloaders(
        {
            "batch_size": 2,
            "data": {
                "mol_source": str(mol_dir),
                "pocket_source": str(pocket_dir),
                "route_source": str(route_dir),
                "num_workers": 0,
                "shuffle": False,
            },
        }
    )
    batch = next(iter(loaders["paired"]))

    assert list(loaders) == ["paired"]
    assert batch["pair_type"] == ["mol_pocket", "mol_route"]
    assert batch["ligand_smiles"] == ["CCO", "CCN"]


def test_create_dataloaders_builds_validation_split_when_eval_enabled(tmp_path):
    from humu_pretrain.data_loader import create_dataloaders

    pocket_dir = tmp_path / "pocket"
    route_dir = tmp_path / "route"
    mol_dir = tmp_path / "mol"
    pocket_dir.mkdir()
    route_dir.mkdir()
    mol_dir.mkdir()
    (mol_dir / "manifest.json").write_text(json.dumps({"shards": []}), encoding="utf-8")
    (pocket_dir / "index.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "index": 0,
                        "pdb_id": "1ABC_A",
                        "pocket_path": "pocket_000000.json",
                        "ligand_smiles": "CCO",
                    }
                ),
                json.dumps(
                    {
                        "index": 1,
                        "pdb_id": "2ABC_A",
                        "pocket_path": "pocket_000001.json",
                        "ligand_smiles": "CCN",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    for index in range(2):
        (pocket_dir / f"pocket_{index:06d}.json").write_text(
            json.dumps(
                {
                    "pocket_atoms": [
                        {"x": 0.0, "y": 0.0, "z": 0.0, "element": "C", "residue": "ALA"}
                    ]
                }
            ),
            encoding="utf-8",
        )
    (route_dir / "routes.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "route-1",
                        "root_smiles": "CCC",
                        "reaction_smiles": "CCO>>CCC",
                        "source_split": "train",
                    }
                ),
                json.dumps(
                    {
                        "id": "route-2",
                        "root_smiles": "CCCl",
                        "reaction_smiles": "CCO>>CCCl",
                        "source_split": "train",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    loaders = create_dataloaders(
        {
            "batch_size": 2,
            "seed": 7,
            "data": {
                "mol_source": str(mol_dir),
                "pocket_source": str(pocket_dir),
                "route_source": str(route_dir),
                "num_workers": 0,
                "shuffle": False,
            },
            "eval": {
                "every_n_epochs": 1,
                "eval_split_ratio": 0.25,
            },
        }
    )

    assert set(loaders) == {"paired", "validation"}
    assert len(loaders["paired"].dataset) == 3
    assert len(loaders["validation"].dataset) == 1


def test_create_dataloaders_rejects_legacy_joint_oversample_factor(tmp_path):
    from humu_pretrain.data_loader import create_dataloaders

    pocket_dir = tmp_path / "pocket"
    route_dir = tmp_path / "route"
    joint_dir = tmp_path / "joint"
    for directory in (pocket_dir, route_dir, joint_dir):
        directory.mkdir()

    (pocket_dir / "index.jsonl").write_text(
        json.dumps(
            {
                "index": 0,
                "pdb_id": "1ABC_A",
                "pocket_path": "pocket_000000.json",
                "ligand_smiles": "CCO",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (pocket_dir / "pocket_000000.json").write_text(
        json.dumps(
            {
                "pocket_atoms": [
                    {"x": 0.0, "y": 0.0, "z": 0.0, "element": "C", "residue": "ALA"}
                ]
            }
        ),
        encoding="utf-8",
    )
    (route_dir / "routes.jsonl").write_text(
        json.dumps(
            {
                "id": "route-1",
                "root_smiles": "CCN",
                "reaction_smiles": "CCO>>CCN",
                "source_split": "train",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (joint_dir / "joint.jsonl").write_text(
        json.dumps(
            {
                "id": "joint-1",
                "pdb_id": "1ABC_A",
                "pocket_path": "pocket_joint.json",
                "ligand_smiles": "CCO",
                "route_id": "route-1",
                "reactions": ["CCN>>CCO"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (joint_dir / "pocket_joint.json").write_text(
        json.dumps(
            {
                "pocket_atoms": [
                    {"x": 0.0, "y": 0.0, "z": 0.0, "element": "C", "residue": "ALA"}
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="data.joint_oversample_factor"):
        create_dataloaders(
            {
                "batch_size": 16,
                "data": {
                    "pocket_source": str(pocket_dir),
                    "route_source": str(route_dir),
                    "joint_source": str(joint_dir),
                    "joint_oversample_factor": 4,
                    "num_workers": 0,
                    "shuffle": False,
                },
                "loss_weights": {"pocket_route": 1.0},
            }
        )


def test_target_ratio_sampler_matches_configured_objective_mix(tmp_path):
    from collections import Counter

    from humu_pretrain.data_loader import create_dataloaders

    sources = _write_minimal_humu_sources(tmp_path)

    loaders = create_dataloaders(
        {
            "batch_size": 10,
            "max_samples": 8,
            "data": {
                "mol_source": str(sources["mol"]),
                "pocket_source": str(sources["pocket"]),
                "route_source": str(sources["route"]),
                "joint_source": str(sources["joint"]),
                "activity_source": str(sources["activity"]),
                "protac_sources": [str(sources["protacpedia"]), str(sources["protacdb"])],
                "route_eval_source": str(sources["route_eval"]),
                "retropath_template_source": str(sources["retropath_templates"]),
                "num_workers": 0,
                "shuffle": False,
                "objective_sampling": {
                    "enabled": True,
                    "steps_per_epoch": 2,
                    "alpha": 0.5,
                    "objectives": {
                        "mol_self": 0.2,
                        "mol_pocket": 0.1,
                        "mol_route": 0.1,
                        "mol_pocket_route": 0.2,
                        "activity_pair": 0.1,
                        "protac_component": 0.2,
                        "route_template": 0.1,
                    },
                },
            },
            "loss_weights": {
                "pocket_route": 1.0,
                "protac_component": 1.0,
            },
        }
    )

    counts = Counter()
    for batch in loaders["paired"]:
        counts.update(batch["pair_type"])

    assert len(loaders["paired"]) == 2
    assert counts == {
        "mol_self": 4,
        "mol_pocket": 2,
        "mol_route": 2,
        "mol_pocket_route": 4,
        "activity_pair": 2,
        "protac_component": 4,
        "route_template": 2,
    }


def test_training_batch_reports_source_coverage_stats(tmp_path):
    from humu_pretrain.data_loader import create_dataloaders

    sources = _write_minimal_humu_sources(tmp_path)

    loaders = create_dataloaders(
        {
            "batch_size": 7,
            "max_samples": 8,
            "data": {
                "mol_source": str(sources["mol"]),
                "pocket_source": str(sources["pocket"]),
                "route_source": str(sources["route"]),
                "joint_source": str(sources["joint"]),
                "activity_source": str(sources["activity"]),
                "protac_sources": [str(sources["protacpedia"]), str(sources["protacdb"])],
                "route_eval_source": str(sources["route_eval"]),
                "retropath_template_source": str(sources["retropath_templates"]),
                "num_workers": 0,
                "objective_sampling": {
                    "enabled": True,
                    "steps_per_epoch": 1,
                    "objectives": {
                        "mol_self": 1,
                        "mol_pocket": 1,
                        "mol_route": 1,
                        "mol_pocket_route": 1,
                        "activity_pair": 1,
                        "protac_component": 1,
                        "route_template": 1,
                    },
                },
            },
            "loss_weights": {
                "pocket_route": 1.0,
                "protac_component": 1.0,
            },
        }
    )

    batch = next(iter(loaders["paired"]))

    assert batch["pair_type_counts"] == {
        "mol_self": 1,
        "mol_pocket": 1,
        "mol_route": 1,
        "mol_pocket_route": 1,
        "activity_pair": 1,
        "protac_component": 1,
        "route_template": 1,
    }
    assert batch["source_counts"]["mol"] == 1
    assert batch["source_counts"]["pocket"] == 1
    assert batch["source_counts"]["route"] == 1
    assert batch["source_counts"]["joint"] == 1
    assert batch["source_counts"]["activity"] == 1
    assert batch["source_counts"]["protacpedia"] + batch["source_counts"]["protacdb"] == 1
    assert batch["source_counts"]["retropath_templates"] == 1
    assert batch["unique_source_coverage"] == pytest.approx(7 / 7)
    assert batch["source_repeat_rate"] == pytest.approx(0.0)


def test_source_registry_requires_all_humu_datasets(tmp_path):
    from humu_pretrain.data_loader import preflight_humu_data_contract

    sources = _write_minimal_humu_sources(tmp_path)

    report = preflight_humu_data_contract(
        {
            "loss_weights": {"pocket_route": 1.0, "protac_component": 1.0},
            "data": {
                "require_all_humu_sources": True,
                "mol_source": str(sources["mol"]),
                "pocket_source": str(sources["pocket"]),
                "route_source": str(sources["route"]),
                "joint_source": str(sources["joint"]),
                "activity_source": str(sources["activity"]),
                "protac_sources": [str(sources["protacpedia"])],
                "protacdb_source": str(sources["protacdb"]),
                "protac8k_source": str(sources["protac8k"]),
                "rcsb_mmcif_source": str(sources["rcsb_mmcif"]),
                "interface_skempi2_source": str(sources["interface_skempi2"]),
                "pdcdb_source": str(sources["pdcdb"]),
                "route_eval_source": str(sources["route_eval"]),
                "retropath_template_source": str(sources["retropath_templates"]),
            },
        }
    )

    source_registry = report["source_registry"]
    assert set(source_registry) >= {
        "mol",
        "pocket",
        "route",
        "route_eval",
        "joint",
        "activity",
        "protacpedia",
        "protacdb",
        "protac8k",
        "rcsb_mmcif",
        "interface_skempi2",
        "pdcdb",
        "retropath_templates",
    }
    assert all(source["configured"] for source in source_registry.values())
    assert all(source["trainable"] for source in source_registry.values())


def test_default_config_keeps_pdcdb_visible_without_training_objective(tmp_path):
    import yaml
    from humu_pretrain.data_loader import create_dataloaders, preflight_humu_data_contract

    sources = _write_minimal_humu_sources(tmp_path)
    cfg = yaml.safe_load((ROOT / "configs/models/humu_pretrain.yaml").read_text())
    cfg["batch_size"] = 256
    cfg["max_samples"] = 8
    cfg["data"].update(
        {
            "mol_source": str(sources["mol"]),
            "pocket_source": str(sources["pocket"]),
            "route_source": str(sources["route"]),
            "joint_source": str(sources["joint"]),
            "activity_source": str(sources["activity"]),
            "activity_sources": [],
            "protac_sources": [str(sources["protacpedia"])],
            "protacdb_source": str(sources["protacdb"]),
            "protac8k_source": str(sources["protac8k"]),
            "rcsb_mmcif_source": str(sources["rcsb_mmcif"]),
            "interface_skempi2_source": str(sources["interface_skempi2"]),
            "pdcdb_source": str(sources["pdcdb"]),
            "route_eval_source": str(sources["route_eval"]),
            "retropath_template_source": str(sources["retropath_templates"]),
            "num_workers": 0,
        }
    )
    cfg["data"]["objective_sampling"]["steps_per_epoch"] = 1

    report = preflight_humu_data_contract(cfg)
    loader = create_dataloaders(cfg)["paired"]
    batch = next(iter(loader))

    assert report["source_registry"]["pdcdb"]["configured"] is True
    assert report["source_registry"]["pdcdb"]["trainable"] is True
    assert cfg["data"]["objective_sampling"]["objectives"]["pdc_component"] == 0.0
    assert "pdc_component" not in batch["pair_type_counts"]


def test_train_cli_preflight_resolves_local_pipeline_package(tmp_path):
    sources = _write_minimal_humu_sources(tmp_path)
    config_path = tmp_path / "humu_preflight.yaml"
    config_path.write_text(
        "\n".join(
            [
                "loss_weights:",
                "  pocket_route: 1.0",
                "  protac_component: 1.0",
                "  protac_ternary: 1.0",
                "  protein_interface: 1.0",
                "  interface_mutation: 1.0",
                "  pdc_component: 1.0",
                "data:",
                "  require_all_humu_sources: true",
                f"  mol_source: {sources['mol']}",
                f"  pocket_source: {sources['pocket']}",
                f"  route_source: {sources['route']}",
                f"  joint_source: {sources['joint']}",
                f"  activity_source: {sources['activity']}",
                "  protac_sources:",
                f"    - {sources['protacpedia']}",
                f"  protacdb_source: {sources['protacdb']}",
                f"  protac8k_source: {sources['protac8k']}",
                f"  rcsb_mmcif_source: {sources['rcsb_mmcif']}",
                f"  interface_skempi2_source: {sources['interface_skempi2']}",
                f"  pdcdb_source: {sources['pdcdb']}",
                f"  route_eval_source: {sources['route_eval']}",
                f"  retropath_template_source: {sources['retropath_templates']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            str(root / "pipelines" / "humu_pretrain" / "train.py"),
            "--config",
            str(config_path),
            "--preflight-only",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["source_registry"]["mol"]["trainable"] is True


def test_objective_sampling_uses_all_trainable_humu_sources(tmp_path):
    from humu_pretrain.data_loader import create_dataloaders

    sources = _write_minimal_humu_sources(tmp_path)
    loader = create_dataloaders(
        {
            "batch_size": 11,
            "max_samples": 8,
            "loss_weights": {
                "pocket_route": 1.0,
                "protac_component": 1.0,
                "protac_ternary": 1.0,
                "protein_interface": 1.0,
                "interface_mutation": 1.0,
                "pdc_component": 1.0,
            },
            "data": {
                "require_all_humu_sources": True,
                "mol_source": str(sources["mol"]),
                "pocket_source": str(sources["pocket"]),
                "route_source": str(sources["route"]),
                "joint_source": str(sources["joint"]),
                "activity_source": str(sources["activity"]),
                "protac_sources": [str(sources["protacpedia"])],
                "protacdb_source": str(sources["protacdb"]),
                "protac8k_source": str(sources["protac8k"]),
                "rcsb_mmcif_source": str(sources["rcsb_mmcif"]),
                "interface_skempi2_source": str(sources["interface_skempi2"]),
                "pdcdb_source": str(sources["pdcdb"]),
                "route_eval_source": str(sources["route_eval"]),
                "retropath_template_source": str(sources["retropath_templates"]),
                "num_workers": 0,
                "objective_sampling": {
                    "enabled": True,
                    "steps_per_epoch": 1,
                    "objectives": {
                        "mol_self": 1,
                        "mol_pocket": 1,
                        "mol_route": 1,
                        "mol_pocket_route": 1,
                        "activity_pair": 1,
                        "protac_component": 1,
                        "route_template": 1,
                        "protac_ternary": 1,
                        "protein_interface": 1,
                        "interface_mutation": 1,
                        "pdc_component": 1,
                    },
                },
            },
        }
    )["paired"]

    batch = next(iter(loader))

    assert batch["pair_type_counts"] == {
        "mol_self": 1,
        "mol_pocket": 1,
        "mol_route": 1,
        "mol_pocket_route": 1,
        "activity_pair": 1,
        "protac_component": 1,
        "route_template": 1,
        "protac_ternary": 1,
        "protein_interface": 1,
        "interface_mutation": 1,
        "pdc_component": 1,
    }
    for source_name in (
        "protac8k",
        "rcsb_mmcif",
        "interface_skempi2",
        "pdcdb",
    ):
        assert batch["source_counts"][source_name] == 1


def test_default_objective_sampler_includes_protacdb_library_source(tmp_path):
    from humu_pretrain.data_loader import create_dataloaders

    sources = _write_minimal_humu_sources(tmp_path)
    loader = create_dataloaders(
        {
            "batch_size": 2,
            "max_samples": 8,
            "loss_weights": {
                "protac_component_library": 1.0,
            },
            "data": {
                "pocket_source": str(sources["pocket"]),
                "route_source": str(sources["route"]),
                "protac_sources": [str(sources["protacpedia"])],
                "protacdb_source": str(sources["protacdb"]),
                "num_workers": 0,
                "objective_sampling": {
                    "enabled": True,
                    "steps_per_epoch": 1,
                    "objectives": {
                        "protac_component_library": 1,
                    },
                },
            },
        }
    )["paired"]

    batch = next(iter(loader))

    assert batch["pair_type_counts"] == {"protac_component_library": 2}
    assert batch["source_counts"]["protacdb"] == 2


def test_protac8k_feature_matrices_form_trainable_ternary_samples(tmp_path):
    import numpy as np

    from humu_pretrain.data_loader import create_dataloaders

    sources = _write_minimal_humu_sources(tmp_path)
    (sources["protac8k"] / "protac8k.jsonl").unlink()
    feature_dir = sources["protac8k"] / "features"
    feature_dir.mkdir()
    np.save(feature_dir / "protac_feature.npy", np.ones((2, 167), dtype=np.float32))
    np.save(feature_dir / "target_feature.npy", np.ones((2, 30), dtype=np.float32))
    np.save(feature_dir / "e3_feature.npy", np.zeros((2, 30), dtype=np.float32))

    loader = create_dataloaders(
        {
            "batch_size": 1,
            "loss_weights": {"pocket_route": 0.0, "protac_ternary": 1.0},
            "data": {
                "pocket_source": str(sources["pocket"]),
                "route_source": str(sources["route"]),
                "protac8k_source": str(sources["protac8k"]),
                "num_workers": 0,
                "objective_sampling": {
                    "enabled": True,
                    "steps_per_epoch": 1,
                    "objectives": {"protac_ternary": 1},
                },
            },
        }
    )["paired"]

    batch = next(iter(loader))

    assert batch["pair_type_counts"] == {"protac_ternary": 1}
    assert batch["source_counts"]["protac8k"] == 1
    assert batch["ligand_smiles"] == [None]
    assert len(batch["protac_feature"][0]) == 167
    assert len(batch["target_feature"][0]) == 30
    assert len(batch["e3_feature"][0]) == 30


def test_pdcdb_candidate_tables_form_component_samples(tmp_path):
    from humu_pretrain.data_loader import create_dataloaders

    sources = _write_minimal_humu_sources(tmp_path)
    (sources["pdcdb"] / "pdc_components.jsonl").unlink()
    (sources["pdcdb"] / "linker.jsonl").write_text(
        json.dumps(
            {
                "Linker_ID": "LIN1",
                "Linker_Name": "Succinic Acid",
                "canonical_smiles": "O=C(O)CCC(=O)O",
                "source": "PDCdb",
                "record_type": "linker",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    loader = create_dataloaders(
        {
            "batch_size": 1,
            "loss_weights": {"pocket_route": 0.0, "pdc_component": 1.0},
            "data": {
                "pocket_source": str(sources["pocket"]),
                "route_source": str(sources["route"]),
                "pdcdb_source": str(sources["pdcdb"]),
                "num_workers": 0,
                "objective_sampling": {
                    "enabled": True,
                    "steps_per_epoch": 1,
                    "objectives": {"pdc_component": 1},
                },
            },
        }
    )["paired"]

    batch = next(iter(loader))

    assert batch["pair_type_counts"] == {"pdc_component": 1}
    assert batch["source_counts"]["pdcdb"] == 1
    assert batch["component_smiles"] == ["O=C(O)CCC(=O)O"]
    assert batch["peptide_sequence"] == ["CCKIGLFRWR"]


def test_pdcdb_candidate_tables_reject_generic_peptide_names(tmp_path):
    from humu_pretrain.data_loader import _iter_pdc_component_records

    sources = _write_minimal_humu_sources(tmp_path)
    (sources["pdcdb"] / "pdc_components.jsonl").unlink()
    (sources["pdcdb"] / "pdc.jsonl").write_text(
        json.dumps(
            {
                "PDC_ID": "PDC_GENERIC",
                "Peptide_ID": "PEP_GENERIC",
                "Peptide_Name": "DIPEPTIDE",
                "Linker_ID": "LIN1",
                "source": "PDCdb",
                "record_type": "pdc",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (sources["pdcdb"] / "linker.jsonl").write_text(
        json.dumps(
            {
                "Linker_ID": "LIN1",
                "canonical_smiles": "O=C(O)CCC(=O)O",
                "source": "PDCdb",
                "record_type": "linker",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = list(_iter_pdc_component_records(str(sources["pdcdb"])))

    assert records == []


def test_pdcdb_candidate_tables_reject_peptide_name_aliases(tmp_path):
    from humu_pretrain.data_loader import _iter_pdc_component_records

    sources = _write_minimal_humu_sources(tmp_path)
    (sources["pdcdb"] / "pdc_components.jsonl").unlink()
    (sources["pdcdb"] / "pdc.jsonl").write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {
                    "PDC_ID": "PDC_PENETRATIN",
                    "Peptide_ID": "PEP_PENETRATIN",
                    "Peptide_Name": "PENETRATIN",
                    "Linker_ID": "LIN1",
                    "source": "PDCdb",
                    "record_type": "pdc",
                },
                {
                    "PDC_ID": "PDC_TAT",
                    "Peptide_ID": "PEP_TAT",
                    "Peptide_Name": "TAT",
                    "Linker_ID": "LIN1",
                    "source": "PDCdb",
                    "record_type": "pdc",
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (sources["pdcdb"] / "linker.jsonl").write_text(
        json.dumps(
            {
                "Linker_ID": "LIN1",
                "canonical_smiles": "O=C(O)CCC(=O)O",
                "source": "PDCdb",
                "record_type": "linker",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = list(_iter_pdc_component_records(str(sources["pdcdb"])))

    assert records == []


def test_pdcdb_candidate_tables_use_manifest_linker_smiles_cache(tmp_path):
    from humu_pretrain.data_loader import _iter_pdc_component_records

    sources = _write_minimal_humu_sources(tmp_path)
    (sources["pdcdb"] / "pdc_components.jsonl").unlink()
    (sources["pdcdb"] / "linker.jsonl").write_text(
        json.dumps(
            {
                "Linker_ID": "LIN1",
                "PubChem_CID": "1110",
                "source": "PDCdb",
                "record_type": "linker",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (sources["pdcdb"] / "manifest.json").write_text(
        json.dumps(
            {
                "format": "pdcdb_jsonl_candidate",
                "linker_smiles_by_id": {"LIN1": "O=C(O)CCC(=O)O"},
            }
        ),
        encoding="utf-8",
    )

    records = list(_iter_pdc_component_records(str(sources["pdcdb"])))

    assert len(records) == 1
    assert records[0]["component_smiles"] == "O=C(O)CCC(=O)O"
    assert records[0]["peptide_sequence"] == "CCKIGLFRWR"


def test_pdcdb_manifest_component_records_work_without_ignored_jsonl(tmp_path):
    from humu_pretrain.data_loader import _iter_pdc_component_records

    pdcdb_dir = tmp_path / "pdcdb"
    pdcdb_dir.mkdir()
    (pdcdb_dir / "manifest.json").write_text(
        json.dumps(
            {
                "format": "pdcdb_jsonl_candidate",
                "pdc_component_records": [
                    {
                        "record_id": "PDC_MANIFEST",
                        "ligand_smiles": "CCOC(=O)N",
                        "component_smiles": "O=C(O)CCC(=O)O",
                        "component_type": "linker",
                        "source_dataset": "PDCdb",
                        "source_name": "pdcdb",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    records = list(_iter_pdc_component_records(str(pdcdb_dir)))

    assert len(records) == 1
    assert records[0]["record_id"] == "PDC_MANIFEST"
    assert records[0]["ligand_smiles"] == "CCOC(=O)N"
    assert records[0]["component_smiles"] == "O=C(O)CCC(=O)O"


def test_skempi_parser_handles_chain_and_multi_mutations():
    from humu_pretrain.data_loader import _parse_skempi_mutations

    assert _parse_skempi_mutations("LI38G") == [
        {"wildtype": "L", "chain_id": "I", "residue_number": 38, "mutant": "G"}
    ]
    assert _parse_skempi_mutations("SI40E,RI39M") == [
        {"wildtype": "S", "chain_id": "I", "residue_number": 40, "mutant": "E"},
        {"wildtype": "R", "chain_id": "I", "residue_number": 39, "mutant": "M"},
    ]
    assert _parse_skempi_mutations("bad") is None


def test_skempi_multi_mutation_payload_applies_all_residue_changes():
    from humu_pretrain.data_loader import _mutated_residue_payload

    payload = {
        "coords": [[0.0, 0.0, 0.0]] * 4,
        "elements": ["C", "C", "C", "C"],
        "residue_types": ["LEU", "SER", "ARG", "ALA"],
        "atom_chain_ids": ["I", "I", "I", "J"],
        "residue_ids": ["38", "40", "39", "38"],
    }
    mutations = [
        {"wildtype": "L", "chain_id": "I", "residue_number": 38, "mutant": "G"},
        {"wildtype": "S", "chain_id": "I", "residue_number": 40, "mutant": "E"},
        {"wildtype": "R", "chain_id": "I", "residue_number": 39, "mutant": "M"},
    ]

    mutated = _mutated_residue_payload(payload, mutations)

    assert mutated["residue_types"] == ["GLY", "GLU", "MET", "ALA"]
    assert payload["residue_types"] == ["LEU", "SER", "ARG", "ALA"]


def test_skempi2_multimutation_without_explicit_views_is_loaded(tmp_path):
    from humu_pretrain.data_loader import _iter_interface_mutation_records

    skempi_dir = tmp_path / "interface_skempi2"
    skempi_dir.mkdir()
    (skempi_dir / "manifest.json").write_text(
        json.dumps({"structure_archive": str(tmp_path / "skempi_v2_cache.zip")}),
        encoding="utf-8",
    )
    (skempi_dir / "skempi2.jsonl").write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {
                    "id": "SKEMPI2:multi",
                    "pdb_complex": "1ABC_A_B",
                    "mutations_cleaned": "RI48A,RI46A",
                    "affinity_mut_m": "1e-8",
                    "affinity_wt_m": "1e-9",
                },
                {
                    "id": "SKEMPI2:single",
                    "pdb_complex": "1ABC_A_B",
                    "mutations_cleaned": "RA48A",
                    "affinity_mut_m": "1e-8",
                    "affinity_wt_m": "1e-9",
                },
                {
                    "id": "SKEMPI2:bad",
                    "pdb_complex": "1ABC_A_B",
                    "mutations_cleaned": "not-a-mutation",
                    "affinity_mut_m": "1e-8",
                    "affinity_wt_m": "1e-9",
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    records = list(_iter_interface_mutation_records(str(skempi_dir)))

    assert [record["record_id"] for record in records] == ["SKEMPI2:multi", "SKEMPI2:single"]


def test_default_config_rejects_missing_dataset_contracts(tmp_path):
    from humu_pretrain.data_loader import preflight_humu_data_contract

    sources = _write_minimal_humu_sources(tmp_path)

    with pytest.raises(FileNotFoundError, match="protac8k_source"):
        preflight_humu_data_contract(
            {
                "data": {
                    "require_all_humu_sources": True,
                    "mol_source": str(sources["mol"]),
                    "pocket_source": str(sources["pocket"]),
                    "route_source": str(sources["route"]),
                    "joint_source": str(sources["joint"]),
                    "activity_source": str(sources["activity"]),
                    "protac_sources": [str(sources["protacpedia"])],
                    "protacdb_source": str(sources["protacdb"]),
                    "rcsb_mmcif_source": str(sources["rcsb_mmcif"]),
                    "interface_skempi2_source": str(sources["interface_skempi2"]),
                    "pdcdb_source": str(sources["pdcdb"]),
                    "route_eval_source": str(sources["route_eval"]),
                    "retropath_template_source": str(sources["retropath_templates"]),
                },
            }
        )


def test_create_dataloaders_requires_joint_source_for_enabled_joint_loss(tmp_path):
    from humu_pretrain.data_loader import create_dataloaders

    pocket_dir = tmp_path / "pocket"
    route_dir = tmp_path / "route"
    pocket_dir.mkdir()
    route_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="joint_source"):
        create_dataloaders(
            {
                "batch_size": 2,
                "loss_weights": {"pocket_route": 1.0},
                "data": {
                    "pocket_source": str(pocket_dir),
                    "route_source": str(route_dir),
                    "num_workers": 0,
                },
            }
        )


def test_in_batch_contrastive_loss_penalizes_shuffled_pairs():
    from humu_pretrain.pipeline import _in_batch_contrastive_loss
    from mf_humu.manifold.lorentz import LorentzManifold

    spatial = torch.zeros(3, 128)
    spatial[0, 0] = 0.2
    spatial[1, 1] = 0.2
    spatial[2, 2] = 0.2
    time_coord = torch.sqrt(1 + spatial.pow(2).sum(dim=-1, keepdim=True))
    anchors = torch.cat([time_coord, spatial], dim=-1)
    positives = anchors.clone()
    shuffled = positives[[1, 0, 2]]
    manifold = LorentzManifold(curvature=1.0)

    aligned = _in_batch_contrastive_loss(anchors, positives, manifold, temperature=0.1)
    misaligned = _in_batch_contrastive_loss(anchors, shuffled, manifold, temperature=0.1)

    assert aligned < misaligned


def test_retrieval_top1_accuracy_drops_when_pairs_are_shuffled():
    from humu_pretrain.pipeline import _retrieval_top1_accuracy
    from mf_humu.manifold.lorentz import LorentzManifold

    spatial = torch.zeros(3, 128)
    spatial[0, 0] = 0.2
    spatial[1, 1] = 0.2
    spatial[2, 2] = 0.2
    time_coord = torch.sqrt(1 + spatial.pow(2).sum(dim=-1, keepdim=True))
    anchors = torch.cat([time_coord, spatial], dim=-1)
    positives = anchors.clone()
    shuffled = positives[[1, 0, 2]]
    manifold = LorentzManifold(curvature=1.0)

    aligned = _retrieval_top1_accuracy(anchors, positives, manifold)
    misaligned = _retrieval_top1_accuracy(anchors, shuffled, manifold)

    assert aligned == 1.0
    assert misaligned < aligned


def test_compute_losses_uses_joint_objectives():
    from humu_pretrain.pipeline import _compute_losses

    spatial = torch.zeros(3, 128)
    spatial[0, 0] = 0.2
    spatial[1, 1] = 0.2
    spatial[2, 2] = 0.2
    time_coord = torch.sqrt(1 + spatial.pow(2).sum(dim=-1, keepdim=True))
    mol_emb = torch.cat([time_coord, spatial], dim=-1)
    pocket_emb = mol_emb.clone()
    route_emb = mol_emb.clone()
    off_manifold = mol_emb.clone()
    off_manifold[:, 0] = 1.0

    losses = _compute_losses(
        off_manifold,
        pocket_emb,
        route_emb,
        {
            "mol_pocket": 1.0,
            "mol_route": 1.0,
            "pocket_route": 1.0,
            "curvature_reg": 0.25,
        },
        {"temperature": 0.1, "negative_sampling": "in_batch"},
        route_mol_emb=off_manifold,
        pocket_route_pocket_emb=pocket_emb,
        pocket_route_route_emb=route_emb,
    )

    assert "l_pocket_route" in losses
    assert "l_curvature_reg" in losses
    assert losses["l_curvature_reg"] > 0
    expected_total = (
        losses["l_mol_pocket"]
        + losses["l_mol_route"]
        + losses["l_pocket_route"]
        + losses["l_curvature_reg"]
    )
    assert losses["total"] == expected_total
    assert losses["positive_distance"] < losses["negative_distance"]
    assert losses["retrieval_top1"] == 1.0
    assert losses["embedding_variance"] > 0
    assert losses["collapse_ratio"] == 0.0
    assert "lorentz_norm_deviation" in losses


def test_compute_losses_uses_protac_component_objective():
    from humu_pretrain.pipeline import _compute_losses

    spatial = torch.zeros(3, 128)
    spatial[0, 0] = 0.2
    spatial[1, 1] = 0.2
    spatial[2, 2] = 0.2
    time_coord = torch.sqrt(1 + spatial.pow(2).sum(dim=-1, keepdim=True))
    anchor_emb = torch.cat([time_coord, spatial], dim=-1)
    component_emb = anchor_emb.clone()

    losses = _compute_losses(
        None,
        None,
        None,
        {
            "mol_pocket": 0.0,
            "mol_route": 0.0,
            "pocket_route": 0.0,
            "protac_component": 1.0,
        },
        {"temperature": 0.1, "negative_sampling": "in_batch"},
        protac_anchor_emb=anchor_emb,
        protac_component_emb=component_emb,
    )

    assert "l_protac_component" in losses
    assert losses["l_protac_component"] > 0
    assert losses["total"] == losses["l_protac_component"]
    assert losses["retrieval_top1"] == 1.0


def test_default_config_one_batch_forward_gate(monkeypatch):
    import yaml

    from humu_pretrain.data_loader import create_dataloaders
    from humu_pretrain.pipeline import _forward_paired_batch
    from mf_encoders.humu_pocket.encoder import HUMUPocketEncoder

    class FastTextEncoder(torch.nn.Module):
        def __init__(self, dim: int = 128):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.dim = dim

        def forward(self, items):
            if not isinstance(items, list):
                items = [items]
            spatial = torch.zeros(len(items), self.dim) * self.weight
            for row, item in enumerate(items):
                text = item if isinstance(item, str) else json.dumps(item, sort_keys=True)
                spatial[row, row % self.dim] = (len(text) % 17 + 1) / 100.0
            time_coord = torch.sqrt(1 + spatial.pow(2).sum(dim=-1, keepdim=True))
            return torch.cat([time_coord, spatial], dim=-1)

    class FastPocketEncoder(torch.nn.Module):
        def __init__(self, dim: int = 128):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.dim = dim

        def forward(self, items):
            if not isinstance(items, list):
                items = [items]
            spatial = torch.zeros(len(items), self.dim) * self.weight
            for row, item in enumerate(items):
                coords = item.get("coords", []) if isinstance(item, dict) else []
                residues = item.get("residue_types", []) if isinstance(item, dict) else []
                spatial[row, 0] = float(len(coords)) / 100.0
                spatial[row, 1] = float(len(residues)) / 100.0
            time_coord = torch.sqrt(1 + spatial.pow(2).sum(dim=-1, keepdim=True))
            return torch.cat([time_coord, spatial], dim=-1)

    class FastFeatureEncoder(torch.nn.Module):
        def __init__(self, dim: int = 128):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.dim = dim

        def forward(self, items):
            if not isinstance(items, list):
                items = [items]
            spatial = torch.zeros(len(items), self.dim) * self.weight
            for row, item in enumerate(items):
                values = torch.as_tensor(item, dtype=torch.float32).reshape(-1)
                spatial[row, 0] = values[:8].mean() if values.numel() else 0.0
                spatial[row, 1] = float(values.numel()) / 1000.0
            time_coord = torch.sqrt(1 + spatial.pow(2).sum(dim=-1, keepdim=True))
            return torch.cat([time_coord, spatial], dim=-1)

    def fake_esm2_batch_embeddings(self, sequences):
        return torch.zeros(len(sequences), self.esm2_dim, device=self._param_device())

    monkeypatch.setattr(
        HUMUPocketEncoder,
        "_compute_esm2_batch_embeddings",
        fake_esm2_batch_embeddings,
    )

    cfg = yaml.safe_load((ROOT / "configs/models/humu_pretrain.yaml").read_text())
    cfg["device"] = "cpu"
    cfg["batch_size"] = 256
    cfg["max_samples"] = 100
    cfg["use_amp"] = False
    cfg["data"]["num_workers"] = 0
    cfg["data"]["pin_memory"] = False
    cfg["data"]["objective_sampling"]["steps_per_epoch"] = 1

    encoders = {
        "mol": FastTextEncoder(),
        "pocket": FastPocketEncoder(),
        "route": FastTextEncoder(),
        "protac_feature": FastFeatureEncoder(),
        "protac_context_feature": FastFeatureEncoder(),
    }
    batch = next(iter(create_dataloaders(cfg)["paired"]))
    losses = _forward_paired_batch(encoders, batch, cfg)

    assert torch.isfinite(losses["total"])
    for pair_type in (
        "mol_self",
        "mol_pocket",
        "mol_route",
        "mol_pocket_route",
        "activity_pair",
        "protac_component",
        "protac_component_library",
        "route_template",
        "protac_ternary",
        "protein_interface",
        "interface_mutation",
    ):
        assert batch["pair_type_counts"][pair_type] > 0
    assert "pdc_component" not in batch["pair_type_counts"]
    for source_name in (
        "mol",
        "pocket",
        "route",
        "joint",
        "activity",
        "bindingdb_activity",
        "protacpedia",
        "protacdb",
        "protac8k",
        "rcsb_mmcif",
        "interface_skempi2",
        "route_eval",
        "retropath_templates",
    ):
        assert batch["source_counts"][source_name] > 0
    for loss_key in (
        "l_mol_pocket",
        "l_mol_route",
        "l_pocket_route",
        "l_protac_component",
        "l_protac_component_library",
        "l_mol_self",
        "l_activity_supervised",
        "l_route_template",
        "l_protac_ternary",
        "l_protein_interface",
        "l_interface_mutation",
        "l_pdc_component",
    ):
        assert loss_key in losses
        assert torch.isfinite(losses[loss_key])


def test_compute_losses_uses_mol_self_activity_route_template_and_hard_negative():
    from humu_pretrain.pipeline import _compute_losses

    spatial = torch.zeros(3, 128)
    spatial[0, 0] = 0.2
    spatial[1, 1] = 0.2
    spatial[2, 2] = 0.2
    time_coord = torch.sqrt(1 + spatial.pow(2).sum(dim=-1, keepdim=True))
    anchors = torch.cat([time_coord, spatial], dim=-1)
    positives = anchors.clone()
    activity_delta = torch.tensor([2.0, 1.0, 0.5])

    losses = _compute_losses(
        None,
        None,
        None,
        {
            "mol_pocket": 0.0,
            "mol_route": 0.0,
            "mol_self": 1.0,
            "activity_supervised": 0.5,
            "route_template": 0.25,
            "hard_negative": 0.1,
        },
        {
            "temperature": 0.1,
            "negative_sampling": "hard_negative",
            "hard_negative_margin": 0.2,
        },
        mol_self_anchor_emb=anchors,
        mol_self_positive_emb=positives,
        activity_anchor_emb=anchors,
        activity_positive_emb=positives,
        activity_delta=activity_delta,
        route_template_mol_emb=anchors,
        route_template_route_emb=positives,
    )

    for key in (
        "l_mol_self",
        "l_activity_supervised",
        "l_route_template",
        "l_hard_negative",
    ):
        assert key in losses
        assert torch.isfinite(losses[key])
    assert losses["total"] == (
        losses["l_mol_pocket"]
        + losses["l_mol_route"]
        + losses["l_pocket_route"]
        + losses["l_protac_component"]
        + losses["l_mol_self"]
        + losses["l_activity_supervised"]
        + losses["l_route_template"]
        + losses["l_hard_negative"]
        + losses["l_curvature_reg"]
    )
    assert losses["mol_self_retrieval_top1"] == 1.0
    assert losses["route_template_retrieval_top1"] == 1.0
    assert "activity_pair_margin" in losses


def test_compute_losses_uses_large_source_objectives():
    from humu_pretrain.pipeline import _compute_losses

    spatial = torch.zeros(3, 128)
    spatial[0, 0] = 0.2
    spatial[1, 1] = 0.2
    spatial[2, 2] = 0.2
    time_coord = torch.sqrt(1 + spatial.pow(2).sum(dim=-1, keepdim=True))
    anchors = torch.cat([time_coord, spatial], dim=-1)
    positives = anchors.clone()
    affinity_delta = torch.tensor([2.0, 1.0, 0.5])

    losses = _compute_losses(
        None,
        None,
        None,
        {
            "mol_pocket": 0.0,
            "mol_route": 0.0,
            "protac_ternary": 1.0,
            "protein_interface": 0.5,
            "interface_mutation": 0.25,
            "pdc_component": 0.75,
        },
        {"temperature": 0.1, "negative_sampling": "in_batch"},
        protac_ternary_anchor_emb=anchors,
        protac_ternary_positive_emb=positives,
        protein_interface_anchor_emb=anchors,
        protein_interface_positive_emb=positives,
        interface_mutation_anchor_emb=anchors,
        interface_mutation_positive_emb=positives,
        interface_affinity_delta=affinity_delta,
        pdc_anchor_emb=anchors,
        pdc_component_emb=positives,
    )

    for key in (
        "l_protac_ternary",
        "l_protein_interface",
        "l_interface_mutation",
        "l_pdc_component",
    ):
        assert key in losses
        assert torch.isfinite(losses[key])
    assert losses["protac_ternary_retrieval_top1"] == 1.0
    assert losses["protein_interface_retrieval_top1"] == 1.0
    assert losses["pdc_component_retrieval_top1"] == 1.0
    assert "interface_mutation_margin" in losses


def test_forward_paired_batch_computes_all_enabled_losses():
    from humu_pretrain.pipeline import _forward_paired_batch

    class SmilesEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, items):
            if not isinstance(items, list):
                items = [items]
            spatial = torch.zeros(len(items), 128) * self.weight
            for index, item in enumerate(items):
                smiles = item if isinstance(item, str) else item.get("smiles", "")
                spatial[index, index % 8] = float(len(smiles)) / 100.0
            time_coord = torch.sqrt(1 + spatial.pow(2).sum(dim=-1, keepdim=True))
            return torch.cat([time_coord, spatial], dim=-1)

    class RouteEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, items):
            if not isinstance(items, list):
                items = [items]
            spatial = torch.zeros(len(items), 128) * self.weight
            for index, item in enumerate(items):
                spatial[index, index % 8] = float(len(item.get("reactions", []))) / 10.0
            time_coord = torch.sqrt(1 + spatial.pow(2).sum(dim=-1, keepdim=True))
            return torch.cat([time_coord, spatial], dim=-1)

    class PocketEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, items):
            if not isinstance(items, list):
                items = [items]
            spatial = torch.zeros(len(items), 128) * self.weight
            for index, item in enumerate(items):
                spatial[index, index % 8] = float(len(item.get("coords", []))) / 10.0
            time_coord = torch.sqrt(1 + spatial.pow(2).sum(dim=-1, keepdim=True))
            return torch.cat([time_coord, spatial], dim=-1)

    encoders = {
        "mol": SmilesEncoder(),
        "pocket": PocketEncoder(),
        "route": RouteEncoder(),
    }
    pocket = {
        "coords": [[0.0, 0.0, 0.0]],
        "elements": ["C"],
        "residue_types": ["ALA"],
    }
    route = {"id": "r1", "reactions": ["CCO>>CCN"], "steps": 1}
    batch = {
        "ligand_smiles": [
            "CCO",
            "CCN",
            "CCC",
            "CCCC",
            "CCCl",
            "CCBr",
            "CCCOCCN",
            None,
            None,
            None,
        ],
        "positive_smiles": ["CCO", "CCO", None, None, None, None, None, None, None, None],
        "component_smiles": [None, None, None, None, "COC", None, None, None, None, "COC"],
        "activity_delta": [None, 2.0, None, None, None, None, None, None, None, None],
        "target_ligand_smiles": [None, None, None, None, None, None, "CCO", None, None, None],
        "e3_ligand_smiles": [None, None, None, None, None, None, "CCN", None, None, None],
        "interface_affinity_delta": [None, None, None, None, None, None, None, None, 2.0, None],
        "pair_type": [
            "mol_self",
            "activity_pair",
            "mol_pocket",
            "mol_route",
            "protac_component",
            "route_template",
            "protac_ternary",
            "protein_interface",
            "interface_mutation",
            "pdc_component",
        ],
        "pocket": [None, None, pocket, None, None, None, None, None, None, None],
        "route": [None, None, None, route, None, route, None, None, None, None],
        "target_pocket": [None, None, None, None, None, None, pocket, None, None, None],
        "e3_pocket": [None, None, None, None, None, None, pocket, None, None, None],
        "interface_anchor": [None, None, None, None, None, None, None, pocket, pocket, None],
        "interface_positive": [None, None, None, None, None, None, None, pocket, pocket, None],
        "peptide_pocket": [None, None, None, None, None, None, None, None, None, pocket],
    }

    losses = _forward_paired_batch(
        encoders,
        batch,
        {
            "loss_weights": {
                "mol_self": 1.0,
                "activity_supervised": 1.0,
                "mol_pocket": 1.0,
                "mol_route": 1.0,
                "protac_component": 1.0,
                "route_template": 1.0,
                "protac_ternary": 1.0,
                "protein_interface": 1.0,
                "interface_mutation": 1.0,
                "pdc_component": 1.0,
            },
            "contrastive": {"temperature": 0.1, "negative_sampling": "in_batch"},
        },
    )

    for key in (
        "l_mol_self",
        "l_activity_supervised",
        "l_mol_pocket",
        "l_mol_route",
        "l_protac_component",
        "l_route_template",
        "l_protac_ternary",
        "l_protein_interface",
        "l_interface_mutation",
        "l_pdc_component",
    ):
        assert key in losses
        assert torch.isfinite(losses[key])


def test_forward_paired_batch_computes_pdc_molecule_component_loss():
    from humu_pretrain.pipeline import _forward_paired_batch

    class SmilesEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, items):
            if not isinstance(items, list):
                items = [items]
            spatial = torch.zeros(len(items), 128) * self.weight
            for index, item in enumerate(items):
                smiles = item if isinstance(item, str) else item.get("smiles", "")
                spatial[index, index % 4] = float(len(smiles)) / 100.0
            time_coord = torch.sqrt(1 + spatial.pow(2).sum(dim=-1, keepdim=True))
            return torch.cat([time_coord, spatial], dim=-1)

    class EmptyPocketEncoder(torch.nn.Module):
        def forward(self, items):
            if not isinstance(items, list):
                items = [items]
            spatial = torch.zeros(len(items), 128)
            time_coord = torch.ones(len(items), 1)
            return torch.cat([time_coord, spatial], dim=-1)

    class EmptyRouteEncoder(torch.nn.Module):
        def forward(self, items):
            if not isinstance(items, list):
                items = [items]
            spatial = torch.zeros(len(items), 128)
            time_coord = torch.ones(len(items), 1)
            return torch.cat([time_coord, spatial], dim=-1)

    encoders = {
        "mol": SmilesEncoder(),
        "pocket": EmptyPocketEncoder(),
        "route": EmptyRouteEncoder(),
    }
    batch = {
        "ligand_smiles": ["CCOC(=O)N", "CCNC(=O)O"],
        "component_smiles": ["O=C(O)CCC(=O)O", "C(CC(=O)O)CN"],
        "pair_type": ["pdc_component", "pdc_component"],
    }

    losses = _forward_paired_batch(
        encoders,
        batch,
        {
            "loss_weights": {"pdc_component": 1.0},
            "contrastive": {"temperature": 0.1, "negative_sampling": "in_batch"},
        },
    )

    assert "l_pdc_component" in losses
    assert torch.isfinite(losses["l_pdc_component"])
    assert losses["total"] == losses["l_pdc_component"]


def test_forward_paired_batch_computes_protac8k_feature_ternary_loss():
    from humu_pretrain.pipeline import _forward_paired_batch

    class FailingMolEncoder(torch.nn.Module):
        def forward(self, items):
            raise AssertionError("feature-only PROTAC-8K samples must not call mol encoder")

    class EmptyEncoder(torch.nn.Module):
        def forward(self, items):
            if not isinstance(items, list):
                items = [items]
            spatial = torch.zeros(len(items), 2)
            time_coord = torch.ones(len(items), 1)
            return torch.cat([time_coord, spatial], dim=-1)

    class FeatureEncoder(torch.nn.Module):
        def forward(self, items):
            rows = []
            for item in items:
                x = torch.tensor(float(item[0]), dtype=torch.float32)
                y = torch.tensor(float(item[1]), dtype=torch.float32)
                spatial = torch.stack([x, y])
                time_coord = torch.sqrt(torch.tensor(1.0) + (spatial * spatial).sum())
                rows.append(torch.cat([time_coord.reshape(1), spatial]))
            return torch.stack(rows, dim=0)

    batch = {
        "pair_type": ["protac_ternary", "protac_ternary"],
        "ligand_smiles": [None, None],
        "protac_feature": [
            [0.1, 0.0] + [0.0] * 165,
            [0.0, 0.1] + [0.0] * 165,
        ],
        "target_feature": [
            [0.1, 0.0] + [0.0] * 28,
            [0.0, 0.1] + [0.0] * 28,
        ],
        "e3_feature": [
            [0.1, 0.0] + [0.0] * 28,
            [0.0, 0.1] + [0.0] * 28,
        ],
    }

    losses = _forward_paired_batch(
        {
            "mol": FailingMolEncoder(),
            "pocket": EmptyEncoder(),
            "route": EmptyEncoder(),
            "protac_feature": FeatureEncoder(),
            "protac_context_feature": FeatureEncoder(),
        },
        batch,
        {
            "loss_weights": {"protac_ternary": 1.0},
            "contrastive": {"temperature": 0.1, "negative_sampling": "in_batch"},
        },
    )

    assert "l_protac_ternary" in losses
    assert torch.isfinite(losses["l_protac_ternary"])
    assert losses["protac_ternary_retrieval_top1"] >= 0.0


def test_route_template_objective_uses_distinct_template_embedding():
    from humu_pretrain.pipeline import _forward_paired_batch

    class FailingMol(torch.nn.Module):
        def forward(self, items):
            raise AssertionError("route_template must not require molecule encoding")

    class RouteEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.calls = []

        def forward(self, items):
            if not isinstance(items, list):
                items = [items]
            self.calls.append(items)
            spatial = torch.zeros(len(items), 128) * self.weight
            for index, item in enumerate(items):
                spatial[index, 0] = 0.1 if item.get("template") else 0.2
                spatial[index, 1] = float(len(item.get("reactions", []))) / 10.0
            time_coord = torch.sqrt(1 + spatial.pow(2).sum(dim=-1, keepdim=True))
            return torch.cat([time_coord, spatial], dim=-1)

    class EmptyTower(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, items):
            if not isinstance(items, list):
                items = [items]
            spatial = torch.zeros(len(items), 128) * self.weight
            time_coord = torch.ones(len(items), 1)
            return torch.cat([time_coord, spatial], dim=-1)

    route_encoder = RouteEncoder()
    route = {
        "id": "template-1",
        "reactions": ["[C:1]>>[C:1]O"],
        "template": "[C:1]>>[C:1]O",
        "steps": 1,
    }

    losses = _forward_paired_batch(
        {"mol": FailingMol(), "pocket": EmptyTower(), "route": route_encoder},
        {
            "ligand_smiles": [None],
            "pair_type": ["route_template"],
            "pocket": [None],
            "route": [route],
        },
        {
            "loss_weights": {
                "mol_pocket": 0.0,
                "mol_route": 0.0,
                "route_template": 1.0,
            },
            "contrastive": {"temperature": 0.1, "negative_sampling": "in_batch"},
        },
    )

    assert len(route_encoder.calls) == 2
    assert route_encoder.calls[0][0]["id"] == "template-1"
    assert route_encoder.calls[1][0]["id"] == "template:template-1"
    assert losses["l_route_template"] >= 0


def test_distributed_loader_shards_target_ratio_batch_sampler(tmp_path):
    from humu_pretrain.data_loader import create_dataloaders
    from humu_pretrain.pipeline import DistributedContext, _prepare_distributed_loaders

    sources = _write_minimal_humu_sources(tmp_path)
    cfg = {
        "batch_size": 4,
        "max_samples": 8,
        "data": {
            "mol_source": str(sources["mol"]),
            "pocket_source": str(sources["pocket"]),
            "route_source": str(sources["route"]),
            "joint_source": str(sources["joint"]),
            "activity_source": str(sources["activity"]),
            "protac_sources": [str(sources["protacpedia"])],
            "num_workers": 0,
            "objective_sampling": {
                "enabled": True,
                "steps_per_epoch": 2,
                "objectives": {
                    "mol_pocket": 1,
                    "mol_route": 1,
                    "mol_pocket_route": 1,
                    "protac_component": 1,
                },
            },
        },
        "loss_weights": {"pocket_route": 1.0, "protac_component": 1.0},
    }
    loader_rank0 = _prepare_distributed_loaders(
        create_dataloaders(cfg),
        DistributedContext(enabled=True, rank=0, world_size=2, local_rank=0),
    )["paired"]
    loader_rank1 = _prepare_distributed_loaders(
        create_dataloaders(cfg),
        DistributedContext(enabled=True, rank=1, world_size=2, local_rank=1),
    )["paired"]

    batch0 = next(iter(loader_rank0))
    batch1 = next(iter(loader_rank1))

    assert batch0["mol_id"] != batch1["mol_id"]
    assert batch0["pair_type_counts"] == batch1["pair_type_counts"]


def test_create_dataloaders_enforces_required_source_registry(tmp_path):
    from humu_pretrain.data_loader import create_dataloaders

    sources = _write_minimal_humu_sources(tmp_path)

    with pytest.raises(FileNotFoundError, match="protac8k_source"):
        create_dataloaders(
            {
                "batch_size": 2,
                "data": {
                    "require_all_humu_sources": True,
                    "mol_source": str(sources["mol"]),
                    "pocket_source": str(sources["pocket"]),
                    "route_source": str(sources["route"]),
                    "joint_source": str(sources["joint"]),
                    "activity_source": str(sources["activity"]),
                    "protac_sources": [str(sources["protacpedia"])],
                    "protacdb_source": str(sources["protacdb"]),
                    "rcsb_mmcif_source": str(sources["rcsb_mmcif"]),
                    "interface_skempi2_source": str(sources["interface_skempi2"]),
                    "pdcdb_source": str(sources["pdcdb"]),
                    "route_eval_source": str(sources["route_eval"]),
                    "retropath_template_source": str(sources["retropath_templates"]),
                    "num_workers": 0,
                },
            }
        )


def test_protacdb_component_records_do_not_claim_protac_anchor(tmp_path):
    from humu_pretrain.data_loader import PairedHUMUDataset

    pocket_dir = tmp_path / "pocket"
    route_dir = tmp_path / "route"
    protacdb_dir = tmp_path / "protacdb"
    for directory in (pocket_dir, route_dir, protacdb_dir):
        directory.mkdir()
    (pocket_dir / "index.jsonl").write_text("", encoding="utf-8")
    (route_dir / "routes.jsonl").write_text("", encoding="utf-8")
    (protacdb_dir / "e3_ligand.jsonl").write_text(
        json.dumps(
            {
                "record_id": "e3-1",
                "canonical_smiles": "NC1=CC=CC=C1",
                "component": "e3_ligand",
                "smiles_valid": True,
                "source": "PROTAC-DB",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = PairedHUMUDataset(
        str(pocket_dir),
        str(route_dir),
        protac_dirs=[str(protacdb_dir)],
    )

    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["pair_type"] == "protac_component_library"
    assert sample["component_smiles"] == "NC1=CC=CC=C1"
    assert sample["ligand_smiles"] is None


def test_early_stopping_triggers_after_patience_without_improvement():
    from humu_pretrain.pipeline import _early_stopping_state, _update_early_stopping

    cfg = {
        "early_stopping": {
            "enabled": True,
            "monitor": "val_loss",
            "mode": "min",
            "patience": 2,
            "min_delta": 0.1,
        }
    }
    state = _early_stopping_state(cfg)

    stop, state = _update_early_stopping(state, {"val_loss": 1.0}, epoch=0)
    assert stop is False
    assert state["best_value"] == 1.0

    stop, state = _update_early_stopping(state, {"val_loss": 0.95}, epoch=1)
    assert stop is False
    assert state["bad_checks"] == 1

    stop, state = _update_early_stopping(state, {"val_loss": 0.96}, epoch=2)
    assert stop is True
    assert state["stop_epoch"] == 3


def test_early_stopping_ignores_missing_monitor():
    from humu_pretrain.pipeline import _early_stopping_state, _update_early_stopping

    state = _early_stopping_state(
        {"early_stopping": {"enabled": True, "monitor": "val_loss", "patience": 1}}
    )

    stop, state = _update_early_stopping(state, {"retrieval_top1": 0.5}, epoch=0)

    assert stop is False
    assert state["best_value"] is None


def test_log_step_preserves_each_step_line(capsys):
    from humu_pretrain.pipeline import _log_step

    _log_step(0, 1, 2, {"total": torch.tensor(1.23456)}, 3e-4, preserve=True)
    _log_step(0, 2, 2, {"total": torch.tensor(0.98765)}, 2e-4, preserve=True)

    output = capsys.readouterr().out
    lines = output.splitlines()

    assert len(lines) == 2
    assert "Epoch 1" in lines[0]
    assert "Batch 1/2" in lines[0]
    assert "Loss: 1.2346" in lines[0]
    assert "LR: 0.000300" in lines[0]
    assert "Batch 2/2" in lines[1]
    assert "\r" not in output


def test_should_validate_epoch_requires_validation_loader_and_interval():
    from humu_pretrain.pipeline import _should_validate_epoch

    assert _should_validate_epoch(0, {"eval": {"every_n_epochs": 1}}, {"validation": [1]})
    assert not _should_validate_epoch(0, {"eval": {"every_n_epochs": 0}}, {"validation": [1]})
    assert not _should_validate_epoch(0, {"eval": {"every_n_epochs": 1}}, {})
    assert not _should_validate_epoch(1, {"eval": {"every_n_epochs": 3}}, {"validation": [1]})


def test_validate_epoch_returns_validation_metrics_and_restores_train_mode():
    from humu_pretrain.pipeline import DistributedContext, _validate_epoch

    class FakeTower(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, items):
            if not isinstance(items, list):
                items = [items]
            spatial = torch.zeros(len(items), 128)
            if items:
                spatial[:, 0] = torch.arange(len(items), dtype=torch.float32) * 0.1
            spatial = spatial * self.weight
            time_coord = torch.sqrt(1 + spatial.pow(2).sum(dim=-1, keepdim=True))
            return torch.cat([time_coord, spatial], dim=-1)

    encoders = {
        "mol": FakeTower(),
        "pocket": FakeTower(),
        "route": FakeTower(),
    }
    for model in encoders.values():
        model.train()
    loader = [
        {
            "ligand_smiles": ["CCO", "CCN"],
            "pair_type": ["mol_pocket", "mol_route"],
            "pocket": [
                {
                    "coords": [[0.0, 0.0, 0.0]],
                    "elements": ["C"],
                    "residue_types": ["ALA"],
                },
                None,
            ],
            "route": [
                None,
                {
                    "id": "route-1",
                    "reactions": ["CCO>>CCN"],
                    "steps": 1,
                    "intermediates": [],
                    "score": 0.0,
                },
            ],
        }
    ]

    metrics = _validate_epoch(
        encoders,
        loader,
        {
            "loss_weights": {"mol_pocket": 1.0, "mol_route": 0.5},
            "contrastive": {"temperature": 0.07, "negative_sampling": "in_batch"},
            "eval": {"metrics": ["mol_pocket_retrieval", "cliff_separation_auroc"]},
        },
        torch.device("cpu"),
        DistributedContext(enabled=False, rank=0, world_size=1, local_rank=0),
    )

    assert metrics["val_loss"] >= 0
    assert metrics["val_batches"] == 1
    assert metrics["val_samples"] == 2
    assert metrics["mol_pocket_retrieval"] == metrics["retrieval_top1"]
    assert metrics["distance_margin"] == pytest.approx(
        metrics["negative_distance"] - metrics["positive_distance"]
    )
    assert metrics["cliff_separation_auroc"] is None
    assert metrics["cliff_separation_auroc_status"] == "missing_activity_cliff_labels"
    assert all(model.training for model in encoders.values())


def test_validate_epoch_computes_activity_cliff_auroc_from_activity_source(tmp_path):
    from humu_pretrain.pipeline import DistributedContext, _validate_epoch

    activity_dir = tmp_path / "activity"
    activity_dir.mkdir()
    (activity_dir / "activity.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "ligand_smiles": "CCO",
                        "target_id": "CHEMBL_TARGET_1",
                        "activity_value": 8.0,
                        "activity_type": "pIC50",
                        "assay_id": "assay-1",
                    }
                ),
                json.dumps(
                    {
                        "ligand_smiles": "CCO",
                        "target_id": "CHEMBL_TARGET_1",
                        "activity_value": 8.2,
                        "activity_type": "pIC50",
                        "assay_id": "assay-2",
                    }
                ),
                json.dumps(
                    {
                        "ligand_smiles": "CCN",
                        "target_id": "CHEMBL_TARGET_1",
                        "activity_value": 5.0,
                        "activity_type": "pIC50",
                        "assay_id": "assay-3",
                    }
                ),
                json.dumps(
                    {
                        "ligand_smiles": "c1ccccc1",
                        "target_id": "CHEMBL_TARGET_1",
                        "activity_value": 5.1,
                        "activity_type": "pIC50",
                        "assay_id": "assay-4",
                    }
                ),
                json.dumps(
                    {
                        "ligand_smiles": "CC(=O)O",
                        "target_id": "CHEMBL_TARGET_1",
                        "activity_value": 5.2,
                        "activity_type": "pIC50",
                        "assay_id": "assay-5",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeTower(torch.nn.Module):
        def __init__(self, values: dict[str, float] | None = None):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.values = values or {}

        def forward(self, items):
            if not isinstance(items, list):
                items = [items]
            spatial = torch.zeros(len(items), 128)
            for index, item in enumerate(items):
                key = item if isinstance(item, str) else item.get("id", "")
                spatial[index, 0] = self.values.get(key, 0.0)
            spatial = spatial * self.weight
            time_coord = torch.sqrt(1 + spatial.pow(2).sum(dim=-1, keepdim=True))
            return torch.cat([time_coord, spatial], dim=-1)

    encoders = {
        "mol": FakeTower(
            {
                "[CH3:1][CH2:2][OH:3]": 0.0,
                "CCN": 0.1,
                "c1ccccc1": 4.0,
                "CC(=O)O": 5.0,
            }
        ),
        "pocket": FakeTower(),
        "route": FakeTower(),
    }
    loader = [
        {
            "ligand_smiles": [
                "[CH3:1][CH2:2][OH:3]",
                "CCN",
                "c1ccccc1",
                "CC(=O)O",
            ],
            "pair_type": ["mol_pocket", "mol_pocket", "mol_pocket", "mol_pocket"],
            "pocket": [
                {
                    "id": "p1",
                    "coords": [[0.0, 0.0, 0.0]],
                    "elements": ["C"],
                    "residue_types": ["ALA"],
                },
                {
                    "id": "p2",
                    "coords": [[0.0, 0.0, 0.0]],
                    "elements": ["C"],
                    "residue_types": ["ALA"],
                },
                {
                    "id": "p3",
                    "coords": [[0.0, 0.0, 0.0]],
                    "elements": ["C"],
                    "residue_types": ["ALA"],
                },
                {
                    "id": "p4",
                    "coords": [[0.0, 0.0, 0.0]],
                    "elements": ["C"],
                    "residue_types": ["ALA"],
                },
            ],
            "route": [None, None, None, None],
        }
    ]

    metrics = _validate_epoch(
        encoders,
        loader,
        {
            "loss_weights": {"mol_pocket": 1.0},
            "contrastive": {"temperature": 0.07, "negative_sampling": "in_batch"},
            "data": {"activity_source": str(activity_dir)},
            "eval": {
                "metrics": ["cliff_separation_auroc"],
                "activity_cliff_similarity_threshold": 0.3,
                "activity_cliff_delta_threshold": 1.0,
            },
        },
        torch.device("cpu"),
        DistributedContext(enabled=False, rank=0, world_size=1, local_rank=0),
    )

    assert metrics["cliff_separation_auroc"] == pytest.approx(1.0)
    assert "cliff_separation_auroc_status" not in metrics


def test_load_activity_records_merges_activity_sources(tmp_path):
    from humu_pretrain.pipeline import _activity_sources_from_config, _load_activity_records

    chembl_dir = tmp_path / "activity"
    bindingdb_dir = tmp_path / "bindingdb_activity"
    chembl_dir.mkdir()
    bindingdb_dir.mkdir()
    (chembl_dir / "activity.jsonl").write_text(
        json.dumps(
            {
                "ligand_smiles": "CCO",
                "target_id": "CHEMBL_TARGET_1",
                "activity_value": 8.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (bindingdb_dir / "activity.jsonl").write_text(
        json.dumps(
            {
                "ligand_smiles": "CCN",
                "target_id": "P03367",
                "activity_value": 7.2,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    sources = _activity_sources_from_config(
        {
            "data": {
                "activity_source": str(chembl_dir),
                "activity_sources": [str(bindingdb_dir)],
            }
        }
    )
    records = _load_activity_records(sources)

    assert sources == [str(chembl_dir), str(bindingdb_dir)]
    assert sum(len(items) for items in records.values()) == 2


def test_validate_epoch_reports_missing_activity_thresholds(tmp_path):
    from humu_pretrain.pipeline import DistributedContext, _validate_epoch

    activity_dir = tmp_path / "activity"
    activity_dir.mkdir()
    (activity_dir / "activity.jsonl").write_text(
        json.dumps(
            {
                "ligand_smiles": "CCO",
                "target_id": "CHEMBL_TARGET_1",
                "activity_value": 5.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeTower(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, items):
            if not isinstance(items, list):
                items = [items]
            spatial = torch.zeros(len(items), 128) * self.weight
            time_coord = torch.ones(len(items), 1)
            return torch.cat([time_coord, spatial], dim=-1)

    encoders = {
        "mol": FakeTower(),
        "pocket": FakeTower(),
        "route": FakeTower(),
    }
    loader = [
        {
            "ligand_smiles": ["CCO"],
            "pair_type": ["mol_pocket"],
            "pocket": [
                {"coords": [[0.0, 0.0, 0.0]], "elements": ["C"], "residue_types": ["ALA"]}
            ],
            "route": [None],
        }
    ]

    metrics = _validate_epoch(
        encoders,
        loader,
        {
            "loss_weights": {"mol_pocket": 1.0},
            "contrastive": {"temperature": 0.07, "negative_sampling": "in_batch"},
            "data": {"activity_source": str(activity_dir)},
            "eval": {"metrics": ["cliff_separation_auroc"]},
        },
        torch.device("cpu"),
        DistributedContext(enabled=False, rank=0, world_size=1, local_rank=0),
    )

    assert metrics["cliff_separation_auroc"] is None
    assert metrics["cliff_separation_auroc_status"] == "missing_activity_cliff_thresholds"


def test_write_validation_metrics_appends_jsonl(tmp_path):
    from humu_pretrain.pipeline import _write_validation_metrics

    path = _write_validation_metrics(
        tmp_path,
        4,
        {
            "val_loss": 1.25,
            "retrieval_top1": 0.5,
            "cliff_separation_auroc": None,
        },
    )

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert path.name == "validation_metrics.jsonl"
    assert records == [
        {
            "epoch": 5,
            "val_loss": 1.25,
            "retrieval_top1": 0.5,
            "cliff_separation_auroc": None,
        }
    ]


def test_humu_background_script_streams_realtime_logs():
    root = Path(__file__).resolve().parents[2]
    script = root / "pipelines" / "humu_pretrain" / "run_humu_4h200_background.sh"
    env_file = root / "pipelines" / "humu_pretrain" / ".env"

    assert script.exists()
    assert os.access(script, os.X_OK)
    assert env_file.exists()

    text = script.read_text()
    assert "stdbuf -oL -eL" in text
    assert "tee -a" in text
    assert "RUN_MANIFEST" in text
    assert "sha256sum" in text
    assert "PID_FILE" in text
    assert "LOG_FILE" in text
    assert "torch.distributed.run" in text
    assert "--nproc_per_node" in text
    assert "source \"$ENV_FILE\"" in text
    assert "PYTORCH_CUDA_ALLOC_CONF" in text

    env_text = env_file.read_text()
    assert "CONFIG_PATH=" in env_text
    assert "PYTHON_BIN=" in env_text
    assert "NPROC_PER_NODE=" in env_text
    assert "CUDA_VISIBLE_DEVICES=" in env_text
    assert "PYTORCH_CUDA_ALLOC_CONF=" in env_text
    assert "PYTHONUNBUFFERED=" in env_text

    result = subprocess.run(
        [
            "bash",
            "-c",
            (
                "set -Eeuo pipefail; "
                "PROJECT_ROOT=\"$1\"; "
                "RUN_NAME=resume_run; "
                "CUDA_VISIBLE_DEVICES=7; "
                "PYTORCH_CUDA_ALLOC_CONF=custom_alloc; "
                "source \"$2\"; "
                "printf '%s\\n%s\\n%s\\n' "
                "\"$RUN_NAME\" \"$CUDA_VISIBLE_DEVICES\" \"$PYTORCH_CUDA_ALLOC_CONF\""
            ),
            "bash",
            str(root),
            str(env_file),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.splitlines() == ["resume_run", "7", "custom_alloc"]

    stop_script = root / "pipelines" / "humu_pretrain" / "stop_humu_background.sh"
    assert stop_script.exists()
    assert os.access(stop_script, os.X_OK)
    stop_text = stop_script.read_text()
    assert "PID_FILE" in stop_text
    assert "pkill" not in stop_text
    assert "kill -TERM" in stop_text


def test_checkpoint_save_load():
    import torch.optim as optim
    from humu_pretrain.pipeline import _build_encoders, _load_checkpoint, _save_checkpoint

    cfg = {"embed_dim": 129, "curvature": 1.0, "encoders": {}}
    device = torch.device("cpu")
    encoders = _build_encoders(cfg, device)

    optimizer = optim.AdamW(
        [p for m in encoders.values() for p in m.parameters()],
        lr=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_checkpoint.pt")
        _save_checkpoint(encoders, optimizer, scheduler, 5, 0.5, path)

        encoders2 = _build_encoders(cfg, device)
        epoch, loss = _load_checkpoint(encoders2, path, device)
        assert epoch == 6
        assert loss == 0.5


def test_checkpoint_excludes_frozen_esm2_model_weights(tmp_path):
    import torch.optim as optim
    from humu_pretrain.pipeline import _save_checkpoint

    class PocketWithFrozenESM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(1, 1)
            self.inner = torch.nn.Module()
            self.inner._esm2_model = torch.nn.Linear(1, 1)
            self.inner._esm2_projection = torch.nn.Linear(1, 1)

    encoders = {"pocket": PocketWithFrozenESM()}
    optimizer = optim.AdamW(encoders["pocket"].parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
    checkpoint_path = tmp_path / "checkpoint.pt"

    _save_checkpoint(encoders, optimizer, scheduler, 0, 1.0, checkpoint_path)

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    pocket_state = state["encoder_pocket"]
    assert "inner._esm2_projection.weight" in pocket_state
    assert not any(key.startswith("inner._esm2_model.") for key in pocket_state)


def test_load_checkpoint_maps_legacy_encoder_projection_keys(tmp_path):
    from humu_pretrain.pipeline import _build_encoders, _load_checkpoint

    cfg = {
        "embed_dim": 9,
        "curvature": 1.0,
        "encoders": {
            "mol": {"hidden_dim": 9},
            "pocket": {"hidden_dim": 9},
            "route": {"hidden_dim": 8, "n_heads": 8},
        },
    }
    encoders = _build_encoders(cfg, torch.device("cpu"))
    checkpoint_path = tmp_path / "legacy_checkpoint.pt"
    legacy_weight = torch.full((9, 9), 0.25)
    legacy_bias = torch.full((9,), 0.5)
    state = {
        "epoch": 0,
        "loss": 1.0,
        "encoder_mol": {
            "inner._atom_projection.weight": legacy_weight,
            "inner._atom_projection.bias": legacy_bias,
        },
        "encoder_pocket": {
            "inner._point_projection.weight": legacy_weight[:, :12],
            "inner._point_projection.bias": legacy_bias,
        },
        "encoder_route": {
            "inner._route_projection.0.weight": torch.full((8, 18), 0.1),
            "inner._route_projection.0.bias": torch.full((8,), 0.2),
            "inner._route_projection.2.weight": legacy_weight[:, :8],
            "inner._route_projection.2.bias": legacy_bias,
        },
    }
    torch.save(state, checkpoint_path)

    _load_checkpoint(encoders, checkpoint_path, torch.device("cpu"))

    assert torch.allclose(encoders["mol"].inner._atom_projection[-1].weight, legacy_weight)
    assert torch.allclose(encoders["mol"].inner._atom_projection[-1].bias, legacy_bias)
    assert torch.allclose(
        encoders["pocket"].inner._point_projection[-1].weight,
        legacy_weight[:, :12],
    )
    assert torch.allclose(
        encoders["route"].inner._output_projection.weight,
        legacy_weight[:, :8],
    )


def test_load_checkpoint_restores_training_state_for_resume(tmp_path):
    import torch.optim as optim
    from humu_pretrain.pipeline import _load_checkpoint, _save_checkpoint

    encoders = {
        "mol": torch.nn.Linear(1, 1),
        "pocket": torch.nn.Linear(1, 1),
        "route": torch.nn.Linear(1, 1),
    }
    optimizer = optim.AdamW(
        [p for model in encoders.values() for p in model.parameters()],
        lr=1e-3,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
    loss = sum(model(torch.ones(1, 1)).sum() for model in encoders.values())
    loss.backward()
    optimizer.step()
    scheduler.step(2.5)

    checkpoint_path = tmp_path / "checkpoint_epoch_0003.pt"
    _save_checkpoint(encoders, optimizer, scheduler, 2, 4.1708, checkpoint_path)

    resumed_encoders = {
        "mol": torch.nn.Linear(1, 1),
        "pocket": torch.nn.Linear(1, 1),
        "route": torch.nn.Linear(1, 1),
    }
    resumed_optimizer = optim.AdamW(
        [p for model in resumed_encoders.values() for p in model.parameters()],
        lr=5e-1,
    )
    resumed_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        resumed_optimizer,
        T_0=10,
    )

    next_epoch, best_loss = _load_checkpoint(
        resumed_encoders,
        checkpoint_path,
        torch.device("cpu"),
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
    )

    assert next_epoch == 3
    assert best_loss == 4.1708
    assert resumed_optimizer.state_dict()["state"]
    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(
        optimizer.param_groups[0]["lr"]
    )
    assert resumed_scheduler.state_dict()["last_epoch"] == scheduler.state_dict()["last_epoch"]


def test_training_resume_legacy_step_checkpoint_restarts_checkpoint_epoch(tmp_path):
    import torch.optim as optim
    from humu_pretrain.pipeline import _load_training_checkpoint, _save_checkpoint

    encoders = {
        "mol": torch.nn.Linear(1, 1),
        "pocket": torch.nn.Linear(1, 1),
        "route": torch.nn.Linear(1, 1),
    }
    optimizer = optim.AdamW(
        [p for model in encoders.values() for p in model.parameters()],
        lr=1e-3,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
    checkpoint_path = tmp_path / "checkpoint_step_00010500.pt"
    _save_checkpoint(encoders, optimizer, scheduler, 4, 1.7272, checkpoint_path)

    resumed = _load_training_checkpoint(
        encoders,
        checkpoint_path,
        torch.device("cpu"),
        optimizer=optimizer,
        scheduler=scheduler,
    )

    assert resumed.start_epoch == 4
    assert resumed.epoch_step == 0
    assert resumed.best_loss == 1.7272
    assert resumed.epoch_loss_sum == 0.0
    assert resumed.epoch_loss_count == 0


def test_rotate_checkpoints_skips_cleanup_when_keep_last_n_is_none(tmp_path):
    from humu_pretrain.pipeline import _rotate_checkpoints

    for epoch in (5, 10, 15, 20):
        (tmp_path / f"checkpoint_epoch_{epoch:04d}.pt").write_text(
            str(epoch),
            encoding="utf-8",
        )

    _rotate_checkpoints(tmp_path, None)

    assert sorted(path.name for path in tmp_path.glob("checkpoint_epoch_*.pt")) == [
        "checkpoint_epoch_0005.pt",
        "checkpoint_epoch_0010.pt",
        "checkpoint_epoch_0015.pt",
        "checkpoint_epoch_0020.pt",
    ]


def test_step_checkpoint_saves_training_resume_metadata(tmp_path):
    import torch.optim as optim
    from humu_pretrain.pipeline import _load_training_checkpoint, _save_checkpoint

    encoders = {
        "mol": torch.nn.Linear(1, 1),
        "pocket": torch.nn.Linear(1, 1),
        "route": torch.nn.Linear(1, 1),
    }
    optimizer = optim.AdamW(
        [p for model in encoders.values() for p in model.parameters()],
        lr=1e-3,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
    checkpoint_path = tmp_path / "checkpoint_step_00010500.pt"

    _save_checkpoint(
        encoders,
        optimizer,
        scheduler,
        4,
        1.7272,
        checkpoint_path,
        checkpoint_type="step",
        global_step=10500,
        epoch_step=1788,
        n_batches=2178,
        best_loss=1.55,
        epoch_loss_sum=3500.0,
        epoch_loss_count=1788,
    )

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert state["checkpoint_type"] == "step"
    assert state["global_step"] == 10500
    assert state["epoch_step"] == 1788
    assert state["n_batches"] == 2178
    assert state["best_loss"] == 1.55
    assert state["epoch_loss_sum"] == 3500.0
    assert state["epoch_loss_count"] == 1788

    resumed = _load_training_checkpoint(
        encoders,
        checkpoint_path,
        torch.device("cpu"),
        optimizer=optimizer,
        scheduler=scheduler,
    )

    assert resumed.start_epoch == 4
    assert resumed.epoch_step == 1788
    assert resumed.best_loss == 1.55
    assert resumed.epoch_loss_sum == 3500.0
    assert resumed.epoch_loss_count == 1788


def test_hfm3d_generator_save_load():
    from mf_generators.hfm_3d.generator import HFM3DGenerator
    gen = HFM3DGenerator(device="cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "hfm3d_test.pt")
        gen.save_checkpoint(path)
        assert os.path.exists(path)

        gen2 = HFM3DGenerator(checkpoint_path=path, device="cpu")
        # After loading checkpoint, model params should be loaded
        assert gen2._model is not None
        assert gen2._decoder is not None


@pytest.mark.asyncio
async def test_hfm3d_generate_smoke():
    from mf_generators.hfm_3d.generator import HFM3DGenerator
    gen = HFM3DGenerator(device="cpu", mode="local_demo")
    mols = await gen.generate(batch_size=3)
    assert len(mols) == 3
    for mol in mols:
        assert mol.smiles is not None
        assert len(mol.smiles) > 0


@pytest.mark.asyncio
async def test_hfm3d_generation_runs_lorentz_flow_before_decoding():
    from mf_generators.hfm_3d.generator import HFM3DGenerator

    class RecordingFlow:
        def __init__(self) -> None:
            self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

        def compute_vector_field(self, latent_points: torch.Tensor, t: torch.Tensor):
            self.calls.append((latent_points.detach().clone(), t.detach().clone()))
            velocity = torch.zeros_like(latent_points)
            velocity[..., 1] = 0.05
            return velocity

    flow = RecordingFlow()
    gen = HFM3DGenerator(
        device="cpu",
        mode="local_demo",
        smiles_decoder=lambda _embedding: "CCO",
    )
    gen._model = flow

    molecules = await gen.generate(batch_size=1, sampling_seed=11, flow_steps=2)

    assert len(flow.calls) == 2
    assert molecules[0].metadata["flow_steps"] == "2"
    pre_flow_latent = json.loads(molecules[0].metadata["pre_flow_latent"])
    latent = json.loads(molecules[0].metadata["latent"])
    assert pre_flow_latent != latent


@pytest.mark.asyncio
async def test_hfm3d_generation_uses_molecular_decoder_geometry_after_flow():
    from mf_generators.hfm_3d.generator import HFM3DGenerator

    class RecordingFlow:
        def compute_vector_field(self, latent_points: torch.Tensor, t: torch.Tensor):
            velocity = torch.zeros_like(latent_points)
            velocity[..., 1] = 0.05
            return velocity

    class GeometryDecoder:
        def __init__(self) -> None:
            self.calls: list[torch.Tensor] = []

        def decode(self, embedding: torch.Tensor) -> dict:
            self.calls.append(embedding.detach().cpu().clone())
            return {
                "id": "geometry-decoder",
                "smiles": "CCO",
                "atom_types": ["C", "C", "O"],
                "coordinates": [
                    [0.0, 0.0, 0.0],
                    [1.4, 0.0, 0.0],
                    [2.1, 0.8, 0.0],
                ],
                "metadata": {"decoder_kind": "geometry"},
            }

    decoder = GeometryDecoder()
    gen = HFM3DGenerator(
        device="cpu",
        mode="local_demo",
        molecular_decoder=decoder,
    )
    gen._model = RecordingFlow()

    molecules = await gen.generate(batch_size=1, sampling_seed=11, flow_steps=2)

    assert len(decoder.calls) == 1
    molecule = molecules[0]
    pre_flow_latent = json.loads(molecule.metadata["pre_flow_latent"])
    latent = json.loads(molecule.metadata["latent"])
    assert decoder.calls[0].tolist() == pytest.approx(latent)
    assert decoder.calls[0].tolist() != pytest.approx(pre_flow_latent)
    assert molecule.smiles == "CCO"
    assert molecule.sdf_bytes is not None
    assert b"V2000" in molecule.sdf_bytes or b"V3000" in molecule.sdf_bytes
    assert molecule.metadata["decoder_entry_id"] == "geometry-decoder"
    assert molecule.metadata["decoder_mode"] == "molecular_decoder"
    assert molecule.metadata["decoder_kind"] == "geometry"
    assert json.loads(molecule.metadata["decoded_atom_types"]) == ["C", "C", "O"]
    assert json.loads(molecule.metadata["decoded_coordinates"]) == [
        [0.0, 0.0, 0.0],
        [1.4, 0.0, 0.0],
        [2.1, 0.8, 0.0],
    ]


@pytest.mark.asyncio
async def test_hfm3d_production_requires_decoder_artifact():
    from mf_generators.hfm_3d.generator import HFM3DGenerator

    gen = HFM3DGenerator(device="cpu")

    with pytest.raises(RuntimeError, match="decoder artifact"):
        await gen.generate(batch_size=1)


@pytest.mark.asyncio
async def test_hfm3d_generation_uses_decoder_artifact_conformer_and_provenance(tmp_path):
    from mf_generators.hfm_3d.generator import HFM3DGenerator

    checkpoint_path = tmp_path / "hfm.pt"
    HFM3DGenerator(device="cpu", mode="local_demo").save_checkpoint(str(checkpoint_path))
    decoder_path = tmp_path / "decoder.json"
    decoder_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "id": "ethanol",
                        "smiles": "CCO",
                        "latent": [0.0] * 129,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    generator = HFM3DGenerator(
        checkpoint_path=str(checkpoint_path),
        decoder_path=str(decoder_path),
        device="cpu",
    )

    molecules = await generator.generate(batch_size=1, sampling_seed=7)

    assert len(molecules) == 1
    molecule = molecules[0]
    assert molecule.smiles == "CCO"
    assert molecule.sdf_bytes is not None
    assert b"V2000" in molecule.sdf_bytes or b"V3000" in molecule.sdf_bytes
    assert molecule.metadata["generator_name"] == "hfm_3d"
    assert molecule.metadata["checkpoint"] == str(checkpoint_path)
    assert molecule.metadata["decode_artifact"] == str(decoder_path)
    assert molecule.metadata["sampling_seed"] == "7"
    assert molecule.metadata["decoder_entry_id"] == "ethanol"
    assert json.loads(molecule.metadata["latent"])


@pytest.mark.asyncio
async def test_hfm3d_generation_uses_decoder_artifact_sdf_geometry(tmp_path):
    from mf_generators.hfm_3d.generator import HFM3DGenerator

    checkpoint_path = tmp_path / "hfm.pt"
    fixture_generator = HFM3DGenerator(device="cpu", mode="local_demo")
    fixture_generator.save_checkpoint(str(checkpoint_path))
    artifact_sdf = fixture_generator._build_conformer("CCO", seed=123).decode("utf-8")
    decoder_path = tmp_path / "decoder.json"
    decoder_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "id": "ethanol",
                        "smiles": "CCO",
                        "latent": [0.0] * 129,
                        "sdf": artifact_sdf,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    generator = HFM3DGenerator(
        checkpoint_path=str(checkpoint_path),
        decoder_path=str(decoder_path),
        device="cpu",
    )

    molecules = await generator.generate(batch_size=1, sampling_seed=7)

    molecule = molecules[0]
    assert molecule.smiles == "CCO"
    assert molecule.sdf_bytes == artifact_sdf.encode("utf-8")
    assert molecule.metadata["decoder_entry_id"] == "ethanol"
    assert molecule.metadata["decoder_mode"] == "artifact_sdf"


def test_hfm3d_decoder_artifact_rejects_invalid_sdf_geometry(tmp_path):
    from mf_generators.hfm_3d.generator import HFM3DGenerator

    checkpoint_path = tmp_path / "hfm.pt"
    HFM3DGenerator(device="cpu", mode="local_demo").save_checkpoint(str(checkpoint_path))
    decoder_path = tmp_path / "decoder.json"
    decoder_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "id": "ethanol",
                        "smiles": "CCO",
                        "latent": [0.0] * 129,
                        "sdf": "not a mol block",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="decoder entry sdf"):
        HFM3DGenerator(
            checkpoint_path=str(checkpoint_path),
            decoder_path=str(decoder_path),
            device="cpu",
        )
