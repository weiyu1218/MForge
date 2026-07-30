"""Unit tests for Layer 3 generators — basic generation validation."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import math
import struct
import subprocess
import sys
from pathlib import Path

import pytest
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[2]


def _run_async_iter(ait):
    """Collect an AsyncIterator into a list via asyncio.run."""
    async def _collect():
        return [item async for item in ait]
    return asyncio.run(_collect())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("module_name", "target_name"),
    [
        ("mf_generators.hfm_3d.generator", "hfm-validation"),
        ("mf_generators.crem_3d.generator", "crem-validation"),
        ("mf_generators.fragfm.generator", "fragfm-validation"),
        ("mf_generators.mmpt_rag.generator", "mmpt-validation"),
    ],
)
async def test_validation_bootstrap_concurrent_first_publish_is_idempotent(
    module_name: str,
    target_name: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = importlib.import_module(module_name)
    target = tmp_path / target_name
    original_probe = module._probe_validation_artifacts
    temporary_probe_count = 0
    both_temporary_artifacts_ready = asyncio.Event()

    async def synchronize_temporary_probes(paths: dict[str, Path]) -> None:
        nonlocal temporary_probe_count
        await original_probe(paths)
        if paths["metadata"].parent == target:
            return
        temporary_probe_count += 1
        if temporary_probe_count == 2:
            both_temporary_artifacts_ready.set()
        await asyncio.wait_for(
            both_temporary_artifacts_ready.wait(),
            timeout=30,
        )

    monkeypatch.setattr(
        module,
        "_probe_validation_artifacts",
        synchronize_temporary_probes,
    )

    first, second = await asyncio.gather(
        module.bootstrap_validation_artifacts(target),
        module.bootstrap_validation_artifacts(target),
    )

    assert first == second
    assert first == module._validation_artifact_paths(target)
    await original_probe(first)


def _expected_linear_kd_loss(
    projection: dict,
    features: list[float],
    teacher_embedding: list[float],
) -> float:
    normalized_features = [
        (feature - mean) / scale
        for feature, mean, scale in zip(
            features,
            projection["feature_mean"],
            projection["feature_scale"],
            strict=True,
        )
    ]
    prediction = [
        sum(
            weight * feature
            for weight, feature in zip(row, normalized_features, strict=True)
        )
        + bias
        for row, bias in zip(
            projection["weights"],
            projection["bias"],
            strict=True,
        )
    ]
    mse = sum(
        (actual - expected) ** 2
        for actual, expected in zip(
            prediction,
            teacher_embedding,
            strict=True,
        )
    ) / len(teacher_embedding)
    flattened_parameters = [
        weight for row in projection["weights"] for weight in row
    ] + list(projection["bias"])
    regularization_loss = projection["regularization"] * (
        sum(value**2 for value in flattened_parameters)
        / len(flattened_parameters)
    )
    return projection["kd_weight"] * (mse + regularization_loss)


def _make_hciv_cone(dim: int = 16, seed: int = 42):
    """Create test HCIV and IntentCone."""
    from cig_compiler_svc.domain.hciv_generator import generate_intent_cone, generate_random_hciv
    hciv = generate_random_hciv(dim=dim, seed=seed)
    cone = generate_intent_cone(apex=hciv, dim=dim, seed=seed)
    return hciv, cone


def _make_cig():
    """Create a minimal CIG for testing."""
    from cig_compiler_svc.domain.stages.stage2_cig_build import build_cig
    extracted = {
        "properties": [{"name": "qed", "direction": "maximize", "priority": 1}],
        "constraints": {},
    }
    return build_cig(extracted, "test")


def _load_hfm_train_module():
    script = ROOT / "models/mf-generators/hfm_3d/train.py"
    spec = importlib.util.spec_from_file_location("hfm_3d_train", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_hfm_geometry_decoder_train_module():
    script = ROOT / "models/mf-generators/hfm_3d/train_geometry_decoder.py"
    spec = importlib.util.spec_from_file_location("hfm_3d_train_geometry_decoder", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_fragfm_train_module():
    script = ROOT / "models/mf-generators/fragfm/train.py"
    spec = importlib.util.spec_from_file_location("fragfm_train", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_hfm_decoder_sdf(smiles: str, coordinates: list[list[float]]) -> str:
    from rdkit.Geometry import Point3D

    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    assert mol.GetNumAtoms() == len(coordinates)
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for atom_idx, (x_coord, y_coord, z_coord) in enumerate(coordinates):
        conformer.SetAtomPosition(
            atom_idx,
            Point3D(float(x_coord), float(y_coord), float(z_coord)),
        )
    mol.AddConformer(conformer, assignId=True)
    return Chem.MolToMolBlock(mol)


def _write_hfm_geometry_decoder_source_artifact(tmp_path: Path) -> Path:
    artifact_path = tmp_path / "decoder.json"
    artifact_path.write_text(
        json.dumps(
            {
                "humu_checkpoint": "test-humu.pt",
                "entries": [
                    {
                        "id": "ethanol",
                        "smiles": "CCO",
                        "latent": [1.0] + [0.0] * 128,
                        "sdf": _make_hfm_decoder_sdf(
                            "CCO",
                            [
                                [0.0, 0.0, 0.0],
                                [1.5, 0.0, 0.0],
                                [2.1, 0.8, 0.0],
                            ],
                        ),
                    },
                    {
                        "id": "ethylamine",
                        "smiles": "CCN",
                        "latent": [(1.0 + 0.2 * 0.2) ** 0.5, 0.2] + [0.0] * 127,
                        "sdf": _make_hfm_decoder_sdf(
                            "CCN",
                            [
                                [0.0, 0.0, 0.0],
                                [1.4, 0.0, 0.0],
                                [2.0, -0.7, 0.0],
                            ],
                        ),
                    },
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return artifact_path


class TestHFM3DTraining:
    def test_distributed_context_reads_torchrun_environment(self, monkeypatch) -> None:
        module = _load_hfm_train_module()
        monkeypatch.setenv("RANK", "2")
        monkeypatch.setenv("WORLD_SIZE", "4")
        monkeypatch.setenv("LOCAL_RANK", "2")

        context = module._distributed_context_from_env()

        assert context.enabled is True
        assert context.rank == 2
        assert context.world_size == 4
        assert context.local_rank == 2
        assert module._is_main_process(context) is False

    def test_training_cli_writes_checkpoint_artifacts(self, tmp_path) -> None:
        import torch
        from mf_encoders.humu_mol.encoder import HUMUMoleculeEncoder

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "molecules.jsonl").write_text(
            "\n".join(
                [
                    json.dumps({"id": "ethanol", "smiles": "CCO"}),
                    json.dumps({"id": "ethylamine", "smiles": "CCN"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        output_dir = tmp_path / "out"
        humu_checkpoint = tmp_path / "humu.pt"
        encoder = HUMUMoleculeEncoder(dim=128, curvature=1.0)
        torch.save({"encoder_mol": encoder.state_dict()}, humu_checkpoint)
        script = ROOT / "models/mf-generators/hfm_3d/train.py"

        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--data",
                str(data_dir),
                "--output-dir",
                str(output_dir),
                "--humu-checkpoint",
                str(humu_checkpoint),
                "--decoder-artifact",
                str(output_dir / "decoder.json"),
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--device",
                "cpu",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        checkpoint = torch.load(output_dir / "best_model.pt", map_location="cpu", weights_only=True)
        assert {"epoch", "loss", "model", "flow_model", "decoder", "optimizer"} <= set(checkpoint)
        assert checkpoint["humu_checkpoint"] == str(humu_checkpoint)
        assert checkpoint["decoder_artifact"] == str(output_dir / "decoder.json")
        assert (output_dir / "final_model.pt").is_file()
        decoder_payload = json.loads((output_dir / "decoder.json").read_text(encoding="utf-8"))
        assert decoder_payload["humu_checkpoint"] == str(humu_checkpoint)
        assert len(decoder_payload["entries"]) == 2
        assert len(decoder_payload["entries"][0]["latent"]) == 129
        assert "sdf" in decoder_payload["entries"][0]
        assert Chem.MolFromMolBlock(
            decoder_payload["entries"][0]["sdf"],
            sanitize=False,
            removeHs=False,
        ) is not None

    def test_geometry_decoder_training_cli_writes_artifact(self, tmp_path) -> None:
        from mf_generators.hfm_3d.decoder.neural_geometry_decoder import (
            NeuralGeometryDecoderArtifact,
        )

        module = _load_hfm_geometry_decoder_train_module()
        decoder_artifact = _write_hfm_geometry_decoder_source_artifact(tmp_path)
        output_artifact = tmp_path / "geometry_decoder.pt"

        exit_code = module.main(
            [
                "--decoder-artifact",
                str(decoder_artifact),
                "--output-artifact",
                str(output_artifact),
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--device",
                "cpu",
            ]
        )
        artifact = NeuralGeometryDecoderArtifact.load(
            output_artifact,
            map_location="cpu",
        )

        assert exit_code == 0
        assert output_artifact.exists()
        assert artifact.latent_dim == 129
        assert {entry.entry_id for entry in artifact.entries} == {
            "ethanol",
            "ethylamine",
        }

    def test_training_cli_writes_kd_embedding_loss_metadata(self, tmp_path) -> None:
        import torch
        from mf_encoders.humu_mol.encoder import HUMUMoleculeEncoder

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "molecules.jsonl").write_text(
            json.dumps({"id": "ethanol", "smiles": "CCO"}) + "\n",
            encoding="utf-8",
        )
        output_dir = tmp_path / "out"
        humu_checkpoint = tmp_path / "humu.pt"
        encoder = HUMUMoleculeEncoder(dim=128, curvature=1.0)
        torch.save({"encoder_mol": encoder.state_dict()}, humu_checkpoint)
        kd_teacher_embeddings = tmp_path / "teacher_embeddings.json"
        kd_teacher_embeddings.write_text(
            json.dumps({"teacher_embeddings": [[0.0] * 129]}),
            encoding="utf-8",
        )
        script = ROOT / "models/mf-generators/hfm_3d/train.py"

        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--data",
                str(data_dir),
                "--output-dir",
                str(output_dir),
                "--humu-checkpoint",
                str(humu_checkpoint),
                "--kd-teacher-embeddings",
                str(kd_teacher_embeddings),
                "--kd-weight",
                "0.25",
                "--epochs",
                "1",
                "--batch-size",
                "1",
                "--device",
                "cpu",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        checkpoint = torch.load(output_dir / "best_model.pt", map_location="cpu", weights_only=True)
        assert checkpoint["kd_teacher_embeddings"] == str(kd_teacher_embeddings)
        assert checkpoint["kd_weight"] == pytest.approx(0.25)

    def test_training_kd_updates_flow_parameters(self, tmp_path) -> None:
        import torch
        from mf_encoders.humu_mol.encoder import HUMUMoleculeEncoder

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "molecules.jsonl").write_text(
            json.dumps({"id": "ethanol", "smiles": "CCO"}) + "\n",
            encoding="utf-8",
        )
        humu_checkpoint = tmp_path / "humu.pt"
        encoder = HUMUMoleculeEncoder(dim=8, curvature=1.0)
        torch.save({"encoder_mol": encoder.state_dict()}, humu_checkpoint)
        teacher_embeddings = tmp_path / "teacher_embeddings.json"
        teacher_embeddings.write_text(
            json.dumps({"teacher_embeddings": [[0.0] * 9]}),
            encoding="utf-8",
        )
        script = ROOT / "models/mf-generators/hfm_3d/train.py"
        checkpoints = []

        for name, kd_weight in (("baseline", "0.0"), ("distilled", "0.5")):
            output_dir = tmp_path / name
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--data",
                    str(data_dir),
                    "--output-dir",
                    str(output_dir),
                    "--humu-checkpoint",
                    str(humu_checkpoint),
                    "--kd-teacher-embeddings",
                    str(teacher_embeddings),
                    "--kd-weight",
                    kd_weight,
                    "--dim",
                    "8",
                    "--epochs",
                    "1",
                    "--batch-size",
                    "1",
                    "--device",
                    "cpu",
                    "--seed",
                    "17",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, result.stderr
            checkpoints.append(
                torch.load(
                    output_dir / "final_model.pt",
                    map_location="cpu",
                    weights_only=True,
                )["flow_model"]
            )

        baseline_state, distilled_state = checkpoints
        assert any(
            not torch.equal(baseline_state[name], distilled_state[name])
            for name in baseline_state
        )

    def test_training_cli_requires_humu_checkpoint(self, tmp_path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "molecules.jsonl").write_text(
            json.dumps({"id": "ethanol", "smiles": "CCO"}) + "\n",
            encoding="utf-8",
        )
        output_dir = tmp_path / "out"
        script = ROOT / "models/mf-generators/hfm_3d/train.py"

        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--data",
                str(data_dir),
                "--output-dir",
                str(output_dir),
                "--epochs",
                "1",
                "--batch-size",
                "1",
                "--device",
                "cpu",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert "HUMU checkpoint is required" in result.stderr

    def test_load_humu_checkpoint_accepts_inner_prefix(self, tmp_path) -> None:
        import torch
        from mf_encoders.humu_mol.encoder import HUMUMoleculeEncoder

        module = _load_hfm_train_module()
        source_encoder = HUMUMoleculeEncoder(dim=8, curvature=1.0)
        checkpoint = tmp_path / "humu.pt"
        torch.save(
            {
                "encoder_mol": {
                    f"inner.{key}": value
                    for key, value in source_encoder.state_dict().items()
                }
                | {"proj.weight": torch.randn(8, 8)}
            },
            checkpoint,
        )
        target_encoder = HUMUMoleculeEncoder(dim=8, curvature=1.0)

        module._load_humu_molecule_encoder_checkpoint(
            target_encoder,
            str(checkpoint),
            torch.device("cpu"),
        )

        for key, value in source_encoder.state_dict().items():
            assert torch.equal(value, target_encoder.state_dict()[key])

    def test_training_cli_rejects_empty_data_dir(self, tmp_path) -> None:
        import torch
        from mf_encoders.humu_mol.encoder import HUMUMoleculeEncoder

        data_dir = tmp_path / "empty"
        data_dir.mkdir()
        output_dir = tmp_path / "out"
        humu_checkpoint = tmp_path / "humu.pt"
        encoder = HUMUMoleculeEncoder(dim=128, curvature=1.0)
        torch.save({"encoder_mol": encoder.state_dict()}, humu_checkpoint)
        script = ROOT / "models/mf-generators/hfm_3d/train.py"

        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--data",
                str(data_dir),
                "--output-dir",
                str(output_dir),
                "--humu-checkpoint",
                str(humu_checkpoint),
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--device",
                "cpu",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert "contains no HFM-3D training records" in result.stderr

    def test_load_molecules_prefers_manifest_shards(self, tmp_path) -> None:
        module = _load_hfm_train_module()
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "manifest.json").write_text(
            json.dumps({"shards": ["shard_0000.jsonl"]}),
            encoding="utf-8",
        )
        (data_dir / "shard_0000.jsonl").write_text(
            json.dumps({"id": "ethanol", "smiles": "CCO"}) + "\n",
            encoding="utf-8",
        )
        (data_dir / "rejects.jsonl").write_text(
            json.dumps({"id": "rejected", "smiles": "CCN"}) + "\n",
            encoding="utf-8",
        )

        samples = module._load_molecules(str(data_dir))

        assert samples == [{"id": "ethanol", "smiles": "CCO"}]

    def test_lorentz_prior_keeps_flow_loss_finite(self) -> None:
        import torch

        module = _load_hfm_train_module()
        from mf_generators.hfm_3d.model.lorentz_flow_matching import LorentzFlowMatching

        flow = LorentzFlowMatching(dim=8, curvature=1.0, n_steps=2)
        manifold = flow.manifold
        x0 = module._sample_lorentz_prior(manifold, batch_size=4, dim=8, device=torch.device("cpu"))
        origin = manifold.origin(8).expand(4, -1)
        tangent = torch.zeros(4, 9)
        tangent[..., 1:] = torch.randn(4, 8) * 0.05
        x1 = manifold.expmap(origin, tangent)

        loss = flow(x0, x1)

        assert torch.isfinite(x0).all()
        assert torch.isfinite(loss)

    def test_lorentz_flow_vector_field_is_tangent(self) -> None:
        import torch
        from mf_generators.hfm_3d.model.lorentz_flow_matching import LorentzFlowMatching

        flow = LorentzFlowMatching(dim=8, curvature=1.0, n_steps=2)
        manifold = flow.manifold
        origin = manifold.origin(8).expand(4, -1)
        tangent = torch.zeros(4, 9)
        tangent[..., 1:] = torch.randn(4, 8) * 0.05
        x_t = manifold.expmap(origin, tangent)
        t = torch.full((4, 1), 0.5)

        v = flow.compute_vector_field(x_t, t)

        assert torch.allclose(manifold.inner(x_t, v), torch.zeros(4, 1), atol=1e-5)


class TestHFM3DGenerator:
    def test_neural_geometry_decoder_loads_sdf_training_examples(self, tmp_path) -> None:
        from mf_generators.hfm_3d.decoder.neural_geometry_decoder import (
            load_geometry_training_examples,
        )

        decoder_artifact = _write_hfm_geometry_decoder_source_artifact(tmp_path)

        examples = load_geometry_training_examples(decoder_artifact)

        assert len(examples) == 2
        assert examples[0].entry_id == "ethanol"
        assert examples[0].smiles == "CCO"
        assert examples[0].latent.shape == (129,)
        assert examples[0].atom_types == ["C", "C", "O"]
        assert examples[0].coordinates.shape == (3, 3)

    def test_neural_geometry_decoder_rejects_invalid_lorentz_latent(
        self,
        tmp_path,
    ) -> None:
        from mf_generators.hfm_3d.decoder.neural_geometry_decoder import (
            load_geometry_training_examples,
        )

        decoder_artifact = _write_hfm_geometry_decoder_source_artifact(tmp_path)
        payload = json.loads(decoder_artifact.read_text(encoding="utf-8"))
        payload["entries"][0]["latent"] = [0.0] * 129
        decoder_artifact.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="Lorentz"):
            load_geometry_training_examples(decoder_artifact)

    def test_neural_geometry_decoder_trains_tiny_artifact(self, tmp_path) -> None:
        import torch
        from mf_generators.hfm_3d.decoder.neural_geometry_decoder import (
            NeuralGeometryDecoderArtifact,
            train_geometry_decoder_artifact,
        )

        decoder_artifact = _write_hfm_geometry_decoder_source_artifact(tmp_path)
        output_artifact = tmp_path / "geometry_decoder.pt"

        train_geometry_decoder_artifact(
            decoder_artifact,
            output_artifact,
            epochs=1,
            batch_size=2,
            device="cpu",
        )
        artifact = NeuralGeometryDecoderArtifact.load(
            output_artifact,
            map_location="cpu",
        )
        coordinates = artifact.predict_coordinates(
            torch.tensor([1.0] + [0.0] * 128, dtype=torch.float32),
            atom_count=3,
        )

        assert output_artifact.exists()
        assert artifact.latent_dim == 129
        assert coordinates.shape == (3, 3)
        assert torch.isfinite(coordinates).all()

    def test_neural_geometry_decoder_runner_outputs_hfm_contract(self, tmp_path) -> None:
        from mf_generators.hfm_3d.decoder.neural_geometry_decoder import (
            train_geometry_decoder_artifact,
        )

        decoder_artifact = _write_hfm_geometry_decoder_source_artifact(tmp_path)
        output_artifact = tmp_path / "geometry_decoder.pt"
        train_geometry_decoder_artifact(
            decoder_artifact,
            output_artifact,
            epochs=1,
            batch_size=2,
            device="cpu",
        )

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "mf_generators.hfm_3d.decoder.neural_geometry_decoder",
                "--artifact",
                str(output_artifact),
            ],
            input=json.dumps({"latent": [1.0] + [0.0] * 128}),
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
        payload = json.loads(result.stdout)

        assert result.returncode == 0, result.stderr
        assert payload["smiles"] == "CCO"
        assert payload["atom_types"] == ["C", "C", "O"]
        assert len(payload["coordinates"]) == 3
        assert payload["decoder_entry_id"] == "ethanol"
        assert payload["metadata"]["decoder_mode"] == "neural_geometry_decoder"

    def test_hfm_generator_consumes_neural_geometry_decoder_output(self, tmp_path) -> None:
        from mf_generators.hfm_3d.decoder.neural_geometry_decoder import (
            train_geometry_decoder_artifact,
        )
        from mf_generators.hfm_3d.generator import HFM3DGenerator

        decoder_artifact = _write_hfm_geometry_decoder_source_artifact(tmp_path)
        output_artifact = tmp_path / "geometry_decoder.pt"
        neural_decoder = train_geometry_decoder_artifact(
            decoder_artifact,
            output_artifact,
            epochs=1,
            batch_size=2,
            device="cpu",
        )
        checkpoint_path = tmp_path / "hfm.pt"
        HFM3DGenerator(mode="local_demo").save_checkpoint(str(checkpoint_path))
        generator = HFM3DGenerator(
            checkpoint_path=str(checkpoint_path),
            mode="production_real",
            molecular_decoder=neural_decoder,
        )

        molecules = asyncio.run(
            generator.generate(batch_size=1, sampling_seed=7, flow_steps=0)
        )
        decoded = molecules[0]

        assert decoded.smiles in {"CCO", "CCN"}
        assert decoded.sdf_bytes
        assert Chem.MolFromMolBlock(decoded.sdf_bytes.decode("utf-8"), sanitize=False)
        assert decoded.metadata["decoder_entry_id"] in {"ethanol", "ethylamine"}
        assert decoded.metadata["decoder_mode"] == "neural_geometry_decoder"

    @pytest.mark.asyncio
    async def test_bootstraps_atomic_opt_in_validation_artifacts_with_production_probe(
        self,
        tmp_path: Path,
    ) -> None:
        from mf_generators.hfm_3d.generator import (
            HFM3DGenerator,
            bootstrap_validation_artifacts,
        )

        target = tmp_path / "hfm-validation"
        paths = await bootstrap_validation_artifacts(target)
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))

        assert target.is_dir()
        assert metadata["schema_version"] == "moleculeforge.validation_artifact.v1"
        assert metadata["purpose"] == "synthetic_pipeline_validation_only"
        assert metadata["generator"] == "hfm_3d"
        assert metadata["seed"] == 7
        generator = HFM3DGenerator(
            checkpoint_path=str(paths["checkpoint"]),
            decoder_path=str(paths["decoder"]),
            mode="production_real",
        )
        molecules = await generator.generate(
            batch_size=2,
            sampling_seed=7,
            flow_steps=1,
        )
        assert len(molecules) == 2
        assert all(Chem.MolFromSmiles(molecule.smiles) is not None for molecule in molecules)

        occupied = tmp_path / "occupied"
        occupied.mkdir()
        sentinel = occupied / "real-checkpoint.pt"
        sentinel.write_bytes(b"real")
        with pytest.raises(RuntimeError, match="refuses to overwrite"):
            await bootstrap_validation_artifacts(occupied)
        assert sentinel.read_bytes() == b"real"

    def test_hfm_checkpoint_rejects_incomplete_model_state(self, tmp_path) -> None:
        import torch
        from mf_generators.hfm_3d.generator import HFM3DGenerator

        checkpoint_path = tmp_path / "incomplete-hfm.pt"
        HFM3DGenerator(mode="local_demo").save_checkpoint(str(checkpoint_path))
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state["model"].pop(next(iter(state["model"])))
        torch.save(state, checkpoint_path)

        with pytest.raises(RuntimeError, match="Missing key"):
            HFM3DGenerator(
                checkpoint_path=str(checkpoint_path),
                decoder_path="",
                mode="production_real",
                molecular_decoder=lambda embedding: {"smiles": "CCO"},
            )

    def test_external_molecular_decoder_preflight_rejects_missing_executable(self) -> None:
        import torch
        from mf_generators.hfm_3d.generator import ExternalMolecularDecoder

        decoder = ExternalMolecularDecoder("missing-hfm-model-decoder --json")

        with pytest.raises(RuntimeError, match="not found"):
            decoder.decode(torch.zeros(129))

    def test_external_molecular_decoder_command_decodes_latent(self, tmp_path) -> None:
        from mf_generators.hfm_3d.generator import (
            ExternalMolecularDecoder,
            HFM3DGenerator,
        )

        checkpoint_path = tmp_path / "hfm.pt"
        HFM3DGenerator(mode="local_demo").save_checkpoint(str(checkpoint_path))
        runner = tmp_path / "hfm_decoder.py"
        runner.write_text(
            "\n".join(
                [
                    "import json",
                    "import sys",
                    "request = json.load(sys.stdin)",
                    "assert len(request['latent']) == 129",
                    "json.dump(",
                    "    {",
                    "        'smiles': 'CCO',",
                    "        'metadata': {'decoder_entry_id': 'runner_decoder'},",
                    "    },",
                    "    sys.stdout,",
                    ")",
                ]
            ),
            encoding="utf-8",
        )
        generator = HFM3DGenerator(
            checkpoint_path=str(checkpoint_path),
            mode="production_real",
            molecular_decoder=ExternalMolecularDecoder(f"{sys.executable} {runner}"),
        )

        molecules = asyncio.run(
            generator.generate(batch_size=1, sampling_seed=7, flow_steps=0)
        )

        assert molecules[0].smiles == "CCO"
        assert molecules[0].sdf_bytes
        assert molecules[0].metadata["decoder_mode"] == "molecular_decoder"
        assert molecules[0].metadata["decoder_entry_id"] == "runner_decoder"

    def test_generation_rejects_nonfinite_flow_latent(self, tmp_path) -> None:
        import torch
        from mf_generators.hfm_3d.generator import HFM3DGenerator

        class NonFiniteFlow:
            n_steps = 1

            def compute_vector_field(
                self,
                latent_points: torch.Tensor,
                t: torch.Tensor,
            ) -> torch.Tensor:
                return torch.full_like(latent_points, float("nan"))

        checkpoint_path = tmp_path / "hfm.pt"
        HFM3DGenerator(mode="local_demo").save_checkpoint(str(checkpoint_path))

        generator = HFM3DGenerator(
            checkpoint_path=str(checkpoint_path),
            mode="production_real",
            molecular_decoder=lambda embedding: {"smiles": "CCO"},
        )
        generator._model = NonFiniteFlow()

        with pytest.raises(RuntimeError, match="non-finite latent"):
            asyncio.run(generator.generate(batch_size=1, sampling_seed=7))

    def test_route_humu_feedback_steers_latent_toward_route_embedding(
        self,
        tmp_path,
    ) -> None:
        import torch
        from mf_generators.hfm_3d.generator import HFM3DGenerator
        from mf_humu.manifold.lorentz import LorentzManifold

        checkpoint_path = tmp_path / "hfm.pt"
        HFM3DGenerator(mode="local_demo").save_checkpoint(str(checkpoint_path))
        generator = HFM3DGenerator(
            checkpoint_path=str(checkpoint_path),
            mode="production_real",
            molecular_decoder=lambda embedding: {"smiles": "CCO"},
        )
        manifold = LorentzManifold(curvature=1.0)
        target = torch.zeros(129)
        target[1] = 0.35
        target = manifold._project(target)
        feedback = [
            {
                "route_id": "route-1",
                "curvature": 1.0,
                "humu_embedding": target.tolist(),
            }
        ]

        baseline = asyncio.run(
            generator.generate(batch_size=1, sampling_seed=7, flow_steps=0)
        )
        steered = asyncio.run(
            generator.generate(
                batch_size=1,
                sampling_seed=7,
                flow_steps=0,
                route_humu_feedback=json.dumps(feedback),
                feedback_steering_weight=0.25,
            )
        )

        baseline_latent = torch.tensor(json.loads(baseline[0].metadata["latent"]))
        steered_latent = torch.tensor(json.loads(steered[0].metadata["latent"]))

        assert manifold.distance(
            steered_latent.unsqueeze(0),
            target.unsqueeze(0),
        ).item() < manifold.distance(
            baseline_latent.unsqueeze(0),
            target.unsqueeze(0),
        ).item()
        assert steered[0].metadata["feedback_steering_sources"] == "route-1"
        assert steered[0].metadata["feedback_steering_count"] == "1"

    def test_jmcg_feedback_envelope_steers_latent_toward_route_embedding(
        self,
        tmp_path,
    ) -> None:
        import torch
        from mf_generators.hfm_3d.generator import HFM3DGenerator
        from mf_humu.manifold.lorentz import LorentzManifold

        checkpoint_path = tmp_path / "hfm.pt"
        HFM3DGenerator(mode="local_demo").save_checkpoint(str(checkpoint_path))
        generator = HFM3DGenerator(
            checkpoint_path=str(checkpoint_path),
            mode="production_real",
            molecular_decoder=lambda embedding: {"smiles": "CCO"},
        )
        manifold = LorentzManifold(curvature=1.0)
        target = torch.zeros(129)
        target[2] = 0.25
        target = manifold._project(target)
        feedback = {
            "schema": "moleculeforge.jmcg.feedback.v1",
            "run_id": "run-1",
            "project_id": "project-1",
            "records": [
                {
                    "kind": "route",
                    "source": "generator_coord",
                    "run_id": "run-1",
                    "subject": {"type": "route", "id": "route-1"},
                    "humu_embedding": target.tolist(),
                    "curvature": 1.0,
                    "weight": 0.8,
                    "polarity": "attract",
                    "confidence": 0.5,
                    "evidence_ids": ["belief-1"],
                }
            ],
        }

        baseline = asyncio.run(
            generator.generate(batch_size=1, sampling_seed=11, flow_steps=0)
        )
        steered = asyncio.run(
            generator.generate(
                batch_size=1,
                sampling_seed=11,
                flow_steps=0,
                jmcg_feedback=json.dumps(feedback),
                feedback_steering_weight=0.25,
            )
        )

        baseline_latent = torch.tensor(json.loads(baseline[0].metadata["latent"]))
        steered_latent = torch.tensor(json.loads(steered[0].metadata["latent"]))

        assert manifold.distance(
            steered_latent.unsqueeze(0),
            target.unsqueeze(0),
        ).item() < manifold.distance(
            baseline_latent.unsqueeze(0),
            target.unsqueeze(0),
        ).item()
        assert steered[0].metadata["feedback_steering_count"] == "1"
        assert steered[0].metadata["feedback_steering_sources"] == "route-1"
        assert steered[0].metadata["feedback_steering_kinds"] == "route"

    def test_jmcg_feedback_zero_effective_weight_does_not_steer(
        self,
        tmp_path,
    ) -> None:
        import torch
        from mf_generators.hfm_3d.generator import HFM3DGenerator
        from mf_humu.manifold.lorentz import LorentzManifold

        checkpoint_path = tmp_path / "hfm.pt"
        HFM3DGenerator(mode="local_demo").save_checkpoint(str(checkpoint_path))
        generator = HFM3DGenerator(
            checkpoint_path=str(checkpoint_path),
            mode="production_real",
            molecular_decoder=lambda embedding: {"smiles": "CCO"},
        )
        manifold = LorentzManifold(curvature=1.0)
        target = torch.zeros(129)
        target[3] = 0.25
        target = manifold._project(target)
        feedback = {
            "schema": "moleculeforge.jmcg.feedback.v1",
            "records": [
                {
                    "kind": "route",
                    "subject": {"type": "route", "id": "route-zero"},
                    "humu_embedding": target.tolist(),
                    "weight": 1.0,
                    "confidence": 0.0,
                    "polarity": "attract",
                }
            ],
        }

        baseline = asyncio.run(
            generator.generate(batch_size=1, sampling_seed=13, flow_steps=0)
        )
        steered = asyncio.run(
            generator.generate(
                batch_size=1,
                sampling_seed=13,
                flow_steps=0,
                jmcg_feedback=json.dumps(feedback),
                feedback_steering_weight=0.25,
            )
        )

        assert json.loads(steered[0].metadata["latent"]) == json.loads(
            baseline[0].metadata["latent"]
        )
        assert "feedback_steering_count" not in steered[0].metadata

    def test_jmcg_property_feedback_without_embedding_does_not_steer(
        self,
        tmp_path,
    ) -> None:
        from mf_generators.hfm_3d.generator import HFM3DGenerator

        checkpoint_path = tmp_path / "hfm.pt"
        HFM3DGenerator(mode="local_demo").save_checkpoint(str(checkpoint_path))
        generator = HFM3DGenerator(
            checkpoint_path=str(checkpoint_path),
            mode="production_real",
            molecular_decoder=lambda embedding: {"smiles": "CCO"},
        )
        feedback = {
            "schema": "moleculeforge.jmcg.feedback.v1",
            "run_id": "run-1",
            "records": [
                {
                    "kind": "property",
                    "source": "validation",
                    "run_id": "run-1",
                    "subject": {"type": "workflow_feedback", "id": "validation-0"},
                    "weight": 1.0,
                    "polarity": "repel",
                    "confidence": 1.0,
                    "evidence_ids": [],
                    "metadata": {"reason": "affinity gate failed"},
                }
            ],
        }

        baseline = asyncio.run(
            generator.generate(batch_size=1, sampling_seed=29, flow_steps=0)
        )
        steered = asyncio.run(
            generator.generate(
                batch_size=1,
                sampling_seed=29,
                flow_steps=0,
                jmcg_feedback=json.dumps(feedback),
                feedback_steering_weight=0.25,
            )
        )

        assert json.loads(steered[0].metadata["latent"]) == json.loads(
            baseline[0].metadata["latent"]
        )
        assert "feedback_steering_count" not in steered[0].metadata

    def test_jmcg_intent_and_pocket_feedback_without_embedding_does_not_steer(
        self,
        tmp_path,
    ) -> None:
        from mf_generators.hfm_3d.generator import HFM3DGenerator

        checkpoint_path = tmp_path / "hfm.pt"
        HFM3DGenerator(mode="local_demo").save_checkpoint(str(checkpoint_path))
        generator = HFM3DGenerator(
            checkpoint_path=str(checkpoint_path),
            mode="production_real",
            molecular_decoder=lambda embedding: {"smiles": "CCO"},
        )
        feedback = {
            "schema": "moleculeforge.jmcg.feedback.v1",
            "run_id": "run-1",
            "records": [
                {
                    "kind": "intent",
                    "source": "orchestrator_svc",
                    "run_id": "run-1",
                    "subject": {"type": "intent", "id": "run-1"},
                    "weight": 1.0,
                    "polarity": "attract",
                    "confidence": 1.0,
                    "evidence_ids": [],
                    "metadata": {"has_hciv": True},
                },
                {
                    "kind": "pocket",
                    "source": "orchestrator_svc",
                    "run_id": "run-1",
                    "subject": {"type": "pocket", "id": "switch-ii"},
                    "weight": 1.0,
                    "polarity": "attract",
                    "confidence": 1.0,
                    "evidence_ids": [],
                    "metadata": {"pocket_id": "switch-ii"},
                },
            ],
        }

        baseline = asyncio.run(
            generator.generate(batch_size=1, sampling_seed=31, flow_steps=0)
        )
        steered = asyncio.run(
            generator.generate(
                batch_size=1,
                sampling_seed=31,
                flow_steps=0,
                jmcg_feedback=json.dumps(feedback),
                feedback_steering_weight=0.25,
            )
        )

        assert json.loads(steered[0].metadata["latent"]) == json.loads(
            baseline[0].metadata["latent"]
        )
        assert "feedback_steering_count" not in steered[0].metadata

    def test_jmcg_feedback_drops_invalid_embedding_dimensions(
        self,
        tmp_path,
    ) -> None:
        from mf_generators.hfm_3d.generator import HFM3DGenerator

        checkpoint_path = tmp_path / "hfm.pt"
        HFM3DGenerator(mode="local_demo").save_checkpoint(str(checkpoint_path))
        generator = HFM3DGenerator(
            checkpoint_path=str(checkpoint_path),
            mode="production_real",
            molecular_decoder=lambda embedding: {"smiles": "CCO"},
        )
        valid_embedding = [0.0] * 129
        valid_embedding[0] = 1.0
        feedback = {
            "schema": "moleculeforge.jmcg.feedback.v1",
            "records": [
                {
                    "kind": "route",
                    "subject": {"type": "route", "id": "bad-route"},
                    "humu_embedding": [1.0, 0.0],
                    "weight": 1.0,
                    "confidence": 1.0,
                    "polarity": "attract",
                },
                {
                    "kind": "molecule",
                    "subject": {"type": "molecule", "id": "candidate-1"},
                    "humu_embedding": valid_embedding,
                    "weight": 1.0,
                    "confidence": 1.0,
                    "polarity": "attract",
                },
            ],
        }

        steered = asyncio.run(
            generator.generate(
                batch_size=1,
                sampling_seed=17,
                flow_steps=0,
                jmcg_feedback=json.dumps(feedback),
                feedback_steering_weight=0.25,
            )
        )

        assert steered[0].metadata["feedback_steering_count"] == "1"
        assert steered[0].metadata["feedback_steering_dropped_count"] == "1"
        assert steered[0].metadata["feedback_steering_sources"] == "candidate-1"
        assert steered[0].metadata["feedback_steering_kinds"] == "molecule"

    def test_jmcg_feedback_drops_invalid_lorentz_embeddings(
        self,
        tmp_path,
    ) -> None:
        from mf_generators.hfm_3d.generator import HFM3DGenerator

        checkpoint_path = tmp_path / "hfm.pt"
        HFM3DGenerator(mode="local_demo").save_checkpoint(str(checkpoint_path))
        generator = HFM3DGenerator(
            checkpoint_path=str(checkpoint_path),
            mode="production_real",
            molecular_decoder=lambda embedding: {"smiles": "CCO"},
        )
        feedback = {
            "schema": "moleculeforge.jmcg.feedback.v1",
            "records": [
                {
                    "kind": "route",
                    "subject": {"type": "route", "id": "bad-route"},
                    "humu_embedding": [0.0] * 129,
                    "weight": 1.0,
                    "confidence": 1.0,
                    "polarity": "attract",
                }
            ],
        }

        steered = asyncio.run(
            generator.generate(
                batch_size=1,
                sampling_seed=17,
                flow_steps=0,
                jmcg_feedback=json.dumps(feedback),
                feedback_steering_weight=0.25,
            )
        )

        assert "feedback_steering_count" not in steered[0].metadata

    def test_jmcg_feedback_repel_moves_latent_away_from_embedding(
        self,
        tmp_path,
    ) -> None:
        import torch
        from mf_generators.hfm_3d.generator import HFM3DGenerator

        checkpoint_path = tmp_path / "hfm.pt"
        HFM3DGenerator(mode="local_demo").save_checkpoint(str(checkpoint_path))
        generator = HFM3DGenerator(
            checkpoint_path=str(checkpoint_path),
            mode="production_real",
            molecular_decoder=lambda embedding: {"smiles": "CCO"},
        )
        baseline = asyncio.run(
            generator.generate(batch_size=1, sampling_seed=19, flow_steps=0)
        )
        baseline_latent = torch.tensor(json.loads(baseline[0].metadata["latent"]))
        feedback = {
            "schema": "moleculeforge.jmcg.feedback.v1",
            "records": [
                {
                    "kind": "molecule",
                    "subject": {"type": "molecule", "id": "reject-1"},
                    "humu_embedding": baseline_latent.tolist(),
                    "weight": 1.0,
                    "confidence": 1.0,
                    "polarity": "repel",
                }
            ],
        }

        steered = asyncio.run(
            generator.generate(
                batch_size=1,
                sampling_seed=19,
                flow_steps=0,
                jmcg_feedback=json.dumps(feedback),
                feedback_steering_weight=0.25,
                feedback_steering_max_step=0.1,
            )
        )
        steered_latent = torch.tensor(json.loads(steered[0].metadata["latent"]))

        assert torch.linalg.vector_norm(steered_latent - baseline_latent).item() > 0.0
        assert steered[0].metadata["feedback_steering_kinds"] == "molecule"

    def test_jmcg_feedback_aggregates_by_kind_before_global_target(
        self,
        tmp_path,
    ) -> None:
        import torch
        from mf_generators.hfm_3d.generator import HFM3DGenerator
        from mf_humu.manifold.lorentz import LorentzManifold

        checkpoint_path = tmp_path / "hfm.pt"
        HFM3DGenerator(mode="local_demo").save_checkpoint(str(checkpoint_path))
        generator = HFM3DGenerator(
            checkpoint_path=str(checkpoint_path),
            mode="production_real",
            molecular_decoder=lambda embedding: {"smiles": "CCO"},
        )
        manifold = LorentzManifold(curvature=1.0)
        route_target = torch.zeros(129)
        route_target[4] = 0.35
        route_target = manifold._project(route_target)
        molecule_target = torch.zeros(129)
        molecule_target[5] = 0.35
        molecule_target = manifold._project(molecule_target)
        feedback = {
            "schema": "moleculeforge.jmcg.feedback.v1",
            "records": [
                {
                    "kind": "route",
                    "subject": {"type": "route", "id": f"route-{idx}"},
                    "humu_embedding": route_target.tolist(),
                    "weight": 1.0,
                    "confidence": 1.0,
                    "polarity": "attract",
                }
                for idx in range(4)
            ]
            + [
                {
                    "kind": "molecule",
                    "subject": {"type": "molecule", "id": "candidate-1"},
                    "humu_embedding": molecule_target.tolist(),
                    "weight": 1.0,
                    "confidence": 1.0,
                    "polarity": "attract",
                }
            ],
        }

        steered = asyncio.run(
            generator.generate(
                batch_size=1,
                sampling_seed=23,
                flow_steps=0,
                jmcg_feedback=json.dumps(feedback),
                feedback_steering_weight=0.25,
            )
        )

        assert steered[0].metadata["feedback_steering_count"] == "5"
        assert steered[0].metadata["feedback_steering_kind_count"] == "2"
        assert steered[0].metadata["feedback_steering_kinds"] == "molecule,route"

    def test_jmcg_engineering_sampler_builds_joint_sample(self) -> None:
        import json

        import torch
        from mf_generators.hfm_3d.inference import JMCGEngineeringSampler
        from mf_humu.manifold.lorentz import LorentzManifold

        manifold = LorentzManifold(curvature=1.0)
        embedding = torch.zeros(129)
        embedding[1] = 0.25
        embedding = manifold._project(embedding).tolist()
        feedback = {
            "schema": "moleculeforge.jmcg.feedback.v1",
            "records": [
                {
                    "kind": "route",
                    "source": "generator_coord",
                    "subject": {"type": "route", "id": "route-1"},
                    "humu_embedding": embedding,
                    "metadata": {"reactions": ["CCBr>>CCO"]},
                },
                {
                    "kind": "pocket",
                    "source": "orchestrator_svc",
                    "subject": {"type": "pocket", "id": "switch-ii"},
                    "humu_embedding": embedding,
                },
                {
                    "kind": "intent",
                    "source": "orchestrator_svc",
                    "subject": {"type": "intent", "id": "intent-1"},
                    "humu_embedding": embedding,
                },
            ],
        }

        samples = JMCGEngineeringSampler().sample(
            [{"smiles": "CCO", "humu_embedding": embedding}],
            property_profile={"targets": {"qed": 0.8}},
            jmcg_feedback=json.dumps(feedback),
        )

        assert len(samples) == 1
        payload = samples[0].to_dict()
        json.dumps(payload, sort_keys=True)
        assert payload["metadata"]["mode"] == "engineering_skeleton"
        assert payload["route"]["route_id"] == "route-1"
        assert payload["pocket"]["subject"]["id"] == "switch-ii"
        assert payload["intent"]["subject"]["id"] == "intent-1"
        assert payload["metadata"]["alignment_pair_count"] == 3
        assert payload["joint_score"] > 0.0

    def test_jmcg_engineering_sampler_keeps_invalid_dimensions_non_steering(self) -> None:
        from mf_generators.hfm_3d.inference import JMCGEngineeringSampler

        bad_embedding = [0.0] * 128
        feedback = {
            "schema": "moleculeforge.jmcg.feedback.v1",
            "records": [
                {
                    "kind": "route",
                    "subject": {"type": "route", "id": "route-128"},
                    "humu_embedding": bad_embedding,
                }
            ],
        }

        sample = JMCGEngineeringSampler().sample(
            [{"smiles": "CCO", "humu_embedding": bad_embedding}],
            jmcg_feedback=feedback,
        )[0].to_dict()

        assert sample["joint_score"] == 0.0
        assert sample["metadata"]["alignment_pair_count"] == 0
        assert sample["metadata"]["ignored_embedding_count"] == 2
        assert sample["metadata"]["non_steering_context"] is True

    def test_jmcg_engineering_sampler_rejects_invalid_lorentz_embeddings(self) -> None:
        from mf_generators.hfm_3d.inference import JMCGEngineeringSampler

        valid_embedding = [1.0] + [0.0] * 128
        invalid_embedding = [0.0] * 129
        feedback = {
            "schema": "moleculeforge.jmcg.feedback.v1",
            "records": [
                {
                    "kind": "route",
                    "subject": {"type": "route", "id": "bad-route"},
                    "humu_embedding": invalid_embedding,
                }
            ],
        }

        sample = JMCGEngineeringSampler().sample(
            [{"smiles": "CCO", "humu_embedding": valid_embedding}],
            jmcg_feedback=feedback,
        )[0].to_dict()

        assert sample["joint_score"] == 0.0
        assert sample["metadata"]["alignment_pair_count"] == 0
        assert sample["metadata"]["ignored_embedding_count"] == 1
        assert sample["metadata"]["non_steering_context"] is True

    def test_jmcg_engineering_sampler_decodes_packed_float32_molecule_embedding(self) -> None:
        from mf_core.types.molecule import Molecule
        from mf_generators.hfm_3d.inference import JMCGEngineeringSampler

        embedding = [1.0] + [0.0] * 128
        packed_embedding = struct.pack("<129f", *embedding)
        feedback = {
            "schema": "moleculeforge.jmcg.feedback.v1",
            "records": [
                {
                    "kind": "route",
                    "subject": {"type": "route", "id": "route-1"},
                    "humu_embedding": embedding,
                }
            ],
        }

        sample = JMCGEngineeringSampler().sample(
            [Molecule(smiles="CCO", humu_embedding=packed_embedding)],
            jmcg_feedback=feedback,
        )[0].to_dict()

        assert sample["joint_score"] > 0.0
        assert sample["metadata"]["alignment_pair_count"] == 1
        assert sample["metadata"]["ignored_embedding_count"] == 0

    def test_parse_jmcg_context_preserves_property_and_route_records(self) -> None:
        import json

        from mf_generators.hfm_3d.inference import parse_jmcg_context

        payload = {
            "schema": "moleculeforge.jmcg.feedback.v1",
            "records": [
                {
                    "kind": "property",
                    "source": "validation",
                    "subject": {"type": "workflow_feedback", "id": "prop-1"},
                    "metadata": {"reason": "affinity gate failed"},
                },
                {
                    "kind": "route",
                    "source": "generator_coord",
                    "subject": {"type": "route", "id": "route-1"},
                    "humu_embedding": [1.0] + [0.0] * 128,
                },
            ],
        }

        records = parse_jmcg_context(json.dumps(payload))

        assert [record.kind for record in records] == ["property", "route"]
        assert records[0].humu_embedding is None
        assert records[0].metadata == {"reason": "affinity gate failed"}
        assert len(records[1].humu_embedding) == 129


class TestMMPTRAGGenerator:
    def test_external_patent_rag_preflight_rejects_missing_executable(self) -> None:
        from mf_generators.mmpt_rag.generator import ExternalPatentRAGRetriever

        retriever = ExternalPatentRAGRetriever("missing-mmpt-model-rag --json")

        with pytest.raises(RuntimeError, match="not found"):
            asyncio.run(retriever.retrieve({"seed_smiles": ["CCO"]}))

    def test_external_seq2seq_decoder_preflight_rejects_missing_executable(self) -> None:
        from mf_generators.mmpt_rag.generator import ExternalSeq2SeqDecoder

        decoder = ExternalSeq2SeqDecoder("missing-mmpt-model-decoder --json")

        with pytest.raises(RuntimeError, match="not found"):
            asyncio.run(decoder.decode({"seed_smiles": "CCO"}))

    def test_generates_valid_smiles(self) -> None:
        from mf_generators.mmpt_rag.generator import MMPTRAGGenerator

        gen = MMPTRAGGenerator()
        hciv, cone = _make_hciv_cone()
        cig = _make_cig()

        mols = _run_async_iter(gen.generate(hciv, cone, cig, n_samples=3, seed=42))
        assert len(mols) == 3
        assert len({Chem.MolToSmiles(Chem.MolFromSmiles(mol.smiles)) for mol in mols}) == 3
        for mol in mols:
            assert mol.smiles
            rdkit_mol = Chem.MolFromSmiles(mol.smiles)
            assert rdkit_mol is not None, f"Invalid SMILES: {mol.smiles}"

    def test_rejects_batch_larger_than_unique_candidate_capacity(
        self,
    ) -> None:
        from mf_generators.mmpt_rag.generator import MMPTRAGGenerator

        generator = MMPTRAGGenerator(
            mmp_database=[
                {
                    "id": "only_transform",
                    "pattern": "F",
                    "replacement": "Cl",
                    "seed_smiles": "c1ccccc1F",
                }
            ]
        )
        hciv, cone = _make_hciv_cone()
        cig = _make_cig()

        with pytest.raises(
            RuntimeError,
            match="produced 1 unique valid candidates, requested 5",
        ):
            _run_async_iter(
                generator.generate(hciv, cone, cig, n_samples=5, seed=42)
            )

    def test_deduplicates_candidates_by_canonical_smiles(self) -> None:
        from mf_generators.mmpt_rag.generator import MMPTRAGGenerator

        generator = MMPTRAGGenerator(
            mmp_database=[
                {
                    "id": "forward_spelling",
                    "pattern": "Xe",
                    "replacement": "Cl",
                    "seed_smiles": "c1ccccc1F",
                    "product_smiles": "c1ccccc1Cl",
                },
                {
                    "id": "reverse_spelling",
                    "pattern": "Kr",
                    "replacement": "Cl",
                    "seed_smiles": "c1ccccc1F",
                    "product_smiles": "Clc1ccccc1",
                },
            ]
        )
        hciv, cone = _make_hciv_cone()
        cig = _make_cig()

        with pytest.raises(
            RuntimeError,
            match="produced 1 unique valid candidates, requested 2",
        ):
            _run_async_iter(
                generator.generate(hciv, cone, cig, n_samples=2, seed=42)
            )

    def test_rdkit_substructure_replace(self) -> None:
        from mf_generators.mmpt_rag.generator import MMPTRAGGenerator

        gen = MMPTRAGGenerator()
        result = gen._simple_replace("c1ccccc1", "c1ccccc1", "c1ccncc1")
        assert result is not None
        mol = Chem.MolFromSmiles(result)
        assert mol is not None

    def test_skips_patent_negative_smiles(self) -> None:
        from mf_generators.mmpt_rag.generator import MMPTRAGGenerator

        gen = MMPTRAGGenerator(
            mmp_database=[
                {"id": "blocked_chloro", "pattern": "F", "replacement": "Cl"},
                {"id": "allowed_amino", "pattern": "F", "replacement": "N"},
            ],
            patent_negative_smiles=[
                "c1ccccc1Cl",
                "CC(=O)Oc1ccccc1Cl",
                "Clc1ccc(N)cc1",
            ],
        )
        hciv, cone = _make_hciv_cone()
        cig = _make_cig()

        mols = _run_async_iter(gen.generate(hciv, cone, cig, n_samples=3, seed=42))

        smiles = {mol.smiles for mol in mols}
        assert "c1ccccc1Cl" not in smiles
        assert "CC(=O)Oc1ccccc1Cl" not in smiles
        assert "Clc1ccc(N)cc1" not in smiles
        assert {mol.properties["transform_id"] for mol in mols} == {"allowed_amino"}

    def test_contrastive_decoding_ranks_positive_evidence_above_patent_neighbor(self) -> None:
        from mf_generators.mmpt_rag.generator import MMPTRAGGenerator

        gen = MMPTRAGGenerator(
            mmp_database=[
                {
                    "id": "near_patent",
                    "pattern": "F",
                    "replacement": "Cl",
                    "negative_smiles": ["CC(=O)Oc1ccccc1Cl"],
                },
                {
                    "id": "positive_neighbor",
                    "pattern": "F",
                    "replacement": "N",
                    "positive_smiles": ["c1ccccc1N"],
                },
            ],
        )
        hciv, cone = _make_hciv_cone()
        cig = _make_cig()

        mols = _run_async_iter(gen.generate(hciv, cone, cig, n_samples=1, seed=42))

        assert mols[0].smiles == "c1ccccc1N"
        assert mols[0].properties["transform_id"] == "positive_neighbor"
        assert mols[0].properties["contrastive_score"] > 0.0

    def test_index_generation_uses_retrieved_seed_evidence(self, tmp_path) -> None:
        from mf_generators.mmpt_rag.generator import MMPTRAGGenerator

        index_path = tmp_path / "mmpt_index.json"
        index_path.write_text(
            json.dumps(
                {
                    "transforms": [
                        {
                            "id": "retrieved_patent_transform",
                            "pattern": "F",
                            "replacement": "N",
                            "seed_smiles": "OCCF",
                            "product_smiles": "OCCN",
                            "retrieval_score": 1.0,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        gen = MMPTRAGGenerator(index_path=str(index_path))
        hciv, cone = _make_hciv_cone()
        cig = _make_cig()

        mols = _run_async_iter(gen.generate(hciv, cone, cig, n_samples=1, seed=42))

        assert mols[0].smiles == "OCCN"
        assert mols[0].properties["source_seed"] == "OCCF"
        assert mols[0].properties["transform_id"] == "retrieved_patent_transform"

    @pytest.mark.asyncio
    async def test_bootstraps_atomic_unique_index_with_production_probe(
        self,
        tmp_path: Path,
    ) -> None:
        from mf_generators.mmpt_rag.generator import (
            MMPTRAGGenerator,
            bootstrap_validation_artifacts,
        )

        target = tmp_path / "mmpt-validation"
        paths = await bootstrap_validation_artifacts(target)
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))

        assert metadata["schema_version"] == "moleculeforge.validation_artifact.v1"
        assert metadata["purpose"] == "synthetic_pipeline_validation_only"
        assert metadata["generator"] == "mmpt_rag"
        assert metadata["seed"] == 7
        generator = MMPTRAGGenerator(index_path=str(paths["index"]))
        molecules = [
            molecule
            async for molecule in generator.generate(
                None,
                None,
                None,
                n_samples=256,
                seed=7,
            )
        ]
        canonical_smiles = [molecule.canonical_smiles for molecule in molecules]
        assert len(molecules) == 256
        assert len(set(canonical_smiles)) == 256
        assert all(Chem.MolFromSmiles(smiles) is not None for smiles in canonical_smiles)

        occupied = tmp_path / "occupied-mmpt"
        occupied.mkdir()
        sentinel = occupied / "mmpt_index.json"
        sentinel.write_text('{"production":true}', encoding="utf-8")
        with pytest.raises(RuntimeError, match="refuses to overwrite"):
            await bootstrap_validation_artifacts(occupied)
        assert sentinel.read_text(encoding="utf-8") == '{"production":true}'

    def test_external_patent_rag_command_supplies_retrieved_transforms(self, tmp_path) -> None:
        from mf_generators.mmpt_rag.generator import (
            ExternalPatentRAGRetriever,
            MMPTRAGGenerator,
        )

        runner = tmp_path / "patent_retriever.py"
        runner.write_text(
            "\n".join(
                [
                    "import json",
                    "import sys",
                    "request = json.load(sys.stdin)",
                    "assert request['n_samples'] == 1",
                    "json.dump(",
                    "    {",
                    "        'transforms': [",
                    "            {",
                    "                'id': 'runner_transform',",
                    "                'pattern': 'F',",
                    "                'replacement': 'I',",
                    "                'seed_smiles': 'OCCF',",
                    "                'product_smiles': 'OCCI',",
                    "                'retrieval_score': 2.0,",
                    "            }",
                    "        ]",
                    "    },",
                    "    sys.stdout,",
                    ")",
                ]
            ),
            encoding="utf-8",
        )
        gen = MMPTRAGGenerator(
            mmp_database=[{"id": "local_transform", "pattern": "F", "replacement": "Cl"}],
            patent_retriever=ExternalPatentRAGRetriever(f"{sys.executable} {runner}"),
        )
        hciv, cone = _make_hciv_cone()
        cig = _make_cig()

        mols = _run_async_iter(gen.generate(hciv, cone, cig, n_samples=1, seed=42))

        assert mols[0].smiles == "OCCI"
        assert mols[0].properties["source_seed"] == "OCCF"
        assert mols[0].properties["transform_id"] == "runner_transform"

    def test_external_seq2seq_decoder_command_decodes_transform(self, tmp_path) -> None:
        from mf_generators.mmpt_rag.generator import (
            ExternalSeq2SeqDecoder,
            MMPTRAGGenerator,
        )

        runner = tmp_path / "seq2seq_decoder.py"
        runner.write_text(
            "\n".join(
                [
                    "import json",
                    "import sys",
                    "request = json.load(sys.stdin)",
                    "assert request['seed_smiles'] == 'OCCF'",
                    "assert request['transform']['id'] == 'seq2seq_transform'",
                    "json.dump({'smiles': 'OCCN'}, sys.stdout)",
                ]
            ),
            encoding="utf-8",
        )
        gen = MMPTRAGGenerator(
            mmp_database=[
                {
                    "id": "seq2seq_transform",
                    "pattern": "F",
                    "replacement": "Cl",
                    "seed_smiles": "OCCF",
                }
            ],
            seq2seq_decoder=ExternalSeq2SeqDecoder(f"{sys.executable} {runner}"),
        )
        hciv, cone = _make_hciv_cone()
        cig = _make_cig()

        mols = _run_async_iter(gen.generate(hciv, cone, cig, n_samples=1, seed=42))

        assert mols[0].smiles == "OCCN"
        assert mols[0].properties["source_seed"] == "OCCF"
        assert mols[0].properties["transform_id"] == "seq2seq_transform"


class TestCReM3DGenerator:
    def test_requires_mmp_database_artifact_in_production(self) -> None:
        from mf_generators.crem_3d.generator import CReM3DGenerator

        with pytest.raises(RuntimeError, match="MMP database artifact"):
            CReM3DGenerator()

    def test_uses_mmp_database_mutations_and_validity_check(self, tmp_path) -> None:
        from mf_generators.crem_3d.generator import CReM3DGenerator

        mmp_path = tmp_path / "crem_mmp.json"
        mmp_path.write_text(
            json.dumps(
                {
                    "mutations": [
                        {
                            "id": "benzene_fluoro",
                            "seed_smiles": "c1ccccc1",
                            "product": "Fc1ccccc1",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        generator = CReM3DGenerator(mmp_db_path=str(mmp_path))

        molecules = asyncio.run(
            generator.generate(batch_size=2, seed_smiles="c1ccccc1")
        )

        assert [mol.smiles for mol in molecules] == ["Fc1ccccc1", "Fc1ccccc1"]
        assert all(Chem.MolFromSmiles(mol.smiles) is not None for mol in molecules)
        assert molecules[0].metadata["generator_name"] == "crem_3d"
        assert molecules[0].metadata["mutation_id"] == "benzene_fluoro"

    @pytest.mark.asyncio
    async def test_bootstraps_atomic_validation_database_with_production_probe(
        self,
        tmp_path: Path,
    ) -> None:
        from mf_generators.crem_3d.generator import (
            CReM3DGenerator,
            bootstrap_validation_artifacts,
        )

        target = tmp_path / "crem-validation"
        paths = await bootstrap_validation_artifacts(target)
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))

        assert metadata["schema_version"] == "moleculeforge.validation_artifact.v1"
        assert metadata["purpose"] == "synthetic_pipeline_validation_only"
        assert metadata["generator"] == "crem_3d"
        assert metadata["seed"] == 7
        generator = CReM3DGenerator(mmp_db_path=str(paths["mmp_database"]))
        molecules = await generator.generate(
            batch_size=3,
            seed_smiles="c1ccccc1",
        )
        assert len(molecules) == 3
        assert all(Chem.MolFromSmiles(molecule.smiles) is not None for molecule in molecules)

        occupied = tmp_path / "occupied-crem"
        occupied.mkdir()
        sentinel = occupied / "real-database.json"
        sentinel.write_text('{"production":true}', encoding="utf-8")
        with pytest.raises(RuntimeError, match="refuses to overwrite"):
            await bootstrap_validation_artifacts(occupied)
        assert sentinel.read_text(encoding="utf-8") == '{"production":true}'

    def test_uses_docking_scorer_to_rank_mutations(self, tmp_path) -> None:
        from mf_generators.crem_3d.generator import CReM3DGenerator

        class DockingScorer:
            def __init__(self) -> None:
                self.calls: list[str] = []

            async def score(self, smiles: str) -> dict:
                self.calls.append(smiles)
                return {
                    "docking_score": -9.0 if smiles == "Clc1ccccc1" else -6.0,
                    "docking_engine": "diffdock_l",
                }

        mmp_path = tmp_path / "crem_mmp.json"
        mmp_path.write_text(
            json.dumps(
                {
                    "mutations": [
                        {
                            "id": "benzene_fluoro",
                            "seed_smiles": "c1ccccc1",
                            "product": "Fc1ccccc1",
                        },
                        {
                            "id": "benzene_chloro",
                            "seed_smiles": "c1ccccc1",
                            "product": "Clc1ccccc1",
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        scorer = DockingScorer()
        generator = CReM3DGenerator(
            mmp_db_path=str(mmp_path),
            docking_scorer=scorer,
        )

        molecules = asyncio.run(
            generator.generate(batch_size=2, seed_smiles="c1ccccc1")
        )

        assert scorer.calls == ["Fc1ccccc1", "Clc1ccccc1"]
        assert [mol.metadata["mutation_id"] for mol in molecules] == [
            "benzene_chloro",
            "benzene_fluoro",
        ]
        assert molecules[0].metadata["docking_engine"] == "diffdock_l"
        assert molecules[0].metadata["docking_score"] == "-9.0"

    def test_uses_dock_oracle_grpc_scorer_batch_to_rank_mutations(self, tmp_path) -> None:
        from mf_core.proto_gen.moleculeforge.v1.oracle import oracle_pb2
        from mf_generators.crem_3d.generator import CReM3DGenerator, DockOracleGrpcScorer

        class DockStub:
            def __init__(self) -> None:
                self.requests = []

            async def Evaluate(self, request):
                self.requests.append(request)
                return oracle_pb2.OracleBatchResponse(
                    evaluations=[
                        oracle_pb2.OracleEvaluation(
                            oracle_name="diffdock_l",
                            molecule_smiles="Fc1ccccc1",
                            level=oracle_pb2.L2_DOCKING,
                            scores={"docking_score": -6.0},
                            success=True,
                        ),
                        oracle_pb2.OracleEvaluation(
                            oracle_name="diffdock_l",
                            molecule_smiles="Clc1ccccc1",
                            level=oracle_pb2.L2_DOCKING,
                            scores={"docking_score": -9.0},
                            success=True,
                        ),
                    ]
                )

        mmp_path = tmp_path / "crem_mmp.json"
        mmp_path.write_text(
            json.dumps(
                {
                    "mutations": [
                        {
                            "id": "benzene_fluoro",
                            "seed_smiles": "c1ccccc1",
                            "product": "Fc1ccccc1",
                        },
                        {
                            "id": "benzene_chloro",
                            "seed_smiles": "c1ccccc1",
                            "product": "Clc1ccccc1",
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        stub = DockStub()
        generator = CReM3DGenerator(
            mmp_db_path=str(mmp_path),
            docking_scorer=DockOracleGrpcScorer(stub=stub),
        )

        molecules = asyncio.run(
            generator.generate(batch_size=2, seed_smiles="c1ccccc1")
        )

        assert len(stub.requests) == 1
        assert stub.requests[0].level == oracle_pb2.L2_DOCKING
        assert list(stub.requests[0].requested_properties) == ["docking_score"]
        assert [mol.metadata["mutation_id"] for mol in molecules] == [
            "benzene_chloro",
            "benzene_fluoro",
        ]
        assert molecules[0].metadata["oracle_name"] == "diffdock_l"
        assert molecules[0].metadata["docking_score"] == "-9.0"

    def test_uses_pharmacophore_scorer_to_rank_mutations(self, tmp_path) -> None:
        from mf_generators.crem_3d.generator import CReM3DGenerator

        class PharmacophoreScorer:
            async def score_batch(self, smiles_list: list[str]) -> dict[str, dict]:
                return {
                    smiles: {
                        "pharmacophore_score": 0.9 if smiles == "Clc1ccccc1" else 0.2,
                        "pharmacophore_model": "unit-pharm3d",
                    }
                    for smiles in smiles_list
                }

        mmp_path = tmp_path / "crem_mmp.json"
        mmp_path.write_text(
            json.dumps(
                {
                    "mutations": [
                        {
                            "id": "benzene_fluoro",
                            "seed_smiles": "c1ccccc1",
                            "product": "Fc1ccccc1",
                        },
                        {
                            "id": "benzene_chloro",
                            "seed_smiles": "c1ccccc1",
                            "product": "Clc1ccccc1",
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        generator = CReM3DGenerator(
            mmp_db_path=str(mmp_path),
            pharmacophore_scorer=PharmacophoreScorer(),
        )

        molecules = asyncio.run(
            generator.generate(batch_size=2, seed_smiles="c1ccccc1")
        )

        assert [mol.metadata["mutation_id"] for mol in molecules] == [
            "benzene_chloro",
            "benzene_fluoro",
        ]
        assert molecules[0].metadata["pharmacophore_model"] == "unit-pharm3d"
        assert molecules[0].metadata["pharmacophore_score"] == "0.9"

    def test_uses_humu_embedding_scorer_to_align_mutations_with_intent_cone(self, tmp_path) -> None:
        from mf_core.types.humu import IntentCone
        from mf_generators.crem_3d.generator import CReM3DGenerator

        class HUMUScorer:
            async def score_batch(
                self,
                smiles_list: list[str],
                intent_cone: IntentCone | None = None,
            ) -> dict[str, dict]:
                return {
                    smiles: {
                        "humu_embedding": [1.0, 0.0]
                        if smiles == "Clc1ccccc1"
                        else [0.0, 1.0],
                    }
                    for smiles in smiles_list
                }

        mmp_path = tmp_path / "crem_mmp.json"
        mmp_path.write_text(
            json.dumps(
                {
                    "mutations": [
                        {
                            "id": "benzene_fluoro",
                            "seed_smiles": "c1ccccc1",
                            "product": "Fc1ccccc1",
                        },
                        {
                            "id": "benzene_chloro",
                            "seed_smiles": "c1ccccc1",
                            "product": "Clc1ccccc1",
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        generator = CReM3DGenerator(
            mmp_db_path=str(mmp_path),
            humu_embedding_scorer=HUMUScorer(),
        )

        molecules = asyncio.run(
            generator.generate(
                batch_size=2,
                seed_smiles="c1ccccc1",
                intent_cone=IntentCone(axis=[1.0, 1.0, 0.0]),
            )
        )

        assert [mol.metadata["mutation_id"] for mol in molecules] == [
            "benzene_chloro",
            "benzene_fluoro",
        ]
        assert molecules[0].metadata["humu_alignment_score"] == "1.0"
        assert json.loads(molecules[0].humu_embedding.decode("utf-8")) == [1.0, 0.0]

    def test_attachment_points(self) -> None:
        from mf_generators.crem_3d.fragment_replacement import get_attachment_points

        mol = Chem.MolFromSmiles("c1ccc(O)cc1")
        points = get_attachment_points(mol)
        assert len(points) > 0

    def test_fragment_replace(self) -> None:
        from mf_generators.crem_3d.fragment_replacement import replace_fragment

        mol = Chem.MolFromSmiles("c1ccccc1")
        result = replace_fragment(mol, 0, "F")
        if result is not None:
            smi = Chem.MolToSmiles(result)
            assert Chem.MolFromSmiles(smi) is not None

    def test_training_cli_writes_mmp_database_artifact(self, tmp_path) -> None:
        from mf_generators.crem_3d.generator import CReM3DGenerator

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "mutations.jsonl").write_text(
            json.dumps(
                {
                    "id": "benzene_fluoro",
                    "seed_smiles": "c1ccccc1",
                    "product": "Fc1ccccc1",
                },
            )
            + "\n",
            encoding="utf-8",
        )
        output_path = tmp_path / "crem_mmp.json"
        script = ROOT / "models/mf-generators/crem_3d/train.py"

        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--data",
                str(data_dir),
                "--output",
                str(output_path),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        assert payload["mutations"][0]["product"] == "Fc1ccccc1"
        assert "kd_alignment_score" not in payload["mutations"][0]
        generator = CReM3DGenerator(mmp_db_path=str(output_path))
        molecules = asyncio.run(generator.generate(batch_size=1, seed_smiles="c1ccccc1"))
        assert molecules[0].smiles == "Fc1ccccc1"

    def test_training_cli_writes_kd_embedding_loss_metadata(self, tmp_path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "mutations.jsonl").write_text(
            json.dumps(
                {
                    "id": "benzene_fluoro",
                    "seed_smiles": "c1ccccc1",
                    "product": "Fc1ccccc1",
                },
            )
            + "\n",
            encoding="utf-8",
        )
        teacher_embeddings = tmp_path / "teacher_embeddings.json"
        teacher_embeddings.write_text(
            json.dumps({"teacher_embeddings": [[1.0, -2.0, 3.0]]}),
            encoding="utf-8",
        )
        output_path = tmp_path / "crem_mmp.json"
        script = ROOT / "models/mf-generators/crem_3d/train.py"

        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--data",
                str(data_dir),
                "--output",
                str(output_path),
                "--kd-teacher-embeddings",
                str(teacher_embeddings),
                "--kd-weight",
                "0.5",
                "--kd-generator-idx",
                "2",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        manifest = json.loads(output_path.with_suffix(".manifest.json").read_text())
        assert manifest["kd_teacher_embeddings"] == str(teacher_embeddings)
        assert manifest["kd_weight"] == pytest.approx(0.5)
        assert manifest["kd_generator_idx"] == 2
        assert manifest["kd_loss"] > 0.0
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        projection = payload["kd_projection"]
        assert projection["schema_version"] == "linear_kd_projection.v1"
        assert projection["teacher_dim"] == 3
        assert len(projection["weights"]) == 3
        assert all(len(row) == 4 for row in projection["weights"])
        assert len(projection["bias"]) == 3
        mutation = payload["mutations"][0]
        expected_loss = _expected_linear_kd_loss(
            projection,
            [
                float(len(mutation["seed_smiles"])),
                float(len(mutation["fragment_smiles"])),
                float(mutation["attachment_index"] or 0),
                float(len(mutation["product"])),
            ],
            [1.0, -2.0, 3.0],
        )
        assert manifest["kd_loss"] == pytest.approx(expected_loss)

    def test_training_kd_alignment_is_persisted_and_ranks_mutations(self, tmp_path) -> None:
        from mf_generators.crem_3d.generator import CReM3DGenerator

        seed_smiles = Chem.MolToSmiles(Chem.MolFromSmiles("c1ccccc1"))
        aligned_product = Chem.MolToSmiles(Chem.MolFromSmiles("CCOc1ccccc1"))
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "mutations.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "id": "input_first",
                            "seed_smiles": seed_smiles,
                            "product": "Fc1ccccc1",
                        }
                    ),
                    json.dumps(
                        {
                            "id": "teacher_aligned",
                            "seed_smiles": seed_smiles,
                            "product": aligned_product,
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        teacher_embeddings = tmp_path / "teacher_embeddings.json"
        teacher_embeddings.write_text(
            json.dumps(
                {
                    "teacher_embeddings": [
                        [100.0, -50.0, 25.0],
                        [0.0, 0.0, 0.0],
                    ]
                }
            ),
            encoding="utf-8",
        )
        output_path = tmp_path / "crem_mmp.json"
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "models/mf-generators/crem_3d/train.py"),
                "--data",
                str(data_dir),
                "--output",
                str(output_path),
                "--kd-teacher-embeddings",
                str(teacher_embeddings),
                "--kd-weight",
                "0.5",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        mutations = json.loads(output_path.read_text(encoding="utf-8"))["mutations"]
        assert all(
            math.isfinite(mutation["kd_alignment_score"])
            for mutation in mutations
        )
        assert mutations[1]["kd_alignment_score"] > mutations[0]["kd_alignment_score"]
        assert mutations[1]["kd_weight"] == pytest.approx(0.5)

        generator = CReM3DGenerator(mmp_db_path=str(output_path))
        molecules = asyncio.run(
            generator.generate(batch_size=1, seed_smiles=seed_smiles)
        )
        assert molecules[0].metadata["mutation_id"] == "teacher_aligned"
        assert float(molecules[0].metadata["kd_alignment_score"]) == pytest.approx(
            mutations[1]["kd_alignment_score"]
        )

        baseline_path = tmp_path / "crem_baseline.json"
        baseline_result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "models/mf-generators/crem_3d/train.py"),
                "--data",
                str(data_dir),
                "--output",
                str(baseline_path),
                "--kd-teacher-embeddings",
                str(teacher_embeddings),
                "--kd-weight",
                "0",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert baseline_result.returncode == 0, baseline_result.stderr
        baseline_mutations = json.loads(
            baseline_path.read_text(encoding="utf-8")
        )["mutations"]
        assert all("kd_alignment_score" not in record for record in baseline_mutations)
        assert "kd_projection" not in json.loads(
            baseline_path.read_text(encoding="utf-8")
        )
        baseline_generator = CReM3DGenerator(mmp_db_path=str(baseline_path))
        baseline_molecules = asyncio.run(
            baseline_generator.generate(batch_size=1, seed_smiles=seed_smiles)
        )
        assert baseline_molecules[0].metadata["mutation_id"] == "input_first"

    def test_training_kd_requires_one_finite_teacher_row_per_mutation(
        self,
        tmp_path,
    ) -> None:
        data_path = tmp_path / "mutations.json"
        data_path.write_text(
            json.dumps(
                [
                    {"id": "first", "product": "CCO"},
                    {"id": "second", "product": "CCN"},
                ]
            ),
            encoding="utf-8",
        )
        teacher_embeddings = tmp_path / "teacher_embeddings.json"
        teacher_embeddings.write_text(
            json.dumps({"teacher_embeddings": [[0.0, 1.0, 2.0]]}),
            encoding="utf-8",
        )

        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "models/mf-generators/crem_3d/train.py"),
                "--data",
                str(data_path),
                "--output",
                str(tmp_path / "crem_mmp.json"),
                "--kd-teacher-embeddings",
                str(teacher_embeddings),
                "--kd-weight",
                "0.5",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert "row count must match CReM mutation records" in result.stderr

    def test_training_kd_rejects_non_finite_weight(self, tmp_path) -> None:
        data_path = tmp_path / "mutations.json"
        data_path.write_text(
            json.dumps([{"id": "first", "product": "CCO"}]),
            encoding="utf-8",
        )
        teacher_embeddings = tmp_path / "teacher_embeddings.json"
        teacher_embeddings.write_text(
            json.dumps({"teacher_embeddings": [[0.0, 1.0]]}),
            encoding="utf-8",
        )

        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "models/mf-generators/crem_3d/train.py"),
                "--data",
                str(data_path),
                "--output",
                str(tmp_path / "crem_mmp.json"),
                "--kd-teacher-embeddings",
                str(teacher_embeddings),
                "--kd-weight",
                "nan",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert "--kd-weight must be finite" in result.stderr

    def test_generator_rejects_incomplete_kd_projection(self, tmp_path) -> None:
        from mf_generators.crem_3d.generator import CReM3DGenerator

        artifact_path = tmp_path / "crem_mmp.json"
        artifact_path.write_text(
            json.dumps(
                {
                    "mutations": [
                        {
                            "id": "first",
                            "product": "CCO",
                            "kd_alignment_score": 0.8,
                            "kd_weight": 0.5,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="requires kd_projection"):
            CReM3DGenerator(mmp_db_path=str(artifact_path))


class TestFragFMGenerator:
    def test_requires_vocabulary_artifact_in_production(self) -> None:
        from mf_generators.fragfm.generator import FragFMGenerator

        with pytest.raises(RuntimeError, match="vocabulary artifact"):
            FragFMGenerator()

    def test_requires_model_checkpoint_and_decoder_to_be_configured_together(
        self,
        tmp_path,
    ) -> None:
        from mf_generators.fragfm.generator import FragFMGenerator

        vocab_path = tmp_path / "fragfm_vocab.json"
        vocab_path.write_text(
            json.dumps(
                {
                    "fragments": ["CC", "O"],
                    "assembly_rules": [
                        {
                            "id": "ethanol",
                            "fragments": ["CC", "O"],
                            "product": "CCO",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(RuntimeError, match="checkpoint and decoder"):
            FragFMGenerator(
                checkpoint_path=str(tmp_path / "model.pt"),
                vocab_path=str(vocab_path),
            )
        with pytest.raises(RuntimeError, match="checkpoint and decoder"):
            FragFMGenerator(
                vocab_path=str(vocab_path),
                decoder=lambda logits, *, rule, vocab: "CCO",
            )

    def test_external_decoder_runs_checkpoint_in_inference_mode(
        self,
        tmp_path,
    ) -> None:
        import torch
        from mf_generators.fragfm.generator import (
            ExternalFragFMDecoder,
            FragFMGenerator,
        )
        from mf_generators.fragfm.model.two_level_dfm import TwoLevelDFM

        vocab_path = tmp_path / "fragfm_vocab.json"
        vocab_path.write_text(
            json.dumps(
                {
                    "fragments": ["CC", "N"],
                    "assembly_rules": [
                        {
                            "id": "ethylamine",
                            "fragments": ["CC", "N"],
                            "product": "CCN",
                            "sa_score_bin": 3,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        checkpoint_path = tmp_path / "best_model.pt"
        torch.save(TwoLevelDFM(vocab_size=2).state_dict(), checkpoint_path)
        runner = tmp_path / "fragfm_decoder.py"
        runner.write_text(
            "\n".join(
                [
                    "import json",
                    "import sys",
                    "request = json.load(sys.stdin)",
                    "assert request['rule']['id'] == 'ethylamine'",
                    "assert request['vocabulary'] == ['CC', 'N']",
                    "assert len(request['fragment_logits']) == 1",
                    "assert len(request['fragment_logits'][0]) == 2",
                    "json.dump({'smiles': 'NCC'}, sys.stdout)",
                ]
            ),
            encoding="utf-8",
        )
        generator = FragFMGenerator(
            checkpoint_path=str(checkpoint_path),
            vocab_path=str(vocab_path),
            decoder=ExternalFragFMDecoder(
                f"{sys.executable} {runner}",
                timeout_seconds=2.0,
            ),
        )
        inference_states: list[tuple[bool, bool]] = []
        generator._model.register_forward_pre_hook(
            lambda model, inputs: inference_states.append(
                (model.training, torch.is_inference_mode_enabled())
            )
        )

        molecules = asyncio.run(generator.generate(batch_size=1))

        assert [molecule.smiles for molecule in molecules] == ["CCN"]
        assert molecules[0].metadata["model_checkpoint_applied"] == "true"
        assert inference_states == [(False, True)]

    @pytest.mark.parametrize(
        ("runner_source", "timeout_seconds", "error"),
        [
            ("raise SystemExit(9)", 2.0, "command failed"),
            ("print('not-json')", 2.0, "invalid JSON"),
            (
                "import json, sys; json.dump({}, sys.stdout)",
                2.0,
                "must return smiles",
            ),
            (
                "import json, sys; json.dump({'smiles': 'not-a-smiles'}, sys.stdout)",
                2.0,
                "invalid SMILES",
            ),
            ("import time; time.sleep(0.1)", 0.01, "timed out"),
        ],
    )
    def test_external_decoder_rejects_runner_failures_without_rule_fallback(
        self,
        tmp_path,
        runner_source,
        timeout_seconds,
        error,
    ) -> None:
        from mf_generators.fragfm.generator import (
            ExternalFragFMDecoder,
            FragFMGenerator,
        )
        from mf_generators.fragfm.model.two_level_dfm import TwoLevelDFM

        runner = tmp_path / "fragfm_decoder.py"
        runner.write_text(runner_source, encoding="utf-8")
        vocab_path = tmp_path / "fragfm_vocab.json"
        vocab_path.write_text(
            json.dumps(
                {
                    "fragments": ["CC", "N"],
                    "assembly_rules": [
                        {
                            "id": "ethylamine",
                            "fragments": ["CC", "N"],
                            "product": "CCN",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        generator = FragFMGenerator(
            vocab_path=str(vocab_path),
            model=TwoLevelDFM(vocab_size=2),
            decoder=ExternalFragFMDecoder(
                f"{sys.executable} {runner}",
                timeout_seconds=timeout_seconds,
            ),
        )

        with pytest.raises(RuntimeError, match=error):
            asyncio.run(generator.generate(batch_size=1))

    def test_uses_fragment_vocabulary_rules_and_validity_check(self, tmp_path) -> None:
        from mf_generators.fragfm.generator import FragFMGenerator

        vocab_path = tmp_path / "fragfm_vocab.json"
        vocab_path.write_text(
            json.dumps(
                {
                    "fragments": ["CC", "O"],
                    "assembly_rules": [
                        {
                            "id": "ethanol",
                            "fragments": ["CC", "O"],
                            "product": "CCO",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        generator = FragFMGenerator(vocab_path=str(vocab_path))

        molecules = asyncio.run(generator.generate(batch_size=2))

        assert [mol.smiles for mol in molecules] == ["CCO", "CCO"]
        assert all(Chem.MolFromSmiles(mol.smiles) is not None for mol in molecules)
        assert molecules[0].metadata["generator_name"] == "fragfm"
        assert molecules[0].metadata["assembly_rule_id"] == "ethanol"

    @pytest.mark.asyncio
    async def test_bootstraps_vocab_and_rate_without_claiming_checkpoint_inference(
        self,
        tmp_path: Path,
    ) -> None:
        import torch
        from mf_generators.fragfm.generator import (
            FragFMGenerator,
            bootstrap_validation_artifacts,
        )

        target = tmp_path / "fragfm-validation"
        paths = await bootstrap_validation_artifacts(target)
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))

        assert metadata["schema_version"] == "moleculeforge.validation_artifact.v1"
        assert metadata["purpose"] == "synthetic_pipeline_validation_only"
        assert metadata["generator"] == "fragfm"
        assert metadata["seed"] == 7
        assert metadata["model_checkpoint_included"] is False
        assert "checkpoint" not in paths
        generator = FragFMGenerator(
            vocab_path=str(paths["vocabulary"]),
            rate_matrix_path=str(paths["rate_matrix"]),
            checkpoint_path="",
            mode="production_real",
        )
        molecules = await generator.generate(batch_size=3)
        assert generator._model is None
        assert len(molecules) == 3
        assert all(Chem.MolFromSmiles(molecule.smiles) is not None for molecule in molecules)
        assert all(
            molecule.metadata["model_checkpoint_applied"] == "false"
            for molecule in molecules
        )

        rate_state = torch.load(
            paths["rate_matrix"],
            map_location="cpu",
            weights_only=True,
        )
        rate_state["moleculeforge_validation_artifact"]["seed"] = 8
        torch.save(rate_state, paths["rate_matrix"])
        metadata["artifacts"]["rate_matrix"]["sha256"] = hashlib.sha256(
            paths["rate_matrix"].read_bytes()
        ).hexdigest()
        paths["metadata"].write_text(json.dumps(metadata), encoding="utf-8")

        with pytest.raises(RuntimeError, match="refuses to overwrite"):
            await bootstrap_validation_artifacts(target)
        persisted_state = torch.load(
            paths["rate_matrix"],
            map_location="cpu",
            weights_only=True,
        )
        assert persisted_state["moleculeforge_validation_artifact"]["seed"] == 8

        occupied = tmp_path / "occupied-fragfm"
        occupied.mkdir()
        sentinel = occupied / "best_model.pt"
        sentinel.write_bytes(b"real")
        with pytest.raises(RuntimeError, match="refuses to overwrite"):
            await bootstrap_validation_artifacts(occupied)
        assert sentinel.read_bytes() == b"real"

    def test_intent_cone_conditions_rule_ranking_by_humu_embedding(self, tmp_path) -> None:
        from mf_core.types.humu import IntentCone
        from mf_generators.fragfm.generator import FragFMGenerator

        vocab_path = tmp_path / "fragfm_vocab.json"
        vocab_path.write_text(
            json.dumps(
                {
                    "fragments": ["O", "N"],
                    "assembly_rules": [
                        {
                            "id": "ethanol",
                            "fragments": ["O"],
                            "product": "CCO",
                            "humu_embedding": [1.0, 0.0],
                        },
                        {
                            "id": "ethylamine",
                            "fragments": ["N"],
                            "product": "CCN",
                            "humu_embedding": [0.0, 1.0],
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        generator = FragFMGenerator(vocab_path=str(vocab_path))

        molecules = asyncio.run(
            generator.generate(
                batch_size=1,
                intent_cone=IntentCone(axis=[0.0, 1.0]),
            )
        )

        assert molecules[0].smiles == "CCN"
        assert molecules[0].metadata["assembly_rule_id"] == "ethylamine"
        assert float(molecules[0].metadata["humu_condition_score"]) > 0.0

    def test_shared_humu_latent_sampler_conditions_rule_ranking(self, tmp_path) -> None:
        from mf_core.types.humu import IntentCone
        from mf_generators.fragfm.generator import FragFMGenerator

        class HumuLatentSampler:
            def __init__(self) -> None:
                self.calls: list[tuple[int, IntentCone | None]] = []

            def sample(
                self,
                *,
                batch_size: int,
                intent_cone: IntentCone | None,
            ) -> list[list[float]]:
                self.calls.append((batch_size, intent_cone))
                return [[0.0, 1.0] for _ in range(batch_size)]

        vocab_path = tmp_path / "fragfm_vocab.json"
        vocab_path.write_text(
            json.dumps(
                {
                    "fragments": ["O", "N"],
                    "assembly_rules": [
                        {
                            "id": "ethanol",
                            "fragments": ["O"],
                            "product": "CCO",
                            "humu_embedding": [1.0, 0.0],
                        },
                        {
                            "id": "ethylamine",
                            "fragments": ["N"],
                            "product": "CCN",
                            "humu_embedding": [0.0, 1.0],
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        sampler = HumuLatentSampler()
        generator = FragFMGenerator(
            vocab_path=str(vocab_path),
            humu_latent_sampler=sampler,
        )
        intent_cone = IntentCone(axis=[1.0, 0.0])

        molecules = asyncio.run(
            generator.generate(batch_size=1, intent_cone=intent_cone)
        )

        assert sampler.calls == [(1, intent_cone)]
        assert molecules[0].smiles == "CCN"
        assert molecules[0].metadata["assembly_rule_id"] == "ethylamine"
        assert molecules[0].metadata["humu_latent"] == "0.0,1.0"

    def test_loads_checkpoint_and_rate_matrix_artifacts(self, tmp_path) -> None:
        import torch
        from mf_generators.fragfm.generator import FragFMGenerator
        from mf_generators.fragfm.model.sa_aware_rate_matrix import SAAwareRateMatrix
        from mf_generators.fragfm.model.two_level_dfm import TwoLevelDFM

        vocab_path = tmp_path / "fragfm_vocab.json"
        vocab_path.write_text(
            json.dumps(
                {
                    "fragments": ["CC", "O"],
                    "assembly_rules": [
                        {
                            "id": "ethanol",
                            "fragments": ["CC", "O"],
                            "product": "CCO",
                            "sa_score_bin": 2,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        checkpoint_path = tmp_path / "best_model.pt"
        torch.save(TwoLevelDFM(vocab_size=2).state_dict(), checkpoint_path)
        rate_matrix_path = tmp_path / "rate_matrix.pt"
        trained_rate_matrix = SAAwareRateMatrix(vocab_size=2)
        trained_rate_matrix.base_rate.data.fill_(0.5)
        torch.save(trained_rate_matrix.state_dict(), rate_matrix_path)

        generator = FragFMGenerator(
            checkpoint_path=str(checkpoint_path),
            rate_matrix_path=str(rate_matrix_path),
            vocab_path=str(vocab_path),
            device="cpu",
            decoder=lambda logits, *, rule, vocab: "CCO",
        )

        assert generator._model is not None
        loaded_rate = float(generator.rate_matrix.base_rate[0, 0].detach().cpu().item())
        assert loaded_rate == pytest.approx(0.5)

    def test_rejects_rate_matrix_artifact_missing_sa_embedding_weight(self, tmp_path) -> None:
        import torch
        from mf_generators.fragfm.generator import FragFMGenerator

        vocab_path = tmp_path / "fragfm_vocab.json"
        vocab_path.write_text(
            json.dumps(
                {
                    "fragments": ["CC", "O"],
                    "assembly_rules": [
                        {
                            "id": "ethanol",
                            "fragments": ["CC", "O"],
                            "product": "CCO",
                            "sa_score_bin": 2,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        rate_matrix_path = tmp_path / "rate_matrix.pt"
        torch.save({"base_rate": torch.zeros(2, 2)}, rate_matrix_path)

        with pytest.raises(ValueError, match="sa_score_embedding.weight"):
            FragFMGenerator(
                rate_matrix_path=str(rate_matrix_path),
                vocab_path=str(vocab_path),
                device="cpu",
            )

    def test_rejects_checkpoint_artifact_missing_fragment_encoder_weight(self, tmp_path) -> None:
        import torch
        from mf_generators.fragfm.generator import FragFMGenerator

        vocab_path = tmp_path / "fragfm_vocab.json"
        vocab_path.write_text(
            json.dumps(
                {
                    "fragments": ["CC", "O"],
                    "assembly_rules": [
                        {
                            "id": "ethanol",
                            "fragments": ["CC", "O"],
                            "product": "CCO",
                            "sa_score_bin": 2,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        checkpoint_path = tmp_path / "best_model.pt"
        torch.save({"unrelated.weight": torch.zeros(2, 4)}, checkpoint_path)

        with pytest.raises(ValueError, match="fragment_encoder.weight"):
            FragFMGenerator(
                checkpoint_path=str(checkpoint_path),
                vocab_path=str(vocab_path),
                device="cpu",
                decoder=lambda logits, *, rule, vocab: "CCO",
            )

    def test_rejects_missing_checkpoint_artifact_when_path_is_explicit(
        self,
        tmp_path,
    ) -> None:
        from mf_generators.fragfm.generator import FragFMGenerator

        vocab_path = tmp_path / "fragfm_vocab.json"
        vocab_path.write_text(
            json.dumps(
                {
                    "fragments": ["CC", "O"],
                    "assembly_rules": [
                        {
                            "id": "ethanol",
                            "fragments": ["CC", "O"],
                            "product": "CCO",
                            "sa_score_bin": 2,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        checkpoint_path = tmp_path / "missing_best_model.pt"

        with pytest.raises(FileNotFoundError, match="checkpoint artifact"):
            FragFMGenerator(
                checkpoint_path=str(checkpoint_path),
                vocab_path=str(vocab_path),
                device="cpu",
                decoder=lambda logits, *, rule, vocab: "CCO",
            )

    def test_training_cli_writes_checkpoint_and_vocab_artifacts(self, tmp_path) -> None:
        import torch

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "fragfm.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "id": "ethanol",
                            "fragments": ["CC", "O"],
                            "product": "CCO",
                            "sa_score_bin": 2,
                        },
                    ),
                    json.dumps(
                        {
                            "id": "ethylamine",
                            "fragments": ["CC", "N"],
                            "product": "CCN",
                            "sa_score_bin": 3,
                        },
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        output_dir = tmp_path / "out"
        script = ROOT / "models/mf-generators/fragfm/train.py"

        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--data",
                str(data_dir),
                "--output-dir",
                str(output_dir),
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--hidden-dim",
                "16",
                "--device",
                "cpu",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        vocab = json.loads((output_dir / "vocab.json").read_text(encoding="utf-8"))
        assert vocab["fragments"] == ["CC", "N", "O"]
        assert [rule["id"] for rule in vocab["assembly_rules"]] == ["ethanol", "ethylamine"]
        state = torch.load(output_dir / "best_model.pt", map_location="cpu", weights_only=True)
        assert "fragment_encoder.weight" in state
        assert (output_dir / "final_model.pt").is_file()

    def test_training_cli_records_rate_optimizer_controls(self, tmp_path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "fragfm.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "id": "ethanol",
                            "fragments": ["CC", "O"],
                            "product": "CCO",
                            "sa_score_bin": 2,
                        },
                    ),
                    json.dumps(
                        {
                            "id": "ethylamine",
                            "fragments": ["CC", "N"],
                            "product": "CCN",
                            "sa_score_bin": 3,
                        },
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        output_dir = tmp_path / "out"
        script = ROOT / "models/mf-generators/fragfm/train.py"

        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--data",
                str(data_dir),
                "--output-dir",
                str(output_dir),
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--hidden-dim",
                "16",
                "--device",
                "cpu",
                "--rate-optimizer",
                "sgd",
                "--disable-rate-grad-clip",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        manifest = json.loads((output_dir / "training_manifest.json").read_text(encoding="utf-8"))
        assert manifest["rate_optimizer"] == "sgd"
        assert manifest["rate_grad_clip"] is False
        assert (output_dir / "rate_matrix.pt").is_file()

    def test_training_batch_log_policy_logs_first_interval_and_final_batches(self) -> None:
        module = _load_fragfm_train_module()

        assert module._should_log_batch(batch_number=1, total_batches=7, log_every=3) is True
        assert module._should_log_batch(batch_number=2, total_batches=7, log_every=3) is False
        assert module._should_log_batch(batch_number=3, total_batches=7, log_every=3) is True
        assert module._should_log_batch(batch_number=7, total_batches=7, log_every=3) is True
        assert module._should_log_batch(batch_number=1, total_batches=7, log_every=0) is False

    def test_training_manifest_records_runtime_controls_without_launching_training(
        self,
        tmp_path,
        monkeypatch,
    ) -> None:
        module = _load_fragfm_train_module()
        monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)

        assert str(module._resolve_training_device("cuda")) == "cpu"
        assert str(module._resolve_training_device("cpu")) == "cpu"

        manifest = module._training_manifest_payload(
            records=[
                {"id": "ethanol", "humu_embedding": [1.0] + [0.0] * 128},
                {"id": "ethylamine"},
            ],
            fragments=["CC", "N", "O"],
            epochs=1,
            best_loss=1.25,
            vocab_path=tmp_path / "vocab.json",
            output_dir=tmp_path / "out",
            kd_teacher_embeddings="",
            kd_weight=0.0,
            kd_generator_idx=0,
            humu_embedding_dim=129,
            humu_curvature=1.0,
            rate_optimizer="sgd",
            rate_grad_clip=False,
            requested_device="cuda",
            actual_device="cpu",
            log_every=1,
        )

        assert manifest["records"] == 2
        assert manifest["fragments"] == 3
        assert manifest["best_loss"] == pytest.approx(1.25)
        assert manifest["requested_device"] == "cuda"
        assert manifest["actual_device"] == "cpu"
        assert manifest["log_every"] == 1
        assert manifest["rate_optimizer"] == "sgd"
        assert manifest["rate_grad_clip"] is False
        assert manifest["humu_embedding_count"] == 1
        assert manifest["humu_embedding_coverage"] == pytest.approx(0.5)

    def test_training_cli_preserves_valid_humu_embeddings_in_vocab_artifact(
        self,
        tmp_path,
    ) -> None:
        module = _load_fragfm_train_module()

        def lorentz_embedding(first_spatial: float) -> list[float]:
            return [(1.0 + first_spatial * first_spatial) ** 0.5, first_spatial] + [0.0] * 127

        ethanol_embedding = lorentz_embedding(0.0)
        ethylamine_embedding = lorentz_embedding(1.0)
        records = [
            module._normalize_record(
                0,
                {
                    "id": "ethanol",
                    "fragments": ["CC", "O"],
                    "product": "CCO",
                    "sa_score_bin": 2,
                    "humu_embedding": ethanol_embedding,
                },
            ),
            module._normalize_record(
                1,
                {
                    "id": "ethylamine",
                    "fragments": ["CC", "N"],
                    "product": "CCN",
                    "sa_score_bin": 3,
                    "humu_embedding": ethylamine_embedding,
                },
            ),
        ]
        vocab_path = tmp_path / "vocab.json"

        module._write_vocab_artifact(vocab_path, ["CC", "N", "O"], records)

        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        rules = {rule["id"]: rule for rule in vocab["assembly_rules"]}
        assert rules["ethanol"]["humu_embedding"] == pytest.approx(ethanol_embedding)
        assert rules["ethylamine"]["humu_embedding"] == pytest.approx(ethylamine_embedding)
        stats = module._humu_embedding_stats(records)
        assert stats["humu_embedding_count"] == 2
        assert stats["humu_embedding_coverage"] == pytest.approx(1.0)

    def test_training_record_rejects_invalid_humu_embedding(self) -> None:
        module = _load_fragfm_train_module()

        with pytest.raises(ValueError, match="Lorentz full-coordinate"):
            module._normalize_record(
                0,
                {
                    "id": "invalid",
                    "fragments": ["CC", "O"],
                    "product": "CCO",
                    "sa_score_bin": 2,
                    "humu_embedding": [0.0] * 129,
                },
            )

    def test_fragfm_humu_labeling_writes_valid_embeddings_and_preserves_records(
        self,
        tmp_path,
    ) -> None:
        from mf_generators.fragfm.humu_labeling import label_fragfm_records

        class Encoder:
            def encode(self, smiles: str) -> list[float]:
                spatial = 0.1 if smiles == "CCO" else 0.2
                return [(1.0 + spatial * spatial) ** 0.5, spatial] + [0.0] * 127

        input_path = tmp_path / "records.jsonl"
        output_path = tmp_path / "records_humu.jsonl"
        input_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "id": "ethanol",
                            "fragments": ["CC", "O"],
                            "product": "CCO",
                            "sa_score_bin": 2,
                        },
                    ),
                    json.dumps(
                        {
                            "id": "ethylamine",
                            "fragments": ["CC", "N"],
                            "product": "CCN",
                            "sa_score_bin": 3,
                        },
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        report = label_fragfm_records(
            input_path=input_path,
            output_path=output_path,
            encoder=Encoder(),
            expected_humu_dim=129,
            curvature=1.0,
        )

        assert report["status"] == "pass"
        assert report["total_records"] == 2
        assert report["encoded_records"] == 2
        assert report["skipped_records"] == 0
        rows = [
            json.loads(line)
            for line in output_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert [row["id"] for row in rows] == ["ethanol", "ethylamine"]
        assert rows[0]["fragments"] == ["CC", "O"]
        assert len(rows[0]["humu_embedding"]) == 129
        assert rows[0]["humu_embedding"][0] == pytest.approx((1.0 + 0.1 * 0.1) ** 0.5)

    def test_fragfm_humu_labeling_skips_unencodable_products(self, tmp_path) -> None:
        from mf_generators.fragfm.humu_labeling import label_fragfm_records

        class Encoder:
            def encode(self, smiles: str) -> list[float]:
                if smiles == "not-a-smiles":
                    raise ValueError("invalid smiles")
                return [1.0] + [0.0] * 128

        input_path = tmp_path / "records.jsonl"
        output_path = tmp_path / "records_humu.jsonl"
        input_path.write_text(
            "\n".join(
                [
                    json.dumps({"id": "bad", "product": "not-a-smiles", "fragments": ["bad"]}),
                    json.dumps({"id": "ok", "product": "CCO", "fragments": ["CC", "O"]}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        report = label_fragfm_records(
            input_path=input_path,
            output_path=output_path,
            encoder=Encoder(),
        )

        rows = [
            json.loads(line)
            for line in output_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert report["status"] == "pass"
        assert report["total_records"] == 2
        assert report["encoded_records"] == 1
        assert report["invalid_smiles"] == 1
        assert rows[0]["id"] == "ok"

    def test_fragfm_humu_labeling_refuses_to_overwrite_input(self, tmp_path) -> None:
        from mf_generators.fragfm.humu_labeling import label_fragfm_records

        input_path = tmp_path / "records.jsonl"
        input_path.write_text(
            json.dumps({"id": "ethanol", "product": "CCO", "fragments": ["CC", "O"]}) + "\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="output path must differ"):
            label_fragfm_records(
                input_path=input_path,
                output_path=input_path,
                encoder=None,
            )

    def test_fragfm_humu_labeling_strict_fails_below_min_coverage(self, tmp_path) -> None:
        from mf_generators.fragfm.humu_labeling import label_fragfm_records

        class Encoder:
            def encode(self, smiles: str) -> list[float]:
                if smiles == "bad":
                    raise ValueError("invalid smiles")
                return [1.0] + [0.0] * 128

        input_path = tmp_path / "records.jsonl"
        output_path = tmp_path / "records_humu.jsonl"
        input_path.write_text(
            "\n".join(
                [
                    json.dumps({"id": "bad", "product": "bad", "fragments": ["bad"]}),
                    json.dumps({"id": "ok", "product": "CCO", "fragments": ["CC", "O"]}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        report = label_fragfm_records(
            input_path=input_path,
            output_path=output_path,
            encoder=Encoder(),
            min_coverage=0.75,
            strict=True,
        )

        assert report["status"] == "fail"
        assert report["encoded_records"] == 1
        assert report["humu_embedding_coverage"] == pytest.approx(0.5)
        assert any("coverage" in message for message in report["messages"])

    def test_rate_transition_loss_matches_full_rate_matrix_without_materializing_batches(
        self,
    ) -> None:
        import torch

        module = _load_fragfm_train_module()

        class RateMatrix(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base_rate = torch.nn.Parameter(
                    torch.arange(25, dtype=torch.float32).view(5, 5)
                )
                self.sa_score_embedding = torch.nn.Embedding(10, 25)
                torch.nn.init.zeros_(self.sa_score_embedding.weight)

            def forward(self, sa_score_bin: torch.Tensor) -> torch.Tensor:
                modulation = self.sa_score_embedding(sa_score_bin).view(-1, 5, 5)
                return self.base_rate * (1 + torch.tanh(modulation))

        rate_matrix = RateMatrix()
        fragment_ids = torch.tensor(
            [
                [0, 1, 2, 0],
                [2, 3, 0, 0],
                [4, 0, 0, 0],
            ],
            dtype=torch.long,
        )
        lengths = torch.tensor([3, 2, 1], dtype=torch.long)
        sa_bins = torch.tensor([1, 2, 3], dtype=torch.long)
        full_rates = rate_matrix(sa_bins)
        expected_losses = [
            torch.nn.functional.cross_entropy(full_rates[0, 0].unsqueeze(0), torch.tensor([1])),
            torch.nn.functional.cross_entropy(full_rates[0, 1].unsqueeze(0), torch.tensor([2])),
            torch.nn.functional.cross_entropy(full_rates[1, 2].unsqueeze(0), torch.tensor([3])),
        ]
        expected = torch.stack(expected_losses).mean()

        actual = module._rate_transition_loss(rate_matrix, fragment_ids, lengths, sa_bins)

        assert float(actual.detach().cpu().item()) == pytest.approx(
            float(expected.detach().cpu().item())
        )

    def test_rate_transition_loss_gathers_sa_rows_without_embedding_forward(
        self,
    ) -> None:
        import torch

        module = _load_fragfm_train_module()

        class NoFullEmbedding(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(10, 25))

            def forward(self, _indices: torch.Tensor) -> torch.Tensor:
                raise AssertionError("full SA embedding forward should not be called")

        class RateMatrix(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base_rate = torch.nn.Parameter(
                    torch.arange(25, dtype=torch.float32).view(5, 5)
                )
                self.sa_score_embedding = NoFullEmbedding()

        rate_matrix = RateMatrix()
        fragment_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
        lengths = torch.tensor([3], dtype=torch.long)
        sa_bins = torch.tensor([1], dtype=torch.long)
        expected_losses = [
            torch.nn.functional.cross_entropy(
                rate_matrix.base_rate[0].unsqueeze(0),
                torch.tensor([1]),
            ),
            torch.nn.functional.cross_entropy(
                rate_matrix.base_rate[1].unsqueeze(0),
                torch.tensor([2]),
            ),
        ]
        expected = torch.stack(expected_losses).mean()

        actual = module._rate_transition_loss(rate_matrix, fragment_ids, lengths, sa_bins)

        assert float(actual.detach().cpu().item()) == pytest.approx(
            float(expected.detach().cpu().item())
        )

    def test_training_cli_writes_kd_embedding_loss_metadata(self, tmp_path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "fragfm.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "id": "ethanol",
                            "fragments": ["CC", "O"],
                            "product": "CCO",
                            "sa_score_bin": 2,
                        },
                    ),
                    json.dumps(
                        {
                            "id": "ethylamine",
                            "fragments": ["CC", "N"],
                            "product": "CCN",
                            "sa_score_bin": 3,
                        },
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        output_dir = tmp_path / "out"
        teacher_embeddings = tmp_path / "teacher_embeddings.json"
        teacher_embeddings.write_text(
            json.dumps({"teacher_embeddings": [[0.0] * 16]}),
            encoding="utf-8",
        )
        script = ROOT / "models/mf-generators/fragfm/train.py"

        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--data",
                str(data_dir),
                "--output-dir",
                str(output_dir),
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--hidden-dim",
                "16",
                "--device",
                "cpu",
                "--kd-teacher-embeddings",
                str(teacher_embeddings),
                "--kd-weight",
                "0.25",
                "--kd-generator-idx",
                "1",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        manifest = json.loads((output_dir / "training_manifest.json").read_text(encoding="utf-8"))
        assert manifest["kd_teacher_embeddings"] == str(teacher_embeddings)
        assert manifest["kd_weight"] == pytest.approx(0.25)
        assert manifest["kd_generator_idx"] == 1

    def test_fragfm_quality_report_passes_with_valid_shared_humu_artifacts(
        self,
        tmp_path,
    ) -> None:
        import torch
        from mf_generators.fragfm.model.sa_aware_rate_matrix import SAAwareRateMatrix
        from mf_generators.fragfm.model.two_level_dfm import TwoLevelDFM
        from mf_generators.fragfm.quality import build_quality_report

        vocab_path = tmp_path / "fragfm_vocab.json"
        vocab_path.write_text(
            json.dumps(
                {
                    "fragments": ["CC", "O"],
                    "assembly_rules": [
                        {
                            "id": "ethanol",
                            "fragments": ["CC", "O"],
                            "product": "CCO",
                            "sa_score_bin": 2,
                            "humu_embedding": [1.0] + [0.0] * 128,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        checkpoint_path = tmp_path / "best_model.pt"
        rate_matrix_path = tmp_path / "rate_matrix.pt"
        torch.save(TwoLevelDFM(vocab_size=2).state_dict(), checkpoint_path)
        torch.save(SAAwareRateMatrix(vocab_size=2).state_dict(), rate_matrix_path)

        report = build_quality_report(
            vocab_path=vocab_path,
            checkpoint_path=checkpoint_path,
            rate_matrix_path=rate_matrix_path,
            min_humu_coverage=1.0,
            expected_humu_dim=129,
        )

        assert report["status"] == "pass"
        assert report["rules"] == 1
        assert report["humu_embedding_count"] == 1
        assert report["humu_embedding_coverage"] == pytest.approx(1.0)
        assert report["invalid_humu_embeddings"] == 0
        assert report["checkpoint_loadable"] is True
        assert report["rate_matrix_loadable"] is True

    def test_fragfm_quality_report_fails_when_training_manifest_disagrees(
        self,
        tmp_path,
    ) -> None:
        import torch
        from mf_generators.fragfm.model.sa_aware_rate_matrix import SAAwareRateMatrix
        from mf_generators.fragfm.model.two_level_dfm import TwoLevelDFM
        from mf_generators.fragfm.quality import build_quality_report

        vocab_path = tmp_path / "fragfm_vocab.json"
        vocab_path.write_text(
            json.dumps(
                {
                    "fragments": ["CC", "O"],
                    "assembly_rules": [
                        {
                            "id": "ethanol",
                            "fragments": ["CC", "O"],
                            "product": "CCO",
                            "sa_score_bin": 2,
                            "humu_embedding": [1.0] + [0.0] * 128,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        checkpoint_path = tmp_path / "best_model.pt"
        rate_matrix_path = tmp_path / "rate_matrix.pt"
        torch.save(TwoLevelDFM(vocab_size=2).state_dict(), checkpoint_path)
        torch.save(SAAwareRateMatrix(vocab_size=2).state_dict(), rate_matrix_path)
        manifest_path = tmp_path / "training_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": "fragfm_training.v1",
                    "records": 1,
                    "fragments": 999,
                    "vocab_path": str(vocab_path),
                    "checkpoint_path": str(checkpoint_path),
                    "rate_matrix_path": str(rate_matrix_path),
                    "humu_embedding_count": 1,
                    "humu_embedding_coverage": 1.0,
                    "humu_embedding_dim": 129,
                    "humu_curvature": 1.0,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        report = build_quality_report(
            vocab_path=vocab_path,
            checkpoint_path=checkpoint_path,
            rate_matrix_path=rate_matrix_path,
            manifest_path=manifest_path,
            min_humu_coverage=1.0,
            expected_humu_dim=129,
        )

        assert report["status"] == "fail"
        assert report["manifest_consistent"] is False
        assert any(
            "manifest" in message and "fragments" in message
            for message in report["messages"]
        )

    def test_fragfm_quality_cli_strict_fails_when_training_manifest_disagrees(
        self,
        tmp_path,
    ) -> None:
        import torch
        from mf_generators.fragfm.model.sa_aware_rate_matrix import SAAwareRateMatrix
        from mf_generators.fragfm.model.two_level_dfm import TwoLevelDFM
        from mf_generators.fragfm.quality import main

        vocab_path = tmp_path / "fragfm_vocab.json"
        vocab_path.write_text(
            json.dumps(
                {
                    "fragments": ["CC", "O"],
                    "assembly_rules": [
                        {
                            "id": "ethanol",
                            "fragments": ["CC", "O"],
                            "product": "CCO",
                            "sa_score_bin": 2,
                            "humu_embedding": [1.0] + [0.0] * 128,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        checkpoint_path = tmp_path / "best_model.pt"
        rate_matrix_path = tmp_path / "rate_matrix.pt"
        torch.save(TwoLevelDFM(vocab_size=2).state_dict(), checkpoint_path)
        torch.save(SAAwareRateMatrix(vocab_size=2).state_dict(), rate_matrix_path)
        manifest_path = tmp_path / "training_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": "fragfm_training.v1",
                    "records": 1,
                    "fragments": 999,
                    "vocab_path": str(vocab_path),
                    "checkpoint_path": str(checkpoint_path),
                    "rate_matrix_path": str(rate_matrix_path),
                    "humu_embedding_count": 1,
                    "humu_embedding_coverage": 1.0,
                    "humu_embedding_dim": 129,
                    "humu_curvature": 1.0,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        report_path = tmp_path / "quality_report.json"

        exit_code = main(
            [
                "--vocab",
                str(vocab_path),
                "--checkpoint",
                str(checkpoint_path),
                "--rate-matrix",
                str(rate_matrix_path),
                "--manifest",
                str(manifest_path),
                "--min-humu-coverage",
                "1.0",
                "--strict",
                "--output",
                str(report_path),
            ]
        )

        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert exit_code == 1
        assert report["status"] == "fail"
        assert report["manifest_consistent"] is False

    def test_fragfm_quality_cli_does_not_leave_report_when_output_write_fails(
        self,
        tmp_path,
        monkeypatch,
    ) -> None:
        import torch
        from mf_generators.fragfm.model.sa_aware_rate_matrix import SAAwareRateMatrix
        from mf_generators.fragfm.model.two_level_dfm import TwoLevelDFM
        from mf_generators.fragfm.quality import main

        vocab_path = tmp_path / "fragfm_vocab.json"
        vocab_path.write_text(
            json.dumps(
                {
                    "fragments": ["CC", "O"],
                    "assembly_rules": [
                        {
                            "id": "ethanol",
                            "fragments": ["CC", "O"],
                            "product": "CCO",
                            "sa_score_bin": 2,
                            "humu_embedding": [1.0] + [0.0] * 128,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        checkpoint_path = tmp_path / "best_model.pt"
        rate_matrix_path = tmp_path / "rate_matrix.pt"
        torch.save(TwoLevelDFM(vocab_size=2).state_dict(), checkpoint_path)
        torch.save(SAAwareRateMatrix(vocab_size=2).state_dict(), rate_matrix_path)
        report_path = tmp_path / "quality_report.json"
        original_write_text = Path.write_text

        def fail_quality_report_write(path, data, *args, **kwargs):
            if path == report_path or path.name.startswith(f".{report_path.name}."):
                original_write_text(path, "partial", encoding="utf-8")
                raise RuntimeError("simulated quality report write failure")
            return original_write_text(path, data, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", fail_quality_report_write)

        with pytest.raises(RuntimeError, match="simulated quality report write failure"):
            main(
                [
                    "--vocab",
                    str(vocab_path),
                    "--checkpoint",
                    str(checkpoint_path),
                    "--rate-matrix",
                    str(rate_matrix_path),
                    "--min-humu-coverage",
                    "1.0",
                    "--strict",
                    "--output",
                    str(report_path),
                ]
            )

        assert not report_path.exists()
        assert not list(tmp_path.glob(".quality_report.json.*.tmp"))

    def test_fragfm_quality_report_fails_when_humu_coverage_is_too_low(
        self,
        tmp_path,
    ) -> None:
        import torch
        from mf_generators.fragfm.model.sa_aware_rate_matrix import SAAwareRateMatrix
        from mf_generators.fragfm.model.two_level_dfm import TwoLevelDFM
        from mf_generators.fragfm.quality import build_quality_report

        vocab_path = tmp_path / "fragfm_vocab.json"
        vocab_path.write_text(
            json.dumps(
                {
                    "fragments": ["CC", "O"],
                    "assembly_rules": [
                        {
                            "id": "ethanol",
                            "fragments": ["CC", "O"],
                            "product": "CCO",
                            "sa_score_bin": 2,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        checkpoint_path = tmp_path / "best_model.pt"
        rate_matrix_path = tmp_path / "rate_matrix.pt"
        torch.save(TwoLevelDFM(vocab_size=2).state_dict(), checkpoint_path)
        torch.save(SAAwareRateMatrix(vocab_size=2).state_dict(), rate_matrix_path)

        report = build_quality_report(
            vocab_path=vocab_path,
            checkpoint_path=checkpoint_path,
            rate_matrix_path=rate_matrix_path,
            min_humu_coverage=1.0,
            expected_humu_dim=129,
        )

        assert report["status"] == "fail"
        assert report["humu_embedding_count"] == 0
        assert report["humu_embedding_coverage"] == 0.0
        assert any("coverage" in message for message in report["messages"])

    def test_fragfm_sample_export_writes_smiles_and_report(self, tmp_path) -> None:
        from mf_generators.fragfm.sample_export import export_fragfm_samples

        vocab_path = tmp_path / "fragfm_vocab.json"
        vocab_path.write_text(
            json.dumps(
                {
                    "fragments": ["CC", "O", "N"],
                    "assembly_rules": [
                        {
                            "id": "ethanol",
                            "fragments": ["CC", "O"],
                            "product": "CCO",
                            "sa_score_bin": 2,
                        },
                        {
                            "id": "ethylamine",
                            "fragments": ["CC", "N"],
                            "product": "CCN",
                            "sa_score_bin": 3,
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        output_path = tmp_path / "fragfm_generated.smi"
        report_path = tmp_path / "fragfm_generated.report.json"

        report = export_fragfm_samples(
            vocab_path=vocab_path,
            output_path=output_path,
            report_path=report_path,
            sample_count=3,
        )

        smiles = output_path.read_text(encoding="utf-8").splitlines()
        assert len(smiles) == 3
        assert all(Chem.MolFromSmiles(smile) is not None for smile in smiles)
        assert report["schema_version"] == "fragfm_sample_export_report.v1"
        assert report["requested_samples"] == 3
        assert report["generated_samples"] == 3
        assert report["valid_smiles"] == 3
        assert report["validity"] == pytest.approx(1.0)
        assert report["unique_smiles"] == 2
        assert report["output_path"] == str(output_path)
        written_report = json.loads(report_path.read_text(encoding="utf-8"))
        assert written_report == report

    def test_fragfm_sample_export_applies_checkpoint_with_external_decoder(
        self,
        tmp_path,
    ) -> None:
        import torch
        from mf_generators.fragfm.model.two_level_dfm import TwoLevelDFM
        from mf_generators.fragfm.sample_export import export_fragfm_samples

        vocab_path = tmp_path / "fragfm_vocab.json"
        vocab_path.write_text(
            json.dumps(
                {
                    "fragments": ["CC", "N"],
                    "assembly_rules": [
                        {
                            "id": "ethylamine",
                            "fragments": ["CC", "N"],
                            "product": "CCN",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        checkpoint_path = tmp_path / "best_model.pt"
        torch.save(TwoLevelDFM(vocab_size=2).state_dict(), checkpoint_path)
        runner = tmp_path / "fragfm_decoder.py"
        runner.write_text(
            "import json, sys; json.load(sys.stdin); "
            "json.dump({'smiles': 'NCC'}, sys.stdout)",
            encoding="utf-8",
        )
        output_path = tmp_path / "fragfm_generated.smi"

        report = export_fragfm_samples(
            vocab_path=vocab_path,
            output_path=output_path,
            sample_count=1,
            checkpoint_path=checkpoint_path,
            decoder_command=f"{sys.executable} {runner}",
            decoder_timeout_seconds=2.0,
        )

        assert output_path.read_text(encoding="utf-8") == "CCN\n"
        assert report["generated_samples"] == 1

    def test_fragfm_sample_export_does_not_leave_smiles_when_report_write_fails(
        self,
        tmp_path,
    ) -> None:
        from mf_generators.fragfm.sample_export import export_fragfm_samples

        vocab_path = tmp_path / "fragfm_vocab.json"
        vocab_path.write_text(
            json.dumps(
                {
                    "fragments": ["CC", "O"],
                    "assembly_rules": [
                        {
                            "id": "ethanol",
                            "fragments": ["CC", "O"],
                            "product": "CCO",
                            "sa_score_bin": 2,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        output_path = tmp_path / "fragfm_generated.smi"
        report_parent = tmp_path / "not_a_directory"
        report_parent.write_text("file blocks report directory creation", encoding="utf-8")

        with pytest.raises(FileExistsError):
            export_fragfm_samples(
                vocab_path=vocab_path,
                output_path=output_path,
                report_path=report_parent / "fragfm_generated.report.json",
                sample_count=2,
            )

        assert not output_path.exists()

    def test_fragfm_sample_export_rejects_blocked_output_parent_before_generation(
        self,
        monkeypatch,
        tmp_path,
    ) -> None:
        from mf_generators.fragfm import sample_export

        vocab_path = tmp_path / "fragfm_vocab.json"
        vocab_path.write_text(
            json.dumps(
                {
                    "fragments": ["CC", "O"],
                    "assembly_rules": [
                        {
                            "id": "ethanol",
                            "fragments": ["CC", "O"],
                            "product": "CCO",
                            "sa_score_bin": 2,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        output_parent = tmp_path / "not_a_directory"
        output_parent.write_text("file blocks output directory creation", encoding="utf-8")

        class FailingGenerator:
            def __init__(self, *args, **kwargs):
                raise AssertionError("generator should not be constructed")

        monkeypatch.setattr(sample_export, "FragFMGenerator", FailingGenerator)

        with pytest.raises(FileExistsError):
            sample_export.export_fragfm_samples(
                vocab_path=vocab_path,
                output_path=output_parent / "fragfm_generated.smi",
                sample_count=2,
            )

        assert output_parent.read_text(encoding="utf-8") == (
            "file blocks output directory creation"
        )

    def test_fragfm_sample_export_rejects_same_output_and_report_path(
        self,
        tmp_path,
    ) -> None:
        from mf_generators.fragfm.sample_export import export_fragfm_samples

        vocab_path = tmp_path / "fragfm_vocab.json"
        vocab_path.write_text(
            json.dumps(
                {
                    "fragments": ["CC", "O"],
                    "assembly_rules": [
                        {
                            "id": "ethanol",
                            "fragments": ["CC", "O"],
                            "product": "CCO",
                            "sa_score_bin": 2,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        output_path = tmp_path / "fragfm_generated.smi"

        with pytest.raises(ValueError, match="report_path must differ"):
            export_fragfm_samples(
                vocab_path=vocab_path,
                output_path=output_path,
                report_path=output_path,
                sample_count=2,
            )

        assert not output_path.exists()

    def test_fragfm_quality_report_fails_when_checkpoint_schema_missing(
        self,
        tmp_path,
    ) -> None:
        import torch
        from mf_generators.fragfm.quality import build_quality_report

        vocab_path = tmp_path / "fragfm_vocab.json"
        vocab_path.write_text(
            json.dumps(
                {
                    "fragments": ["CC", "O"],
                    "assembly_rules": [
                        {
                            "id": "ethanol",
                            "fragments": ["CC", "O"],
                            "product": "CCO",
                            "sa_score_bin": 2,
                            "humu_embedding": [1.0] + [0.0] * 128,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        checkpoint_path = tmp_path / "best_model.pt"
        rate_matrix_path = tmp_path / "rate_matrix.pt"
        torch.save({"unrelated.weight": torch.zeros(2, 4)}, checkpoint_path)
        torch.save({"base_rate": torch.zeros(2, 2)}, rate_matrix_path)

        report = build_quality_report(
            vocab_path=vocab_path,
            checkpoint_path=checkpoint_path,
            rate_matrix_path=rate_matrix_path,
            min_humu_coverage=1.0,
            expected_humu_dim=129,
        )

        assert report["status"] == "fail"
        assert report["checkpoint_loadable"] is False
        assert any("fragment_encoder.weight" in message for message in report["messages"])

    def test_fragfm_quality_report_fails_when_rate_matrix_schema_missing(
        self,
        tmp_path,
    ) -> None:
        import torch
        from mf_generators.fragfm.quality import build_quality_report

        vocab_path = tmp_path / "fragfm_vocab.json"
        vocab_path.write_text(
            json.dumps(
                {
                    "fragments": ["CC", "O"],
                    "assembly_rules": [
                        {
                            "id": "ethanol",
                            "fragments": ["CC", "O"],
                            "product": "CCO",
                            "sa_score_bin": 2,
                            "humu_embedding": [1.0] + [0.0] * 128,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        checkpoint_path = tmp_path / "best_model.pt"
        rate_matrix_path = tmp_path / "rate_matrix.pt"
        torch.save({"fragment_encoder.weight": torch.zeros(2, 4)}, checkpoint_path)
        torch.save({"unrelated.bias": torch.zeros(2)}, rate_matrix_path)

        report = build_quality_report(
            vocab_path=vocab_path,
            checkpoint_path=checkpoint_path,
            rate_matrix_path=rate_matrix_path,
            min_humu_coverage=1.0,
            expected_humu_dim=129,
        )

        assert report["status"] == "fail"
        assert report["rate_matrix_loadable"] is False
        assert any("base_rate" in message for message in report["messages"])

    def test_fragfm_quality_report_fails_when_rate_matrix_sa_embedding_missing(
        self,
        tmp_path,
    ) -> None:
        import torch
        from mf_generators.fragfm.quality import build_quality_report

        vocab_path = tmp_path / "fragfm_vocab.json"
        vocab_path.write_text(
            json.dumps(
                {
                    "fragments": ["CC", "O"],
                    "assembly_rules": [
                        {
                            "id": "ethanol",
                            "fragments": ["CC", "O"],
                            "product": "CCO",
                            "sa_score_bin": 2,
                            "humu_embedding": [1.0] + [0.0] * 128,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        checkpoint_path = tmp_path / "best_model.pt"
        rate_matrix_path = tmp_path / "rate_matrix.pt"
        torch.save({"fragment_encoder.weight": torch.zeros(2, 4)}, checkpoint_path)
        torch.save({"base_rate": torch.zeros(2, 2)}, rate_matrix_path)

        report = build_quality_report(
            vocab_path=vocab_path,
            checkpoint_path=checkpoint_path,
            rate_matrix_path=rate_matrix_path,
            min_humu_coverage=1.0,
            expected_humu_dim=129,
        )

        assert report["status"] == "fail"
        assert report["rate_matrix_loadable"] is False
        assert any("sa_score_embedding.weight" in message for message in report["messages"])


class TestMMPTRAGTraining:
    def test_training_cli_writes_mmp_index_artifact(self, tmp_path) -> None:
        from mf_generators.mmpt_rag.generator import MMPTRAGGenerator

        data_dir = tmp_path / "mmp"
        data_dir.mkdir()
        (data_dir / "pairs.tsv").write_text(
            "seed_smiles\tproduct_smiles\n"
            "c1ccccc1F\tc1ccccc1Cl\n",
            encoding="utf-8",
        )
        output_path = tmp_path / "mmpt_index.json"
        script = ROOT / "models/mf-generators/mmpt_rag/train.py"

        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--data",
                str(data_dir),
                "--output",
                str(output_path),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        assert payload["transforms"][0]["pattern"] == "F"
        assert payload["transforms"][0]["replacement"] == "Cl"
        assert "kd_alignment_score" not in payload["transforms"][0]
        generator = MMPTRAGGenerator(index_path=str(output_path))
        mols = _run_async_iter(generator.generate(None, None, None, n_samples=1))
        assert mols[0].smiles == payload["transforms"][0]["product_smiles"]

    def test_training_cli_writes_kd_embedding_loss_metadata(self, tmp_path) -> None:
        data_dir = tmp_path / "mmp"
        data_dir.mkdir()
        (data_dir / "pairs.tsv").write_text(
            "seed_smiles\tproduct_smiles\n"
            "c1ccccc1F\tc1ccccc1Cl\n",
            encoding="utf-8",
        )
        teacher_embeddings = tmp_path / "teacher_embeddings.json"
        teacher_embeddings.write_text(
            json.dumps({"teacher_embeddings": [[1.0, -2.0, 3.0, -4.0, 5.0]]}),
            encoding="utf-8",
        )
        output_path = tmp_path / "mmpt_index.json"
        script = ROOT / "models/mf-generators/mmpt_rag/train.py"

        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--data",
                str(data_dir),
                "--output",
                str(output_path),
                "--kd-teacher-embeddings",
                str(teacher_embeddings),
                "--kd-weight",
                "0.25",
                "--kd-generator-idx",
                "3",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        manifest = json.loads(output_path.with_suffix(".manifest.json").read_text())
        assert manifest["kd_teacher_embeddings"] == str(teacher_embeddings)
        assert manifest["kd_weight"] == pytest.approx(0.25)
        assert manifest["kd_generator_idx"] == 3
        assert manifest["kd_loss"] > 0.0
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        projection = payload["kd_projection"]
        assert projection["schema_version"] == "linear_kd_projection.v1"
        assert projection["teacher_dim"] == 5
        assert len(projection["weights"]) == 5
        assert all(len(row) == 4 for row in projection["weights"])
        assert len(projection["bias"]) == 5
        transform = payload["transforms"][0]
        expected_loss = _expected_linear_kd_loss(
            projection,
            [
                float(len(transform["seed_smiles"])),
                float(len(transform["product_smiles"])),
                float(len(transform["pattern"])),
                float(len(transform["replacement"])),
            ],
            [1.0, -2.0, 3.0, -4.0, 5.0],
        )
        assert manifest["kd_loss"] == pytest.approx(expected_loss)

    def test_training_kd_alignment_is_persisted_and_ranks_transforms(self, tmp_path) -> None:
        data_dir = tmp_path / "mmp"
        data_dir.mkdir()
        (data_dir / "pairs.tsv").write_text(
            "id\tseed_smiles\tproduct_smiles\n"
            "input_first\tc1ccccc1F\tc1ccccc1Cl\n"
            "teacher_aligned\tCCF\tCCN\n",
            encoding="utf-8",
        )
        teacher_embeddings = tmp_path / "teacher_embeddings.json"
        teacher_embeddings.write_text(
            json.dumps(
                {
                    "teacher_embeddings": [
                        [100.0, -50.0, 25.0],
                        [0.0, 0.0, 0.0],
                    ]
                }
            ),
            encoding="utf-8",
        )
        output_path = tmp_path / "mmpt_index.json"
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "models/mf-generators/mmpt_rag/train.py"),
                "--data",
                str(data_dir),
                "--output",
                str(output_path),
                "--kd-teacher-embeddings",
                str(teacher_embeddings),
                "--kd-weight",
                "0.25",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        transforms = json.loads(output_path.read_text(encoding="utf-8"))["transforms"]
        assert all(
            math.isfinite(transform["kd_alignment_score"])
            for transform in transforms
        )
        assert transforms[1]["kd_alignment_score"] > transforms[0]["kd_alignment_score"]
        assert transforms[1]["kd_weight"] == pytest.approx(0.25)

        from mf_generators.mmpt_rag.generator import MMPTRAGGenerator

        generator = MMPTRAGGenerator(index_path=str(output_path))
        molecules = _run_async_iter(
            generator.generate(None, None, None, n_samples=1)
        )
        assert molecules[0].properties["transform_id"] == "teacher_aligned"
        assert molecules[0].properties["kd_alignment_score"] == pytest.approx(
            transforms[1]["kd_alignment_score"]
        )

        baseline_path = tmp_path / "mmpt_baseline.json"
        baseline_result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "models/mf-generators/mmpt_rag/train.py"),
                "--data",
                str(data_dir),
                "--output",
                str(baseline_path),
                "--kd-teacher-embeddings",
                str(teacher_embeddings),
                "--kd-weight",
                "0",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert baseline_result.returncode == 0, baseline_result.stderr
        baseline_transforms = json.loads(
            baseline_path.read_text(encoding="utf-8")
        )["transforms"]
        assert all(
            "kd_alignment_score" not in transform
            for transform in baseline_transforms
        )
        assert "kd_projection" not in json.loads(
            baseline_path.read_text(encoding="utf-8")
        )
        baseline_generator = MMPTRAGGenerator(index_path=str(baseline_path))
        baseline_molecules = _run_async_iter(
            baseline_generator.generate(None, None, None, n_samples=1)
        )
        assert baseline_molecules[0].properties["transform_id"] == "input_first"

    def test_training_kd_requires_one_finite_teacher_row_per_pair(
        self,
        tmp_path,
    ) -> None:
        data_path = tmp_path / "pairs.json"
        data_path.write_text(
            json.dumps(
                [
                    {"seed_smiles": "CCF", "product_smiles": "CCN"},
                    {"seed_smiles": "CCCl", "product_smiles": "CCBr"},
                ]
            ),
            encoding="utf-8",
        )
        teacher_embeddings = tmp_path / "teacher_embeddings.json"
        teacher_embeddings.write_text(
            json.dumps({"teacher_embeddings": [[0.0, 1.0, 2.0]]}),
            encoding="utf-8",
        )

        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "models/mf-generators/mmpt_rag/train.py"),
                "--data",
                str(data_path),
                "--output",
                str(tmp_path / "mmpt_index.json"),
                "--kd-teacher-embeddings",
                str(teacher_embeddings),
                "--kd-weight",
                "0.25",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert "row count must match MMPT transform records" in result.stderr

    def test_training_kd_pairs_teacher_rows_with_deduplicated_transforms(
        self,
        tmp_path,
    ) -> None:
        data_path = tmp_path / "pairs.json"
        data_path.write_text(
            json.dumps(
                [
                    {"seed_smiles": "CCF", "product_smiles": "CCCl"},
                    {
                        "seed_smiles": "c1ccccc1F",
                        "product_smiles": "c1ccccc1Cl",
                    },
                ]
            ),
            encoding="utf-8",
        )
        teacher_embeddings = tmp_path / "teacher_embeddings.json"
        teacher_embeddings.write_text(
            json.dumps({"teacher_embeddings": [[1.0, -1.0, 0.5]]}),
            encoding="utf-8",
        )
        output_path = tmp_path / "mmpt_index.json"

        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "models/mf-generators/mmpt_rag/train.py"),
                "--data",
                str(data_path),
                "--output",
                str(output_path),
                "--kd-teacher-embeddings",
                str(teacher_embeddings),
                "--kd-weight",
                "0.25",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        transforms = json.loads(output_path.read_text(encoding="utf-8"))[
            "transforms"
        ]
        assert len(transforms) == 1
        assert math.isfinite(transforms[0]["kd_alignment_score"])

    def test_generator_rejects_non_finite_kd_projection(self, tmp_path) -> None:
        from mf_generators.mmpt_rag.generator import MMPTRAGGenerator

        artifact_path = tmp_path / "mmpt_index.json"
        artifact_path.write_text(
            json.dumps(
                {
                    "transforms": [
                        {
                            "id": "first",
                            "pattern": "F",
                            "replacement": "Cl",
                            "kd_alignment_score": 0.8,
                            "kd_weight": 0.25,
                        }
                    ],
                    "kd_projection": {
                        "schema_version": "linear_kd_projection.v1",
                        "input_features": [
                            "seed_smiles_length",
                            "product_smiles_length",
                            "pattern_length",
                            "replacement_length",
                        ],
                        "input_dim": 4,
                        "teacher_dim": 2,
                        "feature_mean": [1.0, 1.0, 1.0, 1.0],
                        "feature_scale": [1.0, 1.0, 1.0, 1.0],
                        "weights": [[float("nan")] * 4, [0.0] * 4],
                        "bias": [0.0, 0.0],
                        "regularization": 1.0,
                        "kd_weight": 0.25,
                        "generator_idx": 0,
                    },
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="weights must contain finite values"):
            MMPTRAGGenerator(index_path=str(artifact_path))


class TestUASGenerator:
    def test_autoencoder_forward(self) -> None:
        import torch
        from mf_generators.uas.generator import _Autoencoder

        ae = _Autoencoder(dim=128)
        x = torch.randn(4, 128)
        recon, latent = ae(x)
        assert recon.shape == (4, 128)
        assert latent.shape[0] == 4

    def test_reconstruction_loss(self) -> None:
        import torch
        from mf_generators.uas.generator import _Autoencoder

        ae = _Autoencoder(dim=128)
        x = torch.randn(4, 128)
        loss = ae.reconstruction_loss(x)
        assert loss.shape == (4,)
        assert (loss >= 0).all()

    def test_training_cli_rejects_non_humu_embedding_dimension(self, tmp_path) -> None:
        data_path = tmp_path / "embeddings.jsonl"
        data_path.write_text(
            json.dumps({"id": "mol-1", "embedding": [0.0] * 128}) + "\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "models/mf-generators/uas/train.py"),
                "--data",
                str(data_path),
                "--output-dir",
                str(tmp_path / "uas"),
                "--epochs",
                "1",
                "--batch-size",
                "1",
                "--device",
                "cpu",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert "exactly 129 finite values" in result.stderr

    def test_training_cli_rejects_nonfinite_embedding(self, tmp_path) -> None:
        data_path = tmp_path / "embeddings.jsonl"
        data_path.write_text(
            json.dumps(
                {"id": "mol-1", "embedding": [0.0] * 128 + [float("nan")]}
            )
            + "\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "models/mf-generators/uas/train.py"),
                "--data",
                str(data_path),
                "--output-dir",
                str(tmp_path / "uas"),
                "--epochs",
                "1",
                "--batch-size",
                "1",
                "--device",
                "cpu",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert "exactly 129 finite values" in result.stderr

    @pytest.mark.parametrize("invalid_value", [True, "1.5"])
    def test_training_cli_rejects_non_numeric_embedding_value(
        self,
        tmp_path,
        invalid_value,
    ) -> None:
        data_path = tmp_path / "embeddings.jsonl"
        data_path.write_text(
            json.dumps(
                {"id": "mol-1", "embedding": [0.0] * 128 + [invalid_value]}
            )
            + "\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "models/mf-generators/uas/train.py"),
                "--data",
                str(data_path),
                "--output-dir",
                str(tmp_path / "uas"),
                "--epochs",
                "1",
                "--batch-size",
                "1",
                "--device",
                "cpu",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert "exactly 129 finite values" in result.stderr

    def test_training_cli_writes_autoencoder_and_reference_artifacts(self, tmp_path) -> None:
        import torch

        data_path = tmp_path / "embeddings.jsonl"
        data_path.write_text(
            "\n".join(
                [
                    json.dumps({"id": "mol-1", "embedding": [0.1] * 129}),
                    json.dumps({"id": "mol-2", "embedding": [0.2] * 129}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        output_dir = tmp_path / "uas"
        script = ROOT / "models/mf-generators/uas/train.py"

        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--data",
                str(data_path),
                "--output-dir",
                str(output_dir),
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--device",
                "cpu",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        state = torch.load(output_dir / "autoencoder.pt", map_location="cpu", weights_only=True)
        assert "encoder.0.weight" in state
        reference = torch.load(
            output_dir / "reference_embeddings.pt",
            map_location="cpu",
            weights_only=True,
        )
        assert tuple(reference.shape) == (2, 129)
        manifest = json.loads(
            (output_dir / "training_manifest.json").read_text(encoding="utf-8")
        )
        checkpoint_digest = hashlib.sha256(
            (output_dir / "autoencoder.pt").read_bytes()
        ).hexdigest()
        assert manifest["dim"] == 129
        assert manifest["latent_dim"] == 64
        assert manifest["autoencoder_path"] == "autoencoder.pt"
        assert manifest["reference_embeddings_path"] == "reference_embeddings.pt"
        assert manifest["autoencoder_sha256"] == f"sha256:{checkpoint_digest}"

    def test_training_cli_writes_kd_embedding_loss_metadata(self, tmp_path) -> None:
        data_path = tmp_path / "embeddings.jsonl"
        data_path.write_text(
            "\n".join(
                [
                    json.dumps({"id": "mol-1", "embedding": [0.1] * 129}),
                    json.dumps({"id": "mol-2", "embedding": [0.2] * 129}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        teacher_embeddings = tmp_path / "teacher_embeddings.json"
        teacher_embeddings.write_text(
            json.dumps({"teacher_embeddings": [[0.0] * 64]}),
            encoding="utf-8",
        )
        output_dir = tmp_path / "uas"
        script = ROOT / "models/mf-generators/uas/train.py"

        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--data",
                str(data_path),
                "--output-dir",
                str(output_dir),
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--device",
                "cpu",
                "--kd-teacher-embeddings",
                str(teacher_embeddings),
                "--kd-weight",
                "0.5",
                "--kd-generator-idx",
                "5",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        manifest = json.loads((output_dir / "training_manifest.json").read_text(encoding="utf-8"))
        assert manifest["kd_teacher_embeddings"] == str(teacher_embeddings)
        assert manifest["kd_weight"] == pytest.approx(0.5)
        assert manifest["kd_generator_idx"] == 5

    def test_requires_runner_in_production(self) -> None:
        from mf_generators.uas.generator import UASGenerator

        generator = UASGenerator()

        with pytest.raises(RuntimeError, match="UAS_RUNNER"):
            _run_async_iter(generator.generate(None, None, None, n_samples=1))

    def test_delegates_generation_to_runner(self) -> None:
        from mf_generators.uas.generator import UASGenerator

        class RecordingRunner:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            async def generate(self, **kwargs):
                self.calls.append(kwargs)
                return [
                    {
                        "id": "uas-1",
                        "smiles": "CCO",
                        "canonical_smiles": "CCO",
                        "humu_embedding": [0.0, 1.0],
                    }
                ]

        runner = RecordingRunner()
        generator = UASGenerator(runner=runner)

        molecules = _run_async_iter(
            generator.generate("hciv", "cone", "cig", n_samples=1, seed=7)
        )

        assert molecules[0].smiles == "CCO"
        assert molecules[0].generator_name == "uas"
        assert runner.calls == [
            {
                "hciv": "hciv",
                "cone": "cone",
                "cig": "cig",
                "n_samples": 1,
                "seed": 7,
                "dim": 128,
            }
        ]
