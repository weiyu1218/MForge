"""Unit tests for Layer 3 generators — basic generation validation."""

from __future__ import annotations

import asyncio
import importlib.util
import json
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
        assert {"epoch", "loss", "flow_model", "decoder", "optimizer"} <= set(checkpoint)
        assert (output_dir / "final_model.pt").is_file()

    def test_training_cli_rejects_empty_data_dir(self, tmp_path) -> None:
        data_dir = tmp_path / "empty"
        data_dir.mkdir()
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


class TestMMPTRAGGenerator:
    def test_generates_valid_smiles(self) -> None:
        from mf_generators.mmpt_rag.generator import MMPTRAGGenerator

        gen = MMPTRAGGenerator()
        hciv, cone = _make_hciv_cone()
        cig = _make_cig()

        mols = _run_async_iter(gen.generate(hciv, cone, cig, n_samples=5, seed=42))
        assert len(mols) > 0
        for mol in mols:
            assert mol.smiles
            rdkit_mol = Chem.MolFromSmiles(mol.smiles)
            assert rdkit_mol is not None, f"Invalid SMILES: {mol.smiles}"

    def test_rdkit_substructure_replace(self) -> None:
        from mf_generators.mmpt_rag.generator import MMPTRAGGenerator

        gen = MMPTRAGGenerator()
        result = gen._simple_replace("c1ccccc1", "c1ccccc1", "c1ccncc1")
        assert result is not None
        mol = Chem.MolFromSmiles(result)
        assert mol is not None


class TestEvomolRLGenerator:
    def test_hypervolume_improvement(self) -> None:
        from mf_generators.evomol_rl.hypervolume import compute_hypervolume_improvement

        new_point = {"qed": 0.8, "sa": 0.7}
        front = [{"qed": 0.6, "sa": 0.5}, {"qed": 0.7, "sa": 0.4}]
        ref = {"qed": 0.0, "sa": 0.0}

        hvi = compute_hypervolume_improvement(new_point, front, ref)
        assert hvi >= 0.0

    def test_non_dominated_filter(self) -> None:
        import numpy as np
        from mf_generators.evomol_rl.hypervolume import _filter_non_dominated

        # Points: [1,1] is dominated by [2,2]; [3.0, 1.0] and [2,2] are mutually non-dominating
        points = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 1.0]])
        filtered = _filter_non_dominated(points)
        # [2,2] and [3.0, 1.0] should remain (2 non-dominated points)
        assert len(filtered) == 2

    def test_requires_runner_in_production(self) -> None:
        from mf_generators.evomol_rl.generator import EvoMolRLGenerator

        generator = EvoMolRLGenerator()

        with pytest.raises(RuntimeError, match="EVOMOL_RUNNER"):
            asyncio.run(generator.generate(batch_size=1))

    def test_delegates_generation_to_runner(self) -> None:
        from mf_generators.evomol_rl.generator import EvoMolRLGenerator

        class RecordingRunner:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            async def generate(self, **kwargs):
                self.calls.append(kwargs)
                return [
                    {
                        "smiles": "CCO",
                        "metadata": {"oracle_name": "fixture_oracle"},
                    }
                ]

        runner = RecordingRunner()
        generator = EvoMolRLGenerator(
            checkpoint_path="evomol.pt",
            device="cpu",
            runner=runner,
        )

        molecules = asyncio.run(generator.generate(batch_size=1, stage="lead_opt"))

        assert molecules[0].smiles == "CCO"
        assert molecules[0].metadata["oracle_name"] == "fixture_oracle"
        assert runner.calls == [
            {
                "batch_size": 1,
                "intent_cone": None,
                "checkpoint_path": "evomol.pt",
                "device": "cpu",
                "stage": "lead_opt",
            }
        ]


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
        generator = CReM3DGenerator(mmp_db_path=str(output_path))
        molecules = asyncio.run(generator.generate(batch_size=1, seed_smiles="c1ccccc1"))
        assert molecules[0].smiles == "Fc1ccccc1"


class TestFragFMGenerator:
    def test_requires_vocabulary_artifact_in_production(self) -> None:
        from mf_generators.fragfm.generator import FragFMGenerator

        with pytest.raises(RuntimeError, match="vocabulary artifact"):
            FragFMGenerator()

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
        )

        assert generator._model is not None
        loaded_rate = float(generator.rate_matrix.base_rate[0, 0].detach().cpu().item())
        assert loaded_rate == pytest.approx(0.5)

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
        state = torch.load(output_dir / "best_model.pt", map_location="cpu")
        assert "fragment_encoder.weight" in state
        assert (output_dir / "final_model.pt").is_file()


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
        generator = MMPTRAGGenerator(index_path=str(output_path))
        mols = _run_async_iter(generator.generate(None, None, None, n_samples=1))
        assert mols[0].smiles == "c1ccccc1Cl"


class _RecordingLaMGenRunner:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        return [{"smiles": "CCO", "metadata": {"decode_artifact": "lamgen-decoder"}}]


class TestLaMGen3DGenerator:
    def test_requires_runner_in_production(self) -> None:
        from mf_generators.lamgen_3d.generator import LaMGen3DGenerator

        generator = LaMGen3DGenerator(checkpoint_path="checkpoint.pt")

        with pytest.raises(RuntimeError, match="LAMGEN_RUNNER"):
            asyncio.run(generator.generate(batch_size=1))

    def test_delegates_generation_to_runner(self) -> None:
        from mf_generators.lamgen_3d.generator import LaMGen3DGenerator

        runner = _RecordingLaMGenRunner()
        generator = LaMGen3DGenerator(
            checkpoint_path="checkpoint.pt",
            device="cpu",
            runner=runner,
        )

        molecules = asyncio.run(generator.generate(batch_size=1, temperature=0.2))

        assert molecules[0].smiles == "CCO"
        assert molecules[0].metadata["decode_artifact"] == "lamgen-decoder"
        assert runner.calls[0]["batch_size"] == 1
        assert runner.calls[0]["checkpoint_path"] == "checkpoint.pt"
        assert runner.calls[0]["temperature"] == 0.2


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

    def test_training_cli_writes_autoencoder_and_reference_artifacts(self, tmp_path) -> None:
        import torch

        data_path = tmp_path / "embeddings.jsonl"
        data_path.write_text(
            "\n".join(
                [
                    json.dumps({"id": "mol-1", "embedding": [0.1, 0.2, 0.3, 0.4]}),
                    json.dumps({"id": "mol-2", "embedding": [0.2, 0.3, 0.4, 0.5]}),
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
        state = torch.load(output_dir / "autoencoder.pt", map_location="cpu")
        assert "encoder.0.weight" in state
        reference = torch.load(output_dir / "reference_embeddings.pt", map_location="cpu")
        assert tuple(reference.shape) == (2, 4)

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
