from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def _collect(ait):
    async def _run():
        return [item async for item in ait]

    return asyncio.run(_run())


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_iclm_requires_model_or_runner() -> None:
    from mf_generators.incremental_clm.generator import IncrementalCLMGenerator

    generator = IncrementalCLMGenerator()

    with pytest.raises(RuntimeError, match="IncrementalCLM model or runner is required"):
        asyncio.run(generator.generate(batch_size=1))


def test_iclm_uses_online_learner_ewc_and_packnet() -> None:
    from mf_generators.incremental_clm.generator import IncrementalCLMGenerator

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.calls = []

        def forward(self, **kwargs):
            self.calls.append(kwargs)
            return torch.tensor([[1.0]])

    class Decoder:
        def __call__(self, output, batch_size: int):
            assert output.shape == (1, 1)
            assert batch_size == 1
            return ["CCO"]

    class Learner:
        def __init__(self) -> None:
            self.batch = None

        def update(self, batch):
            self.batch = batch
            return 0.125

    class EWC:
        def ewc_loss(self):
            return torch.tensor(0.25)

    class PackNet:
        def __init__(self) -> None:
            self.applied = False

        def apply_mask(self):
            self.applied = True

    model = Model()
    learner = Learner()
    packnet = PackNet()
    generator = IncrementalCLMGenerator(
        model=model,
        decoder=Decoder(),
        online_learner=learner,
        ewc_regularizer=EWC(),
        packnet=packnet,
    )

    molecules = asyncio.run(generator.generate(batch_size=1, online_batch={"smiles": ["CCO"]}))

    assert [mol.smiles for mol in molecules] == ["CCO"]
    assert learner.batch == {"smiles": ["CCO"]}
    assert packnet.applied is True
    assert molecules[0].metadata["ewc_loss"] == "0.25"


def test_iclm_online_learner_adds_teacher_embedding_kd_loss() -> None:
    from mf_generators.incremental_clm.learning.online_learner import OnlineLearner

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.5))

        def forward(self, batch):
            assert batch["training_samples"] == ["CCO"]
            task_loss = self.weight.pow(2)
            student_embedding = self.weight.reshape(1, 1) * 4.0
            return task_loss, student_embedding

    learner = OnlineLearner(Model(), learning_rate=0.0)

    loss = learner.update(
        {
            "training_samples": ["CCO"],
            "kd_teacher_embeddings": [[0.0]],
            "kd_weight": 0.5,
        }
    )

    assert loss == pytest.approx(2.25)
    assert learner.last_kd_loss == pytest.approx(4.0)


def test_uas_requires_candidate_source_reference_and_decoder() -> None:
    from mf_generators.uas.generator import UASGenerator

    generator = UASGenerator(dim=2)

    with pytest.raises(
        RuntimeError,
        match="candidate_source, reference_embeddings, and decoder are required",
    ):
        _collect(generator.generate(None, None, None, n_samples=1))


def test_uas_filters_candidates_by_unfamiliarity() -> None:
    from mf_generators.uas.generator import UASGenerator

    calls = []

    def candidate_source(n_samples: int):
        calls.append(n_samples)
        return torch.tensor([[0.0, 0.0], [4.0, 4.0]], dtype=torch.float32)

    def decoder(embeddings: torch.Tensor):
        assert embeddings.shape == (1, 2)
        return ["CCO"]

    generator = UASGenerator(
        dim=2,
        candidate_source=candidate_source,
        reference_embeddings=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        decoder=decoder,
        unfamiliarity_threshold=0.5,
    )

    molecules = _collect(generator.generate(None, None, None, n_samples=1))

    assert calls == [1]
    assert len(molecules) == 1
    assert molecules[0].smiles == "CCO"
    assert molecules[0].generator_name == "uas"
    assert molecules[0].properties["uas_safety_probability"] == pytest.approx(0.622459, rel=1e-5)


def test_crem_generate_uses_fragment_replacement(tmp_path) -> None:
    from mf_generators.crem_3d.generator import CReM3DGenerator

    mmp_path = tmp_path / "crem_mmp.json"
    mmp_path.write_text(
        json.dumps(
            {
                "mutations": [
                    {
                        "id": "ethyl_fluoro",
                        "seed_smiles": "CC",
                        "fragment_smiles": "F",
                        "attachment_index": 0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    generator = CReM3DGenerator(mmp_db_path=str(mmp_path))

    molecules = asyncio.run(generator.generate(batch_size=1, seed_smiles="CC"))

    assert molecules[0].smiles == "CCF"
    assert molecules[0].metadata["mutation_id"] == "ethyl_fluoro"
    assert molecules[0].metadata["fragment_replacement"] == "true"


def test_fragfm_uses_vocabulary_and_sa_rate_matrix(tmp_path) -> None:
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
    generator = FragFMGenerator(vocab_path=str(vocab_path))

    molecules = asyncio.run(generator.generate(batch_size=1))

    assert molecules[0].smiles == "CCO"
    assert molecules[0].metadata["rate_matrix_applied"] == "true"
    assert molecules[0].metadata["fragment_indices"] == "0,1"


# W12: CReM-pharm-3D scorer quality tests


def test_crem_pharmacophore_scorer_ranks_molecules_by_score(tmp_path) -> None:
    from mf_generators.crem_3d.generator import CReM3DGenerator

    mmp_path = tmp_path / "crem_mmp.json"
    mmp_path.write_text(
        json.dumps(
            {
                "mutations": [
                    {"id": "high", "seed_smiles": "CC", "product": "CCO"},
                    {"id": "low", "seed_smiles": "CC", "product": "CCN"},
                ]
            }
        ),
        encoding="utf-8",
    )

    class MockPharmacophoreScorer:
        async def score_batch(self, smiles_list, *, intent_cone=None):
            scores = {"CCO": 0.9, "CCN": 0.3}
            return {smiles: {"pharmacophore_score": scores.get(smiles, 0.5)} for smiles in smiles_list}

    generator = CReM3DGenerator(
        mmp_db_path=str(mmp_path),
        pharmacophore_scorer=MockPharmacophoreScorer(),
    )
    molecules = asyncio.run(generator.generate(batch_size=2, seed_smiles="CC"))

    assert len(molecules) == 2
    assert molecules[0].smiles == "CCO"
    assert float(molecules[0].metadata["pharmacophore_score"]) == pytest.approx(0.9)
    assert float(molecules[1].metadata["pharmacophore_score"]) == pytest.approx(0.3)


def test_crem_humu_scorer_ranks_by_alignment_and_stores_embedding(tmp_path) -> None:
    from mf_core.types.humu import IntentCone
    from mf_generators.crem_3d.generator import CReM3DGenerator

    mmp_path = tmp_path / "crem_mmp.json"
    mmp_path.write_text(
        json.dumps(
            {
                "mutations": [
                    {"id": "aligned", "seed_smiles": "CC", "product": "CCO"},
                    {"id": "misaligned", "seed_smiles": "CC", "product": "CCN"},
                ]
            }
        ),
        encoding="utf-8",
    )

    class MockHumuScorer:
        async def score_batch(self, smiles_list, *, intent_cone=None):
            embeddings = {"CCO": [0.8] + [0.0] * 127, "CCN": [0.1] + [0.0] * 127}
            return {
                smiles: {
                    "humu_embedding": embeddings.get(smiles, [0.0] * 128),
                    "humu_alignment_score": embeddings.get(smiles, [0.0])[0],
                }
                for smiles in smiles_list
            }

    generator = CReM3DGenerator(
        mmp_db_path=str(mmp_path),
        humu_embedding_scorer=MockHumuScorer(),
    )
    intent_cone = IntentCone(axis=[1.0] + [0.0] * 128, half_angle=0.5)
    molecules = asyncio.run(
        generator.generate(batch_size=2, seed_smiles="CC", intent_cone=intent_cone)
    )

    assert len(molecules) == 2
    assert molecules[0].smiles == "CCO"
    assert molecules[0].humu_embedding is not None
    assert float(molecules[0].metadata["humu_alignment_score"]) == pytest.approx(0.8)


def test_crem_humu_scorer_wrapper_scores_embeddings_with_intent_cone() -> None:
    import math

    module = _load_module(
        "crem_humu_scorer_wrapper_test",
        ROOT / "tools/scorers/crem_humu_scorer.py",
    )

    class Encoder:
        def encode(self, smiles: str) -> list[float]:
            if smiles == "CCO":
                return [math.sqrt(1.0 + 0.8 * 0.8), 0.8] + [0.0] * 127
            return [math.sqrt(1.0 + 0.1 * 0.1), 0.1] + [0.0] * 127

    records = module.score_humu_records(
        ["CCO", "CCN"],
        encoder=Encoder(),
        intent_cone={"axis": [0.0, 1.0] + [0.0] * 127, "half_angle": 0.5},
    )

    assert sorted(records) == ["CCN", "CCO"]
    assert len(records["CCO"]["humu_embedding"]) == 129
    assert records["CCO"]["humu_alignment_score"] > records["CCN"]["humu_alignment_score"]


def test_crem_pharmacophore_scorer_wrapper_uses_reference_similarity(tmp_path) -> None:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    module = _load_module(
        "crem_pharmacophore_scorer_wrapper_test",
        ROOT / "tools/scorers/crem_pharmacophore_scorer.py",
    )

    reference = tmp_path / "reference.sdf"
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert AllChem.EmbedMolecule(mol, randomSeed=61453) == 0
    AllChem.UFFOptimizeMolecule(mol, maxIters=50)
    writer = Chem.SDWriter(str(reference))
    try:
        writer.write(mol)
    finally:
        writer.close()

    records = module.score_pharmacophore_records(
        ["CCO", "c1ccccc1"],
        reference_sdf=str(reference),
    )

    assert records["CCO"]["pharmacophore_score"] > 0.0
    assert (
        records["CCO"]["pharmacophore_score"]
        > records["c1ccccc1"]["pharmacophore_score"]
    )
    assert records["CCO"]["pharmacophore_reference"].endswith("reference.sdf")


@pytest.mark.asyncio
async def test_crem_dock_oracle_grpc_scorer_batch_calls_oracle_service() -> None:
    from unittest.mock import AsyncMock, MagicMock, patch

    from mf_core.proto_gen.moleculeforge.v1.oracle import oracle_pb2
    from mf_generators.crem_3d.generator import DockOracleGrpcScorer

    mock_evaluation = MagicMock()
    mock_evaluation.success = True
    mock_evaluation.molecule_smiles = "CCO"
    mock_evaluation.oracle_name = "diffdock_l"
    mock_evaluation.scores = {"docking_score": -7.5}
    mock_evaluation.error_message = ""

    mock_response = MagicMock()
    mock_response.evaluations = [mock_evaluation]

    mock_stub = AsyncMock()
    mock_stub.Evaluate = AsyncMock(return_value=mock_response)

    scorer = DockOracleGrpcScorer(stub=mock_stub)
    records = await scorer.score_batch(["CCO"])

    assert "CCO" in records
    assert float(records["CCO"]["docking_score"]) == pytest.approx(-7.5)
    assert records["CCO"]["oracle_name"] == "diffdock_l"
