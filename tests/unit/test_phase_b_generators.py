from __future__ import annotations

import asyncio
import json

import pytest
import torch


def _collect(ait):
    async def _run():
        return [item async for item in ait]

    return asyncio.run(_run())


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
