from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
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


def _write_tiny_huggingface_clm(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    pad_marker = "[PAD]"
    eos_marker = "[EOS]"
    unknown_marker = "[UNK]"
    tokenizer_backend = Tokenizer(
        WordLevel(
            vocab={
                pad_marker: 0,
                eos_marker: 1,
                unknown_marker: 2,
                "CCO": 3,
                "CCN": 4,
            },
            unk_token=unknown_marker,
        )
    )
    tokenizer_backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        pad_token=pad_marker,
        eos_token=eos_marker,
        unk_token=unknown_marker,
    )
    torch.manual_seed(7)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=16,
            n_ctx=16,
            n_embd=4,
            n_layer=1,
            n_head=1,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            bos_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    )
    checkpoint_path.mkdir(parents=True)
    tokenizer.save_pretrained(checkpoint_path)
    model.save_pretrained(checkpoint_path)
    (checkpoint_path / "moleculeforge_ewc_replay.json").write_text(
        json.dumps(
            {
                "schema_version": "iclm-ewc-replay.v1",
                "dataset_id": "tiny-smiles-calibration-v1",
                "samples": [
                    {"smiles": "CCO", "weight": 1.0},
                    {"smiles": "CCN", "weight": 1.0},
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }


def _tiny_smiles_nll(checkpoint_path: str | Path, smiles: str) -> float:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    encoded = tokenizer(
        [f"{smiles} {tokenizer.eos_token}"],
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**encoded, return_dict=True).logits[:, :-1, :]
    labels = encoded["input_ids"][:, 1:]
    return float(
        torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
        ).item()
    )


def test_iclm_requires_model_or_runner() -> None:
    from mf_generators.incremental_clm.generator import IncrementalCLMGenerator

    generator = IncrementalCLMGenerator()

    with pytest.raises(RuntimeError, match="IncrementalCLM model or runner is required"):
        asyncio.run(generator.generate(batch_size=1))


def test_iclm_rejects_simultaneous_ewc_and_packnet_strategies() -> None:
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

    with pytest.raises(ValueError, match="mutually exclusive"):
        IncrementalCLMGenerator(
            model=Model(),
            decoder=Decoder(),
            online_learner=Learner(),
            ewc_regularizer=EWC(),
            packnet=PackNet(),
        )


def test_iclm_generation_rejects_online_update_bypass() -> None:
    from mf_generators.incremental_clm.generator import IncrementalCLMGenerator

    class Runner:
        @staticmethod
        def generate(**kwargs):
            raise AssertionError("generation must not start after an online update request")

    generator = IncrementalCLMGenerator(runner=Runner())

    with pytest.raises(RuntimeError, match="UpdateModel"):
        asyncio.run(
            generator.generate(
                batch_size=1,
                online_batch={"samples": [{"smiles": "CCO"}]},
            )
        )


def test_iclm_online_learner_writes_service_checkpoint_result(tmp_path: Path) -> None:
    from mf_generators.incremental_clm.learning.online_learner import OnlineLearner

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.5))

        def forward(self, batch):
            assert batch["samples"] == [{"smiles": "CCO"}]
            return self.weight.pow(2)

    learner = OnlineLearner(
        Model(),
        checkpoint_directory=tmp_path,
        learning_rate=0.0,
    )

    result = learner.update(
        {
            "samples": [{"smiles": "CCO"}],
            "target_checkpoint_version": "iclm-v2",
        }
    )

    checkpoint_path = Path(result["checkpoint_path"])
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    assert result["updated_samples"] == 1
    assert checkpoint_path.is_file()
    assert checkpoint["target_checkpoint_version"] == "iclm-v2"
    assert checkpoint["model_state_dict"]["weight"].item() == pytest.approx(0.5)
    assert checkpoint["optimizer_state_dict"] == learner.optimizer.state_dict()
    assert learner.last_total_loss == pytest.approx(0.25)


def test_iclm_online_learner_rejects_unaligned_teacher_embeddings(
    tmp_path: Path,
) -> None:
    from mf_generators.incremental_clm.learning.online_learner import OnlineLearner

    model = torch.nn.Linear(1, 1, bias=False)
    learner = OnlineLearner(model, checkpoint_directory=tmp_path)

    with pytest.raises(ValueError, match="HuggingFaceCausalLMRunner"):
        learner.update(
            {
                "samples": [{"smiles": "CCO"}],
                "teacher_embeddings": [[0.0]],
                "target_checkpoint_version": "iclm-v2",
            }
        )


def test_iclm_ewc_consolidates_task_fisher_and_restores_state() -> None:
    from mf_generators.incremental_clm.model import EWCRegularizer

    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    regularizer = EWCRegularizer(model)

    regularizer.consolidate((model.weight - 3.0).pow(2).sum())
    state = regularizer.state_dict()
    with torch.no_grad():
        model.weight.fill_(2.0)

    restored = EWCRegularizer(model)
    restored.load_state_dict(state)

    assert state["fisher_diag"]["weight"].item() == pytest.approx(16.0)
    assert state["optimal_params"]["weight"].item() == pytest.approx(1.0)
    assert restored.ewc_loss().item() == pytest.approx(8.0)


def test_iclm_ewc_uses_weighted_per_sample_fisher_without_gradient_cancellation() -> None:
    from mf_generators.incremental_clm.model import EWCRegularizer

    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    task_losses = torch.stack(
        (
            (model.weight - 1.0).pow(2).sum(),
            (model.weight + 1.0).pow(2).sum(),
        )
    )

    regularizer = EWCRegularizer(model)
    regularizer.consolidate(
        task_losses,
        sample_weights=torch.tensor([0.25, 0.75]),
    )

    assert regularizer.fisher_diag["weight"].item() == pytest.approx(2.0)


def test_iclm_ewc_preserves_absolute_sample_weight_strength() -> None:
    from mf_generators.incremental_clm.model import EWCRegularizer

    low_model = torch.nn.Linear(1, 1, bias=False)
    high_model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        low_model.weight.zero_()
        high_model.weight.zero_()
    low_regularizer = EWCRegularizer(low_model)
    high_regularizer = EWCRegularizer(high_model)
    low_regularizer.consolidate(
        (low_model.weight - 1.0).pow(2).reshape(1),
        sample_weights=torch.tensor([0.1]),
    )
    high_regularizer.consolidate(
        (high_model.weight - 1.0).pow(2).reshape(1),
        sample_weights=torch.tensor([0.9]),
    )

    assert low_regularizer.fisher_diag["weight"].item() == pytest.approx(0.4)
    assert high_regularizer.fisher_diag["weight"].item() == pytest.approx(3.6)


def test_iclm_packnet_freezes_allocated_weights_and_restores_mask() -> None:
    from mf_generators.incremental_clm.model import PackNet

    model = torch.nn.Linear(4, 1, bias=True)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.1, 0.2, 3.0, 4.0]]))
        model.bias.fill_(0.75)
    packnet = PackNet(model, prune_ratio=0.5)
    packnet.prune()
    initial_mask = packnet.masks["weight"].clone()
    allocated_before = model.weight.detach().clone()
    bias_before = model.bias.detach().clone()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    frozen_parameters = packnet.capture_allocated_parameters()
    optimizer.zero_grad()
    model(torch.ones(1, 4)).sum().backward()
    packnet.mask_gradients()
    optimizer.step()
    packnet.restore_allocated_parameters(frozen_parameters)

    assert torch.equal(initial_mask, torch.tensor([[0.0, 0.0, 1.0, 1.0]]))
    assert torch.equal(
        model.weight.detach()[initial_mask.bool()],
        allocated_before[initial_mask.bool()],
    )
    assert torch.count_nonzero(model.weight.detach()[~initial_mask.bool()]) == 2
    assert torch.equal(packnet.masks["bias"], torch.ones_like(model.bias))
    assert torch.equal(model.bias.detach(), bias_before)

    packnet.allocate()
    state = packnet.state_dict()
    restored_model = torch.nn.Linear(4, 1, bias=True)
    restored_model.load_state_dict(model.state_dict())
    restored = PackNet(restored_model, prune_ratio=0.5)
    restored.load_state_dict(state)

    assert torch.equal(restored.masks["weight"], packnet.masks["weight"])
    assert torch.equal(restored_model.weight, model.weight)


def test_iclm_default_huggingface_update_applies_teacher_weighted_loss(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from transformers import AutoModelForCausalLM

    model_path = tmp_path / "active"
    initial_parameters = _write_tiny_huggingface_clm(model_path)
    checkpoint_directory = tmp_path / "checkpoints"
    monkeypatch.setenv("ICLM_MODEL_PATH", str(model_path))
    monkeypatch.setenv("ICLM_CHECKPOINT_DIRECTORY", str(checkpoint_directory))
    monkeypatch.delenv("ICLM_UPDATE_COMMAND", raising=False)
    service = _load_module(
        "iclm_default_hf_update_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    generator = service._build_generator()

    result = asyncio.run(
        service._run_update(
            {
                "schema_version": "training-batch.v1",
                "samples": [{"smiles": "CCO"}, {"smiles": "CCN"}],
                "kd_weight": 0.5,
                "run_id": "run-iclm",
                "request_id": "request-iclm",
                "kd_teacher_embeddings": [
                    [0.5, -0.5, 0.25, -0.25],
                    [-0.25, 0.25, -0.5, 0.5],
                ],
                "teacher_source": "hypseek",
                "teacher_version": "teacher-v1",
                "target_checkpoint_version": "iclm-v2",
            },
            generator,
        )
    )

    checkpoint_path = Path(result["checkpoint_path"])
    updated_model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    updated_parameters = dict(updated_model.named_parameters())
    assert checkpoint_path == checkpoint_directory / "iclm-v2"
    assert checkpoint_path.is_dir()
    assert result["updated_samples"] == 2
    assert result["teacher_loss"] > 0.0
    assert result["distillation_loss"] >= 0.0
    assert result["ewc_loss"] > 0.0
    assert result["total_loss"] == pytest.approx(
        0.5 * result["teacher_loss"]
        + 0.5 * result["distillation_loss"]
        + result["ewc_loss"]
    )
    assert result["effective_learning_rate"] == pytest.approx(1e-4)
    assert any(
        not torch.equal(initial_parameters[name], updated_parameters[name])
        for name in initial_parameters
    )


def test_iclm_huggingface_teacher_embeddings_change_parameter_update(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mf_generators.incremental_clm.hf_runner import HuggingFaceCausalLMRunner
    from transformers import AutoModelForCausalLM

    model_path = tmp_path / "active"
    _write_tiny_huggingface_clm(model_path)
    monkeypatch.setenv("ICLM_CHECKPOINT_DIRECTORY", str(tmp_path / "checkpoints"))
    base_payload = {
        "samples": [{"smiles": "CCO"}],
        "kd_weight": 0.5,
    }
    torch.manual_seed(23)
    first_result = HuggingFaceCausalLMRunner(model_path=str(model_path)).update(
        {
            **base_payload,
            "kd_teacher_embeddings": [[10.0, -10.0, 10.0, -10.0]],
            "target_checkpoint_version": "teacher-a",
        }
    )
    torch.manual_seed(23)
    second_result = HuggingFaceCausalLMRunner(model_path=str(model_path)).update(
        {
            **base_payload,
            "kd_teacher_embeddings": [[-10.0, 10.0, -10.0, 10.0]],
            "target_checkpoint_version": "teacher-b",
        }
    )
    first_parameters = dict(
        AutoModelForCausalLM.from_pretrained(
            first_result["checkpoint_path"]
        ).named_parameters()
    )
    second_parameters = dict(
        AutoModelForCausalLM.from_pretrained(
            second_result["checkpoint_path"]
        ).named_parameters()
    )

    assert any(
        not torch.equal(first_parameters[name], second_parameters[name])
        for name in first_parameters
    )


def test_iclm_huggingface_zero_teacher_weight_preserves_task_update(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mf_generators.incremental_clm.hf_runner import HuggingFaceCausalLMRunner
    from transformers import AutoModelForCausalLM

    model_path = tmp_path / "active"
    initial_parameters = _write_tiny_huggingface_clm(model_path)
    monkeypatch.setenv("ICLM_CHECKPOINT_DIRECTORY", str(tmp_path / "checkpoints"))
    base_payload = {
        "samples": [{"smiles": "CCO"}],
        "teacher_weight": 0.0,
    }
    torch.manual_seed(29)
    task_result = HuggingFaceCausalLMRunner(model_path=str(model_path)).update(
        {
            **base_payload,
            "target_checkpoint_version": "task-only",
        }
    )
    torch.manual_seed(29)
    disabled_kd_result = HuggingFaceCausalLMRunner(model_path=str(model_path)).update(
        {
            **base_payload,
            "teacher_embeddings": [[10.0, -10.0, 10.0, -10.0]],
            "target_checkpoint_version": "disabled-kd",
        }
    )
    task_parameters = dict(
        AutoModelForCausalLM.from_pretrained(
            task_result["checkpoint_path"]
        ).named_parameters()
    )
    disabled_kd_parameters = dict(
        AutoModelForCausalLM.from_pretrained(
            disabled_kd_result["checkpoint_path"]
        ).named_parameters()
    )

    assert all(
        torch.equal(task_parameters[name], disabled_kd_parameters[name])
        for name in task_parameters
    )
    assert task_result["total_loss"] == pytest.approx(
        task_result["teacher_loss"] + task_result["ewc_loss"]
    )
    assert any(
        not torch.equal(initial_parameters[name], task_parameters[name])
        for name in initial_parameters
    )


def test_iclm_huggingface_teacher_rewards_change_parameter_update(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mf_generators.incremental_clm.hf_runner import HuggingFaceCausalLMRunner
    from transformers import AutoModelForCausalLM

    model_path = tmp_path / "active"
    _write_tiny_huggingface_clm(model_path)
    monkeypatch.setenv("ICLM_CHECKPOINT_DIRECTORY", str(tmp_path / "checkpoints"))
    base_payload = {
        "kd_weight": 0.5,
        "kd_teacher_embeddings": [
            [0.5, -0.5, 0.25, -0.25],
            [-0.25, 0.25, -0.5, 0.5],
        ],
    }
    torch.manual_seed(29)
    first_result = HuggingFaceCausalLMRunner(model_path=str(model_path)).update(
        {
            **base_payload,
            "samples": [
                {"smiles": "CCO", "reward": 1.0},
                {"smiles": "CCN", "reward": 0.0},
            ],
            "target_checkpoint_version": "reward-a",
        }
    )
    torch.manual_seed(29)
    second_result = HuggingFaceCausalLMRunner(model_path=str(model_path)).update(
        {
            **base_payload,
            "samples": [
                {"smiles": "CCO", "reward": 0.0},
                {"smiles": "CCN", "reward": 1.0},
            ],
            "target_checkpoint_version": "reward-b",
        }
    )
    first_parameters = dict(
        AutoModelForCausalLM.from_pretrained(
            first_result["checkpoint_path"]
        ).named_parameters()
    )
    second_parameters = dict(
        AutoModelForCausalLM.from_pretrained(
            second_result["checkpoint_path"]
        ).named_parameters()
    )

    assert any(
        not torch.equal(first_parameters[name], second_parameters[name])
        for name in first_parameters
    )


def test_iclm_huggingface_single_sample_reward_controls_update_magnitude(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mf_generators.incremental_clm.hf_runner import HuggingFaceCausalLMRunner
    from transformers import AutoModelForCausalLM

    model_path = tmp_path / "active"
    initial_parameters = _write_tiny_huggingface_clm(model_path)
    monkeypatch.setenv("ICLM_CHECKPOINT_DIRECTORY", str(tmp_path / "checkpoints"))
    base_payload = {
        "kd_weight": 0.5,
        "kd_teacher_embeddings": [[0.5, -0.5, 0.25, -0.25]],
    }
    torch.manual_seed(31)
    low_result = HuggingFaceCausalLMRunner(model_path=str(model_path)).update(
        {
            **base_payload,
            "samples": [{"smiles": "CCO", "reward": 0.1}],
            "target_checkpoint_version": "reward-low",
        }
    )
    torch.manual_seed(31)
    high_result = HuggingFaceCausalLMRunner(model_path=str(model_path)).update(
        {
            **base_payload,
            "samples": [{"smiles": "CCO", "reward": 0.9}],
            "target_checkpoint_version": "reward-high",
        }
    )
    low_parameters = dict(
        AutoModelForCausalLM.from_pretrained(low_result["checkpoint_path"]).named_parameters()
    )
    high_parameters = dict(
        AutoModelForCausalLM.from_pretrained(high_result["checkpoint_path"]).named_parameters()
    )
    low_delta = sum(
        torch.linalg.vector_norm(low_parameters[name] - initial).item()
        for name, initial in initial_parameters.items()
    )
    high_delta = sum(
        torch.linalg.vector_norm(high_parameters[name] - initial).item()
        for name, initial in initial_parameters.items()
    )

    assert high_delta > low_delta


@pytest.mark.parametrize(
    ("outcome", "reward", "expected_direction"),
    [("PASS", 1.0, "lower"), ("FAIL", 0.0, "higher")],
)
def test_iclm_huggingface_outcome_controls_likelihood_direction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    outcome: str,
    reward: float,
    expected_direction: str,
) -> None:
    from mf_generators.incremental_clm.hf_runner import HuggingFaceCausalLMRunner

    model_path = tmp_path / "active"
    _write_tiny_huggingface_clm(model_path)
    monkeypatch.setenv("ICLM_CHECKPOINT_DIRECTORY", str(tmp_path / "checkpoints"))
    before = _tiny_smiles_nll(model_path, "CCO")

    result = HuggingFaceCausalLMRunner(
        model_path=str(model_path),
        ewc_weight=0.0,
        learning_rate=1e-3,
    ).update(
        {
            "samples": [
                {
                    "smiles": "CCO",
                    "reward": reward,
                    "outcome": outcome,
                }
            ],
            "teacher_weight": 1.0,
            "target_checkpoint_version": f"direction-{outcome.lower()}",
        }
    )
    after = _tiny_smiles_nll(result["checkpoint_path"], "CCO")

    if expected_direction == "lower":
        assert after < before
    else:
        assert after > before


def test_iclm_huggingface_restores_ewc_and_changes_next_update(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mf_generators.incremental_clm.hf_runner import (
        _CONTINUAL_STATE_FILE,
        HuggingFaceCausalLMRunner,
    )
    from transformers import AutoModelForCausalLM

    model_path = tmp_path / "active"
    _write_tiny_huggingface_clm(model_path)
    monkeypatch.setenv("ICLM_CHECKPOINT_DIRECTORY", str(tmp_path / "checkpoints"))
    first_payload = {
        "samples": [{"smiles": "CCO"}],
        "kd_weight": 0.5,
        "kd_teacher_embeddings": [[0.5, -0.5, 0.25, -0.25]],
        "target_checkpoint_version": "first",
    }
    first_result = HuggingFaceCausalLMRunner(model_path=str(model_path)).update(
        first_payload
    )
    first_checkpoint = Path(first_result["checkpoint_path"])
    first_model = AutoModelForCausalLM.from_pretrained(first_checkpoint)
    continual_state = torch.load(
        first_checkpoint / _CONTINUAL_STATE_FILE,
        map_location="cpu",
        weights_only=True,
    )
    assert any(
        torch.count_nonzero(fisher) > 0
        for fisher in continual_state["ewc"]["fisher_diag"].values()
    )
    base_calibration = continual_state["base_calibration"]
    assert base_calibration["schema_version"] == "iclm-ewc-calibration.v1"
    assert base_calibration["source"] == "versioned_replay"
    assert base_calibration["dataset_id"] == "tiny-smiles-calibration-v1"
    assert base_calibration["sample_count"] == 2
    assert base_calibration["replay_checksum"].startswith("sha256:")
    for name, parameter in first_model.named_parameters():
        assert torch.equal(
            continual_state["ewc"]["optimal_params"][name],
            parameter.detach().cpu(),
        )
    next_payload = {
        "samples": [{"smiles": "CCN"}],
        "kd_weight": 0.5,
        "kd_teacher_embeddings": [[-0.5, 0.5, -0.25, 0.25]],
    }
    with_ewc = HuggingFaceCausalLMRunner(
        model_path=first_result["checkpoint_path"],
        ewc_weight=10.0,
    ).update(
        {
            **next_payload,
            "target_checkpoint_version": "with-ewc",
        }
    )
    without_ewc = HuggingFaceCausalLMRunner(
        model_path=first_result["checkpoint_path"],
        ewc_weight=0.0,
    ).update(
        {
            **next_payload,
            "target_checkpoint_version": "without-ewc",
        }
    )
    with_ewc_parameters = dict(
        AutoModelForCausalLM.from_pretrained(
            with_ewc["checkpoint_path"]
        ).named_parameters()
    )
    without_ewc_parameters = dict(
        AutoModelForCausalLM.from_pretrained(
            without_ewc["checkpoint_path"]
        ).named_parameters()
    )

    assert with_ewc["ewc_loss"] > 0.0
    assert with_ewc["total_loss"] == pytest.approx(
        0.5 * with_ewc["teacher_loss"]
        + 0.5 * with_ewc["distillation_loss"]
        + 10.0 * with_ewc["ewc_loss"]
    )
    assert any(
        not torch.equal(with_ewc_parameters[name], without_ewc_parameters[name])
        for name in with_ewc_parameters
    )


def test_iclm_huggingface_persists_one_ewc_strategy_across_restart(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mf_generators.incremental_clm.hf_runner import (
        _CONTINUAL_STATE_FILE,
        HuggingFaceCausalLMRunner,
    )
    from transformers import AutoModelForCausalLM

    model_path = tmp_path / "active"
    _write_tiny_huggingface_clm(model_path)
    monkeypatch.setenv("ICLM_CHECKPOINT_DIRECTORY", str(tmp_path / "checkpoints"))
    first_result = HuggingFaceCausalLMRunner(model_path=str(model_path)).update(
        {
            "samples": [{"smiles": "CCO"}],
            "kd_weight": 0.5,
            "kd_teacher_embeddings": [[0.5, -0.5, 0.25, -0.25]],
            "target_checkpoint_version": "first",
        }
    )
    first_checkpoint = Path(first_result["checkpoint_path"])
    first_model = AutoModelForCausalLM.from_pretrained(first_checkpoint)
    first_state = torch.load(
        first_checkpoint / _CONTINUAL_STATE_FILE,
        map_location="cpu",
        weights_only=True,
    )
    assert first_state["strategy"] == "ewc"
    assert "ewc" in first_state
    assert "packnet" not in first_state

    second_result = HuggingFaceCausalLMRunner(
        model_path=str(first_checkpoint)
    ).update(
        {
            "samples": [{"smiles": "CCN"}],
            "kd_weight": 0.5,
            "kd_teacher_embeddings": [[-0.5, 0.5, -0.25, 0.25]],
            "target_checkpoint_version": "second",
        }
    )
    second_checkpoint = Path(second_result["checkpoint_path"])
    second_model = AutoModelForCausalLM.from_pretrained(second_checkpoint)
    second_state = torch.load(
        second_checkpoint / _CONTINUAL_STATE_FILE,
        map_location="cpu",
        weights_only=True,
    )

    assert second_state["strategy"] == "ewc"
    assert "packnet" not in second_state
    assert any(
        not torch.equal(
            dict(first_model.named_parameters())[name],
            dict(second_model.named_parameters())[name],
        )
        for name, _ in first_model.named_parameters()
    )


@pytest.mark.parametrize(
    ("rewards", "message"),
    [
        ([0.0, 0.0], "actionable teacher signal"),
        ([float("nan"), 1.0], "sample rewards"),
        ([1.1, 0.0], "sample rewards"),
    ],
)
def test_iclm_huggingface_update_rejects_invalid_teacher_rewards(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    rewards: list[float],
    message: str,
) -> None:
    from mf_generators.incremental_clm.hf_runner import HuggingFaceCausalLMRunner

    model_path = tmp_path / "active"
    _write_tiny_huggingface_clm(model_path)
    monkeypatch.setenv("ICLM_CHECKPOINT_DIRECTORY", str(tmp_path / "checkpoints"))

    with pytest.raises(ValueError, match=message):
        HuggingFaceCausalLMRunner(model_path=str(model_path)).update(
            {
                "samples": [
                    {"smiles": "CCO", "reward": rewards[0]},
                    {"smiles": "CCN", "reward": rewards[1]},
                ],
                "kd_weight": 0.5,
                "kd_teacher_embeddings": [
                    [0.5, -0.5, 0.25, -0.25],
                    [-0.25, 0.25, -0.5, 0.5],
                ],
                "target_checkpoint_version": "iclm-v2",
            }
        )


def test_iclm_huggingface_task_and_teacher_losses_update_parameters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mf_generators.incremental_clm.hf_runner import HuggingFaceCausalLMRunner
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = tmp_path / "active"
    initial_parameters = _write_tiny_huggingface_clm(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    encoded = tokenizer(
        ["CCO [EOS]"],
        return_tensors="pt",
        padding=True,
    )
    encoded.pop("token_type_ids", None)
    with torch.no_grad():
        hidden = model(
            **encoded,
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states[-1]
    mask = encoded["attention_mask"].unsqueeze(-1).to(dtype=hidden.dtype)
    teacher_embedding = (
        (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    ).tolist()
    monkeypatch.setenv("ICLM_CHECKPOINT_DIRECTORY", str(tmp_path / "checkpoints"))

    result = HuggingFaceCausalLMRunner(model_path=str(model_path)).update(
        {
            "samples": [{"smiles": "CCO"}],
            "kd_weight": 0.5,
            "kd_teacher_embeddings": teacher_embedding,
            "target_checkpoint_version": "iclm-v2",
        }
    )

    updated_parameters = dict(
        AutoModelForCausalLM.from_pretrained(
            result["checkpoint_path"]
        ).named_parameters()
    )
    assert result["teacher_loss"] > 0.0
    assert result["distillation_loss"] >= 0.0
    assert result["ewc_loss"] > 0.0
    assert result["total_loss"] == pytest.approx(
        0.5 * result["teacher_loss"]
        + 0.5 * result["distillation_loss"]
        + result["ewc_loss"]
    )
    assert any(
        not torch.equal(initial_parameters[name], updated_parameters[name])
        for name in initial_parameters
    )


def test_iclm_huggingface_failed_update_does_not_mutate_active_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mf_generators.incremental_clm.hf_runner import HuggingFaceCausalLMRunner

    model_path = tmp_path / "active"
    _write_tiny_huggingface_clm(model_path)
    monkeypatch.setenv("ICLM_CHECKPOINT_DIRECTORY", str(tmp_path / "checkpoints"))
    runner = HuggingFaceCausalLMRunner(model_path=str(model_path))
    active_model, _ = runner._load()
    active_parameters = {
        name: parameter.detach().clone()
        for name, parameter in active_model.named_parameters()
    }

    def fail_checkpoint_write(**_kwargs: object) -> Path:
        raise RuntimeError("checkpoint write failed")

    monkeypatch.setattr(runner, "_write_checkpoint", fail_checkpoint_write)

    with pytest.raises(RuntimeError, match="checkpoint write failed"):
        runner.update(
            {
                "samples": [{"smiles": "CCO"}],
                "kd_weight": 0.5,
                "kd_teacher_embeddings": [[0.5, -0.5, 0.25, -0.25]],
                "target_checkpoint_version": "iclm-v2",
            }
        )

    assert all(
        torch.equal(active_parameters[name], parameter)
        for name, parameter in active_model.named_parameters()
    )


def test_iclm_huggingface_update_recovers_matching_complete_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mf_generators.incremental_clm.hf_runner import HuggingFaceCausalLMRunner

    model_path = tmp_path / "active"
    _write_tiny_huggingface_clm(model_path)
    monkeypatch.setenv("ICLM_CHECKPOINT_DIRECTORY", str(tmp_path / "checkpoints"))
    payload = {
        "samples": [{"smiles": "CCO"}],
        "kd_weight": 0.5,
        "kd_teacher_embeddings": [[0.5, -0.5, 0.25, -0.25]],
        "run_id": "run-iclm",
        "request_id": "request-iclm",
        "target_checkpoint_version": "iclm-v2",
    }
    runner = HuggingFaceCausalLMRunner(model_path=str(model_path))

    first_result = runner.update(payload)
    recovered_result = HuggingFaceCausalLMRunner(
        model_path=str(model_path)
    ).update(payload)

    assert recovered_result == first_result


def test_iclm_huggingface_update_rejects_conflicting_existing_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mf_generators.incremental_clm.hf_runner import HuggingFaceCausalLMRunner

    model_path = tmp_path / "active"
    _write_tiny_huggingface_clm(model_path)
    monkeypatch.setenv("ICLM_CHECKPOINT_DIRECTORY", str(tmp_path / "checkpoints"))
    runner = HuggingFaceCausalLMRunner(model_path=str(model_path))
    payload = {
        "samples": [{"smiles": "CCO"}],
        "kd_weight": 0.5,
        "kd_teacher_embeddings": [[0.5, -0.5, 0.25, -0.25]],
        "target_checkpoint_version": "iclm-v2",
    }
    runner.update(payload)

    with pytest.raises(
        RuntimeError,
        match="existing ICLM checkpoint does not match the requested update",
    ):
        runner.update(
            {
                **payload,
                "kd_teacher_embeddings": [[-0.5, 0.5, -0.25, 0.25]],
            }
        )


def test_iclm_huggingface_update_rejects_unloadable_recovery_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mf_generators.incremental_clm.hf_runner import HuggingFaceCausalLMRunner

    model_path = tmp_path / "active"
    _write_tiny_huggingface_clm(model_path)
    monkeypatch.setenv("ICLM_CHECKPOINT_DIRECTORY", str(tmp_path / "checkpoints"))
    runner = HuggingFaceCausalLMRunner(model_path=str(model_path))
    payload = {
        "samples": [{"smiles": "CCO"}],
        "kd_weight": 0.5,
        "kd_teacher_embeddings": [[0.5, -0.5, 0.25, -0.25]],
        "target_checkpoint_version": "iclm-v2",
    }
    result = runner.update(payload)
    checkpoint_path = Path(result["checkpoint_path"])
    (checkpoint_path / "model.safetensors").unlink()

    with pytest.raises(
        RuntimeError,
        match="existing ICLM checkpoint is not loadable",
    ):
        runner.update(payload)


def test_iclm_huggingface_update_rejects_loadable_tampered_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mf_generators.incremental_clm.hf_runner import HuggingFaceCausalLMRunner
    from transformers import AutoModelForCausalLM

    model_path = tmp_path / "active"
    _write_tiny_huggingface_clm(model_path)
    monkeypatch.setenv("ICLM_CHECKPOINT_DIRECTORY", str(tmp_path / "checkpoints"))
    runner = HuggingFaceCausalLMRunner(model_path=str(model_path))
    payload = {
        "samples": [{"smiles": "CCO"}],
        "kd_weight": 0.5,
        "kd_teacher_embeddings": [[0.5, -0.5, 0.25, -0.25]],
        "target_checkpoint_version": "iclm-v2",
    }
    result = runner.update(payload)
    checkpoint_path = Path(result["checkpoint_path"])
    tampered_model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    with torch.no_grad():
        next(tampered_model.parameters()).add_(0.25)
    tampered_model.save_pretrained(checkpoint_path)
    AutoModelForCausalLM.from_pretrained(checkpoint_path)

    with pytest.raises(RuntimeError, match="checkpoint manifest"):
        runner.update(payload)


def test_iclm_huggingface_update_rejects_recovery_without_continual_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mf_generators.incremental_clm.hf_runner import (
        _CONTINUAL_STATE_FILE,
        HuggingFaceCausalLMRunner,
    )

    model_path = tmp_path / "active"
    _write_tiny_huggingface_clm(model_path)
    monkeypatch.setenv("ICLM_CHECKPOINT_DIRECTORY", str(tmp_path / "checkpoints"))
    runner = HuggingFaceCausalLMRunner(model_path=str(model_path))
    payload = {
        "samples": [{"smiles": "CCO"}],
        "kd_weight": 0.5,
        "kd_teacher_embeddings": [[0.5, -0.5, 0.25, -0.25]],
        "target_checkpoint_version": "iclm-v2",
    }
    result = runner.update(payload)
    checkpoint_path = Path(result["checkpoint_path"])
    (checkpoint_path / _CONTINUAL_STATE_FILE).unlink()

    with pytest.raises(
        RuntimeError,
        match="existing ICLM checkpoint is not loadable",
    ):
        runner.update(payload)


def test_iclm_huggingface_update_rejects_non_finite_recovered_ewc_metric(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mf_generators.incremental_clm.hf_runner import (
        _UPDATE_METADATA_FILE,
        HuggingFaceCausalLMRunner,
    )

    model_path = tmp_path / "active"
    _write_tiny_huggingface_clm(model_path)
    monkeypatch.setenv("ICLM_CHECKPOINT_DIRECTORY", str(tmp_path / "checkpoints"))
    runner = HuggingFaceCausalLMRunner(model_path=str(model_path))
    payload = {
        "samples": [{"smiles": "CCO"}],
        "kd_weight": 0.5,
        "kd_teacher_embeddings": [[0.5, -0.5, 0.25, -0.25]],
        "target_checkpoint_version": "iclm-v2",
    }
    result = runner.update(payload)
    metadata_path = Path(result["checkpoint_path"]) / _UPDATE_METADATA_FILE
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["result"]["ewc_loss"] = float("nan")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(
        RuntimeError,
        match="existing ICLM checkpoint metadata is invalid",
    ):
        runner.update(payload)


@pytest.mark.parametrize(
    "target_version",
    ("", ".", "..", "../escape", str(ROOT.parent / "escape")),
)
def test_iclm_huggingface_update_rejects_unsafe_target_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    target_version: str,
) -> None:
    from mf_generators.incremental_clm.hf_runner import HuggingFaceCausalLMRunner

    model_path = tmp_path / "active"
    _write_tiny_huggingface_clm(model_path)
    checkpoint_directory = tmp_path / "checkpoints"
    monkeypatch.setenv("ICLM_CHECKPOINT_DIRECTORY", str(checkpoint_directory))

    with pytest.raises(
        ValueError,
        match="target_checkpoint_version must be a file-safe name",
    ):
        HuggingFaceCausalLMRunner(model_path=str(model_path)).update(
            {
                "samples": [{"smiles": "CCO"}],
                "kd_weight": 0.5,
                "kd_teacher_embeddings": [[0.5, -0.5, 0.25, -0.25]],
                "target_checkpoint_version": target_version,
            }
        )

    assert not checkpoint_directory.exists()


def test_iclm_huggingface_rejects_mismatched_teacher_embedding_dimension(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mf_generators.incremental_clm.hf_runner import HuggingFaceCausalLMRunner

    model_path = tmp_path / "active"
    _write_tiny_huggingface_clm(model_path)
    monkeypatch.setenv("ICLM_CHECKPOINT_DIRECTORY", str(tmp_path / "checkpoints"))

    with pytest.raises(
        ValueError,
        match="teacher embedding dimension 3 does not match student embedding dimension 4",
    ):
        HuggingFaceCausalLMRunner(model_path=str(model_path)).update(
            {
                "samples": [{"smiles": "CCO"}],
                "kd_weight": 0.5,
                "kd_teacher_embeddings": [[0.5, -0.5, 0.25]],
                "target_checkpoint_version": "iclm-v2",
            }
        )


def test_iclm_huggingface_exposes_student_embedding_dimension(tmp_path: Path) -> None:
    from mf_generators.incremental_clm.hf_runner import HuggingFaceCausalLMRunner

    model_path = tmp_path / "active"
    _write_tiny_huggingface_clm(model_path)

    assert HuggingFaceCausalLMRunner(
        model_path=str(model_path)
    ).embedding_dimension() == 4


def test_iclm_validation_checkpoint_bootstrap_runs_initial_update(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mf_generators.incremental_clm.hf_runner import (
        HuggingFaceCausalLMRunner,
        bootstrap_validation_checkpoint,
    )

    model_path = tmp_path / "validation-model"
    checkpoint_path = bootstrap_validation_checkpoint(model_path)
    repeated_path = bootstrap_validation_checkpoint(model_path)
    monkeypatch.setenv("ICLM_CHECKPOINT_DIRECTORY", str(tmp_path / "checkpoints"))

    first_generation = HuggingFaceCausalLMRunner(
        model_path=str(checkpoint_path)
    ).generate(batch_size=64, sampling_seed="0")
    repeated_generation = HuggingFaceCausalLMRunner(
        model_path=str(checkpoint_path)
    ).generate(batch_size=64, sampling_seed="0")

    result = HuggingFaceCausalLMRunner(model_path=str(checkpoint_path)).update(
        {
            "samples": [
                {
                    "smiles": "CCO",
                    "reward": 1.0,
                    "outcome": "PASS",
                },
                {
                    "smiles": "CCN",
                    "reward": 0.0,
                    "outcome": "FAIL",
                },
            ],
            "teacher_weight": 0.5,
            "target_checkpoint_version": "validation-update",
        }
    )

    assert checkpoint_path == model_path.resolve()
    assert repeated_path == checkpoint_path
    assert (checkpoint_path / "moleculeforge_validation_model.json").is_file()
    assert len(first_generation) == 64
    assert first_generation == repeated_generation
    assert result["updated_samples"] == 2
    assert Path(result["checkpoint_path"]).is_dir()
    assert (
        Path(result["checkpoint_path"]) / "moleculeforge_validation_model.json"
    ).is_file()


def test_iclm_huggingface_update_requires_checkpoint_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mf_generators.incremental_clm.hf_runner import HuggingFaceCausalLMRunner

    model_path = tmp_path / "active"
    _write_tiny_huggingface_clm(model_path)
    monkeypatch.delenv("ICLM_CHECKPOINT_DIRECTORY", raising=False)

    with pytest.raises(RuntimeError, match="ICLM_CHECKPOINT_DIRECTORY is required"):
        HuggingFaceCausalLMRunner(model_path=str(model_path)).update(
            {
                "samples": [{"smiles": "CCO"}],
                "kd_weight": 0.5,
                "kd_teacher_embeddings": [[0.5, -0.5, 0.25, -0.25]],
                "target_checkpoint_version": "iclm-v2",
            }
        )


def test_iclm_huggingface_first_update_requires_versioned_ewc_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mf_generators.incremental_clm.hf_runner import HuggingFaceCausalLMRunner

    model_path = tmp_path / "active"
    _write_tiny_huggingface_clm(model_path)
    (model_path / "moleculeforge_ewc_replay.json").unlink()
    monkeypatch.setenv("ICLM_CHECKPOINT_DIRECTORY", str(tmp_path / "checkpoints"))

    with pytest.raises(RuntimeError, match="versioned EWC replay"):
        HuggingFaceCausalLMRunner(model_path=str(model_path)).update(
            {
                "samples": [{"smiles": "CCO"}],
                "kd_weight": 0.5,
                "target_checkpoint_version": "iclm-v2",
            }
        )


def test_iclm_huggingface_rejects_invalid_ewc_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mf_generators.incremental_clm.hf_runner import HuggingFaceCausalLMRunner

    model_path = tmp_path / "active"
    _write_tiny_huggingface_clm(model_path)
    (model_path / "moleculeforge_ewc_replay.json").write_text(
        json.dumps(
            {
                "schema_version": "iclm-ewc-replay.v1",
                "dataset_id": "invalid-calibration",
                "samples": [{"smiles": "CCO", "weight": 0.0}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ICLM_CHECKPOINT_DIRECTORY", str(tmp_path / "checkpoints"))

    with pytest.raises(RuntimeError, match="EWC replay sample weight"):
        HuggingFaceCausalLMRunner(model_path=str(model_path)).update(
            {
                "samples": [{"smiles": "CCO"}],
                "kd_weight": 0.5,
                "target_checkpoint_version": "iclm-v2",
            }
        )


@pytest.mark.parametrize(
    ("samples", "message"),
    [
        ([{"smiles": "not-a-smiles", "weight": 1.0}], "sample smiles"),
        (
            [
                {"smiles": "CCO", "weight": 1.0},
                {"smiles": "OCC", "weight": 1.0},
            ],
            "duplicate canonical",
        ),
    ],
)
def test_iclm_huggingface_rejects_semantically_invalid_ewc_replay(
    tmp_path: Path,
    samples: list[dict[str, object]],
    message: str,
) -> None:
    from mf_generators.incremental_clm.hf_runner import validate_ewc_replay

    replay_path = tmp_path / "moleculeforge_ewc_replay.json"
    replay_path.write_text(
        json.dumps(
            {
                "schema_version": "iclm-ewc-replay.v1",
                "dataset_id": "invalid-molecule-calibration",
                "samples": samples,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match=message):
        validate_ewc_replay(replay_path)


def test_iclm_huggingface_update_rejects_invalid_training_smiles(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mf_generators.incremental_clm.hf_runner import HuggingFaceCausalLMRunner

    model_path = tmp_path / "active"
    _write_tiny_huggingface_clm(model_path)
    monkeypatch.setenv("ICLM_CHECKPOINT_DIRECTORY", str(tmp_path / "checkpoints"))

    with pytest.raises(ValueError, match="sample smiles"):
        HuggingFaceCausalLMRunner(model_path=str(model_path)).update(
            {
                "samples": [{"smiles": "not-a-smiles", "reward": 1.0}],
                "teacher_weight": 0.5,
                "target_checkpoint_version": "invalid-smiles",
            }
        )


def test_iclm_huggingface_generation_does_not_return_prompt_on_invalid_continuation() -> None:
    from mf_generators.incremental_clm.hf_runner import HuggingFaceCausalLMRunner

    class Model:
        device = torch.device("cpu")

        @staticmethod
        def generate(**kwargs):
            return torch.tensor([[3, 9]])

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 1

        @staticmethod
        def __call__(prompts, **kwargs):
            return {
                "input_ids": torch.tensor([[3]]),
                "attention_mask": torch.tensor([[1]]),
            }

        @staticmethod
        def batch_decode(outputs, **kwargs):
            assert outputs.tolist() == [[9]]
            return ["@@@"]

    runner = HuggingFaceCausalLMRunner(model_path="unused")
    runner._model = Model()
    runner._tokenizer = Tokenizer()

    with pytest.raises(RuntimeError, match="did not generate enough valid SMILES"):
        runner.generate(batch_size=1, prompt="C")


def test_iclm_huggingface_generation_preserves_seed_prefix() -> None:
    from mf_generators.incremental_clm.hf_runner import HuggingFaceCausalLMRunner

    class Model:
        device = torch.device("cpu")

        @staticmethod
        def generate(**kwargs):
            return torch.tensor([[3, 9]])

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 1

        @staticmethod
        def __call__(prompts, **kwargs):
            assert prompts == ["CC"]
            return {
                "input_ids": torch.tensor([[3]]),
                "attention_mask": torch.tensor([[1]]),
            }

        @staticmethod
        def batch_decode(outputs, **kwargs):
            assert outputs.tolist() == [[9]]
            return ["O"]

    runner = HuggingFaceCausalLMRunner(model_path="unused")
    runner._model = Model()
    runner._tokenizer = Tokenizer()

    molecules = runner.generate(batch_size=1, seed_smiles="CC")

    assert molecules[0]["smiles"] == "CCO"


def test_iclm_huggingface_checkpoint_is_not_exposed_after_partial_save(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mf_generators.incremental_clm.hf_runner import HuggingFaceCausalLMRunner

    model_path = tmp_path / "active"
    _write_tiny_huggingface_clm(model_path)
    checkpoint_directory = tmp_path / "checkpoints"
    monkeypatch.setenv("ICLM_CHECKPOINT_DIRECTORY", str(checkpoint_directory))
    runner = HuggingFaceCausalLMRunner(model_path=str(model_path))
    training_model, tokenizer = runner._load_checkpoint(model_path)
    target_path = checkpoint_directory / "iclm-v2"

    def fail_partial_save(path: Path) -> None:
        assert not target_path.exists()
        (Path(path) / "partial").write_text("partial", encoding="utf-8")
        raise RuntimeError("save failed")

    monkeypatch.setattr(training_model, "save_pretrained", fail_partial_save)
    monkeypatch.setattr(
        runner,
        "_load_checkpoint",
        lambda _checkpoint_path: (training_model, tokenizer),
    )

    with pytest.raises(RuntimeError, match="save failed"):
        runner.update(
            {
                "samples": [{"smiles": "CCO"}],
                "kd_weight": 0.5,
                "kd_teacher_embeddings": [[0.5, -0.5, 0.25, -0.25]],
                "target_checkpoint_version": "iclm-v2",
            }
        )

    assert not target_path.exists()
    assert list(checkpoint_directory.iterdir()) == []


def test_iclm_huggingface_does_not_report_success_after_directory_fsync_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import mf_generators.incremental_clm.hf_runner as hf_runner

    model_path = tmp_path / "active"
    _write_tiny_huggingface_clm(model_path)
    checkpoint_directory = tmp_path / "checkpoints"
    monkeypatch.setenv("ICLM_CHECKPOINT_DIRECTORY", str(checkpoint_directory))

    def fail_checkpoint_directory_sync(directory: Path) -> None:
        if directory == checkpoint_directory:
            raise OSError("checkpoint directory fsync failed")

    monkeypatch.setattr(
        hf_runner,
        "_fsync_directory",
        fail_checkpoint_directory_sync,
        raising=False,
    )

    with pytest.raises(OSError, match="checkpoint directory fsync failed"):
        hf_runner.HuggingFaceCausalLMRunner(model_path=str(model_path)).update(
            {
                "samples": [{"smiles": "CCO"}],
                "kd_weight": 0.5,
                "kd_teacher_embeddings": [[0.5, -0.5, 0.25, -0.25]],
                "target_checkpoint_version": "iclm-v2",
            }
        )


def test_iclm_huggingface_recovery_resyncs_checkpoint_after_root_fsync_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import mf_generators.incremental_clm.hf_runner as hf_runner

    model_path = tmp_path / "active"
    _write_tiny_huggingface_clm(model_path)
    checkpoint_directory = tmp_path / "checkpoints"
    monkeypatch.setenv("ICLM_CHECKPOINT_DIRECTORY", str(checkpoint_directory))
    real_fsync_directory = hf_runner._fsync_directory
    root_fsync_calls = 0

    def fail_first_root_fsync(directory: Path) -> None:
        nonlocal root_fsync_calls
        if directory == checkpoint_directory:
            root_fsync_calls += 1
            if root_fsync_calls == 1:
                raise OSError("checkpoint root fsync failed")
        real_fsync_directory(directory)

    monkeypatch.setattr(hf_runner, "_fsync_directory", fail_first_root_fsync)
    runner = hf_runner.HuggingFaceCausalLMRunner(model_path=str(model_path))
    payload = {
        "samples": [{"smiles": "CCO"}],
        "kd_weight": 0.5,
        "kd_teacher_embeddings": [[0.5, -0.5, 0.25, -0.25]],
        "target_checkpoint_version": "iclm-v2",
    }

    with pytest.raises(OSError, match="checkpoint root fsync failed"):
        runner.update(payload)

    recovered = hf_runner.HuggingFaceCausalLMRunner(
        model_path=str(model_path)
    ).update(payload)

    assert Path(recovered["checkpoint_path"]) == checkpoint_directory / "iclm-v2"
    assert root_fsync_calls == 2


def test_iclm_huggingface_update_command_satisfies_service_json_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = tmp_path / "active"
    _write_tiny_huggingface_clm(model_path)
    checkpoint_directory = tmp_path / "checkpoints"
    runner_path = (
        ROOT
        / "models/mf-generators/incremental_clm/src"
        / "mf_generators/incremental_clm/hf_runner.py"
    )
    monkeypatch.setenv("ICLM_CHECKPOINT_DIRECTORY", str(checkpoint_directory))
    monkeypatch.setenv("ICLM_UPDATE_COMMAND", f"{sys.executable} {runner_path}")
    service = _load_module(
        "iclm_hf_command_contract_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )

    result = asyncio.run(
        service._run_update_command(
            {
                "samples": [{"smiles": "CCO"}],
                "kd_weight": 0.25,
                "kd_teacher_embeddings": [[0.5, -0.5, 0.25, -0.25]],
                "target_checkpoint_version": "iclm-v3",
            },
            str(model_path),
        )
    )

    checkpoint_path = Path(result["checkpoint_path"])
    assert checkpoint_path == checkpoint_directory / "iclm-v3"
    assert result["updated_samples"] == 1
    AutoModelForCausalLM.from_pretrained(checkpoint_path)
    AutoTokenizer.from_pretrained(checkpoint_path)


def test_iclm_default_huggingface_update_honors_command_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "active"
    model_path.mkdir()
    blocking_runner = tmp_path / "blocking_update.py"
    blocking_runner.write_text(
        "import time\n"
        "time.sleep(5)\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ICLM_MODEL_PATH", str(model_path))
    monkeypatch.setenv("ICLM_CHECKPOINT_DIRECTORY", str(tmp_path / "checkpoints"))
    monkeypatch.setenv("ICLM_UPDATE_TIMEOUT_SECONDS", "0.05")
    monkeypatch.delenv("ICLM_UPDATE_COMMAND", raising=False)
    service = _load_module(
        "iclm_default_hf_timeout_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    monkeypatch.setattr(
        service,
        "_builtin_hf_update_argv",
        lambda: (sys.executable, str(blocking_runner)),
        raising=False,
    )

    with pytest.raises(
        RuntimeError,
        match="built-in ICLM update runner execution failed: timed out",
    ):
        asyncio.run(
            service._run_update(
                {
                    "samples": [{"smiles": "CCO"}],
                    "kd_weight": 0.5,
                    "kd_teacher_embeddings": [[0.5, -0.5, 0.25, -0.25]],
                    "target_checkpoint_version": "iclm-v2",
                },
                service._build_generator(),
            )
        )


def test_iclm_default_huggingface_command_failure_preserves_active_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "active"
    _write_tiny_huggingface_clm(model_path)
    failing_runner = tmp_path / "failing_update.py"
    failing_runner.write_text("raise SystemExit(3)\n", encoding="utf-8")
    monkeypatch.setenv("ICLM_MODEL_PATH", str(model_path))
    monkeypatch.setenv("ICLM_CHECKPOINT_DIRECTORY", str(tmp_path / "checkpoints"))
    monkeypatch.delenv("ICLM_UPDATE_COMMAND", raising=False)
    service = _load_module(
        "iclm_default_hf_failure_isolation_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    generator = service._build_generator()
    active_model, _ = generator.runner._load()
    active_parameters = {
        name: parameter.detach().clone()
        for name, parameter in active_model.named_parameters()
    }
    monkeypatch.setattr(
        service,
        "_builtin_hf_update_argv",
        lambda: (sys.executable, str(failing_runner)),
    )

    with pytest.raises(
        RuntimeError,
        match="built-in ICLM update runner failed",
    ):
        asyncio.run(
            service._run_update(
                {
                    "samples": [{"smiles": "CCO"}],
                    "kd_weight": 0.5,
                    "kd_teacher_embeddings": [[0.5, -0.5, 0.25, -0.25]],
                    "target_checkpoint_version": "iclm-v2",
                },
                generator,
            )
        )

    assert all(
        torch.equal(active_parameters[name], parameter)
        for name, parameter in active_model.named_parameters()
    )


def test_iclm_service_rejects_checkpoint_outside_configured_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service = _load_module(
        "iclm_checkpoint_containment_test",
        ROOT / "services/iclm-svc/src/iclm_svc/main.py",
    )
    checkpoint_directory = tmp_path / "checkpoints"
    active_checkpoint = checkpoint_directory / "active"
    outside_checkpoint = tmp_path / "outside"
    monkeypatch.setenv("ICLM_CHECKPOINT_DIRECTORY", str(checkpoint_directory))

    with pytest.raises(
        RuntimeError,
        match="ICLM update checkpoint must be within ICLM_CHECKPOINT_DIRECTORY",
    ):
        service._new_checkpoint_path(
            {"checkpoint_path": str(outside_checkpoint)},
            active_checkpoint_path=str(active_checkpoint),
        )


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


def test_uas_uses_injected_autoencoder_for_unfamiliarity() -> None:
    from mf_generators.uas.generator import UASGenerator

    class SelectiveAutoencoder(torch.nn.Module):
        def forward(self, embeddings: torch.Tensor):
            return torch.zeros_like(embeddings), embeddings

    def candidate_source(_n_samples: int):
        return torch.tensor([[0.0, 0.0], [4.0, 4.0]], dtype=torch.float32)

    def decoder(embeddings: torch.Tensor):
        assert embeddings.tolist() == [[0.0, 0.0]]
        return ["CCO"]

    generator = UASGenerator(
        dim=2,
        autoencoder=SelectiveAutoencoder(),
        candidate_source=candidate_source,
        decoder=decoder,
        unfamiliarity_threshold=0.5,
    )

    molecules = _collect(generator.generate(None, None, None, n_samples=1))

    assert [molecule.smiles for molecule in molecules] == ["CCO"]


def test_uas_awaits_async_candidate_source() -> None:
    from mf_generators.uas.generator import UASGenerator

    class ZeroAutoencoder(torch.nn.Module):
        def forward(self, embeddings: torch.Tensor):
            return torch.zeros_like(embeddings), embeddings

    async def candidate_source(_n_samples: int):
        return torch.tensor([[0.0, 0.0]], dtype=torch.float32)

    generator = UASGenerator(
        dim=2,
        autoencoder=ZeroAutoencoder(),
        candidate_source=candidate_source,
        decoder=lambda _embeddings: ["CCO"],
    )

    molecules = _collect(generator.generate(None, None, None, n_samples=1))

    assert [molecule.smiles for molecule in molecules] == ["CCO"]


def test_uas_attaches_accepted_embedding_to_decoded_molecule() -> None:
    from mf_generators.uas.generator import UASGenerator

    class ZeroAutoencoder(torch.nn.Module):
        def forward(self, embeddings: torch.Tensor):
            return torch.zeros_like(embeddings), embeddings

    generator = UASGenerator(
        dim=2,
        autoencoder=ZeroAutoencoder(),
        candidate_source=lambda _n_samples: [[0.0, 0.25]],
        decoder=lambda _embeddings: [{"smiles": "CCO"}],
    )

    molecules = _collect(generator.generate(None, None, None, n_samples=1))

    assert molecules[0].humu_embedding == [0.0, 0.25]


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
    from unittest.mock import AsyncMock, MagicMock

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
