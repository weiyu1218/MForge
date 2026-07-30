"""Hugging Face causal language model runner for SMILES generation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as functional
from mf_generators.incremental_clm.model import EWCRegularizer

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover
    Chem = None

_UPDATE_METADATA_FILE = "moleculeforge_update.json"
_UPDATE_METADATA_SCHEMA = "iclm-hf-update.v5"
_CHECKPOINT_MANIFEST_FILE = "moleculeforge_checkpoint_manifest.json"
_CHECKPOINT_MANIFEST_SCHEMA = "iclm-checkpoint-manifest.v1"
_CONTINUAL_STATE_FILE = "moleculeforge_continual_state.pt"
_CONTINUAL_STATE_SCHEMA = "iclm-continual-state.v3"
_PREVIOUS_CONTINUAL_STATE_SCHEMA = "iclm-continual-state.v2"
_LEGACY_CONTINUAL_STATE_SCHEMA = "iclm-continual-state.v1"
_EWC_REPLAY_FILE = "moleculeforge_ewc_replay.json"
_EWC_REPLAY_SCHEMA = "iclm-ewc-replay.v1"
_EWC_CALIBRATION_SCHEMA = "iclm-ewc-calibration.v1"
_VALIDATION_MODEL_METADATA_FILE = "moleculeforge_validation_model.json"
_VALIDATION_MODEL_SCHEMA = "iclm-validation-model.v1"


@dataclass(frozen=True)
class _EWCReplay:
    smiles: list[str]
    weights: list[float]
    calibration: dict[str, object]


@dataclass
class HuggingFaceCausalLMRunner:
    model_path: str
    device: str = "cpu"
    default_prompt: str = "C"
    learning_rate: float = 1e-4
    ewc_weight: float = 1.0
    update_steps: int = 2

    def __post_init__(self) -> None:
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.ewc_weight) or self.ewc_weight < 0.0:
            raise ValueError("ewc_weight must be finite and non-negative")
        if (
            isinstance(self.update_steps, bool)
            or not isinstance(self.update_steps, int)
            or self.update_steps < 2
        ):
            raise ValueError("update_steps must be an integer greater than or equal to two")
        self._model = None
        self._tokenizer = None

    def generate(
        self,
        batch_size: int,
        **kwargs: object,
    ) -> list[dict[str, object]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        model, tokenizer = self._load()
        prompt = str(kwargs.get("prompt") or kwargs.get("seed_smiles") or self.default_prompt)
        explicit_max_new_tokens = "max_new_tokens" in kwargs
        max_new_tokens = _positive_generation_integer(
            kwargs.get("max_new_tokens", 64),
            "max_new_tokens",
        )
        temperature = float(kwargs.get("temperature", 0.8))
        if not math.isfinite(temperature) or temperature < 0.0:
            raise ValueError("temperature must be finite and non-negative")
        sampling_seed = _optional_sampling_seed(kwargs)
        encoded = tokenizer(
            [prompt] * batch_size,
            return_tensors="pt",
            padding=True,
        )
        encoded.pop("token_type_ids", None)
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        input_length = encoded["input_ids"].shape[1]
        context_limit = _model_context_limit(model)
        if context_limit is not None:
            available_tokens = context_limit - input_length
            if available_tokens <= 0:
                raise ValueError("ICLM prompt exceeds the model context window")
            if explicit_max_new_tokens and max_new_tokens > available_tokens:
                raise ValueError("max_new_tokens exceeds the model context window")
            max_new_tokens = min(max_new_tokens, available_tokens)
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": 1,
            "do_sample": temperature > 0.0,
            "pad_token_id": pad_token_id,
        }
        suppressed_tokens = sorted(
            {
                token_id
                for token_id in (
                    getattr(tokenizer, "pad_token_id", None),
                    getattr(tokenizer, "unk_token_id", None),
                )
                if token_id is not None and token_id != tokenizer.eos_token_id
            }
        )
        if suppressed_tokens:
            generation_kwargs["suppress_tokens"] = suppressed_tokens
        if temperature > 0.0:
            generation_kwargs["temperature"] = temperature
        with torch.no_grad():
            if sampling_seed is None:
                outputs = model.generate(**encoded, **generation_kwargs)
            else:
                rng_devices = []
                if model.device.type == "cuda":
                    rng_devices.append(
                        model.device.index
                        if model.device.index is not None
                        else torch.cuda.current_device()
                    )
                with torch.random.fork_rng(devices=rng_devices):
                    torch.manual_seed(sampling_seed)
                    outputs = model.generate(**encoded, **generation_kwargs)
        if outputs.ndim != 2 or outputs.shape[1] <= input_length:
            raise RuntimeError("ICLM runner did not generate a continuation")
        continuations = tokenizer.batch_decode(
            outputs[:, input_length:],
            skip_special_tokens=True,
        )
        molecules = []
        for continuation in continuations:
            if not continuation.strip():
                continue
            smiles = _extract_valid_smiles(f"{prompt}{continuation}")
            if smiles is None:
                continue
            molecules.append(
                {
                    "smiles": smiles,
                    "metadata": {
                        "model_path": self.model_path,
                        "runner": "huggingface_causal_lm",
                    },
                }
            )
        if len(molecules) < batch_size:
            raise RuntimeError("ICLM runner did not generate enough valid SMILES")
        return molecules[:batch_size]

    def update(self, batch: Mapping[str, object]) -> dict[str, object]:
        samples = batch.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError("ICLM update requires non-empty samples")
        smiles: list[str] = []
        directions: list[str] = []
        strengths: list[float] = []
        normalized_samples: list[dict[str, object]] = []
        for sample in samples:
            if not isinstance(sample, Mapping):
                raise ValueError("ICLM update samples must be mappings")
            raw_smiles = sample.get("smiles")
            if not isinstance(raw_smiles, str) or not raw_smiles.strip():
                raise ValueError("ICLM update samples require smiles")
            canonical_smiles = _canonical_training_smiles(raw_smiles.strip())
            raw_reward = sample.get("reward", 1.0)
            if (
                isinstance(raw_reward, bool)
                or not isinstance(raw_reward, int | float)
                or not math.isfinite(float(raw_reward))
                or not 0.0 <= float(raw_reward) <= 1.0
            ):
                raise ValueError("ICLM update sample rewards must be finite values in [0, 1]")
            reward = float(raw_reward)
            outcome = sample.get("outcome")
            if outcome is not None and outcome not in {"PASS", "FAIL"}:
                raise ValueError("ICLM update sample outcome must be PASS or FAIL")
            direction = "unlikelihood" if outcome == "FAIL" else "likelihood"
            strength = 1.0 - reward if outcome == "FAIL" else reward
            smiles.append(canonical_smiles)
            directions.append(direction)
            strengths.append(strength)
            normalized_sample = dict(sample)
            normalized_sample["smiles"] = canonical_smiles
            normalized_sample["reward"] = reward
            normalized_samples.append(normalized_sample)
        if math.fsum(strengths) <= 0.0:
            raise ValueError("ICLM update samples must contain actionable teacher signal")
        teacher_weight = _finite_unit_interval(
            batch.get("teacher_weight", batch.get("kd_weight")),
            "teacher_weight",
        )
        teacher_embeddings = _normalized_teacher_embeddings(
            (
                batch["teacher_embeddings"]
                if "teacher_embeddings" in batch
                else batch.get("kd_teacher_embeddings")
            ),
            expected_rows=len(samples),
        )
        target_version = str(batch.get("target_checkpoint_version", "")).strip()
        target_path = Path(target_version)
        if (
            not target_version
            or target_version in {".", ".."}
            or target_path.is_absolute()
            or "/" in target_version
            or "\\" in target_version
        ):
            raise ValueError("target_checkpoint_version must be a file-safe name")
        checkpoint_root_value = os.environ.get("ICLM_CHECKPOINT_DIRECTORY", "").strip()
        if not checkpoint_root_value:
            raise RuntimeError("ICLM_CHECKPOINT_DIRECTORY is required")
        checkpoint_root = Path(checkpoint_root_value)
        checkpoint_path = checkpoint_root / target_version
        normalized_batch = {**batch, "samples": normalized_samples}
        normalized_batch.pop("kd_weight", None)
        normalized_batch.pop("teacher_weight", None)
        normalized_batch["teacher_weight"] = teacher_weight
        normalized_batch.pop("kd_teacher_embeddings", None)
        normalized_batch.pop("teacher_embeddings", None)
        if teacher_embeddings is not None:
            normalized_batch["teacher_embeddings"] = teacher_embeddings
        update_fingerprint = self._update_fingerprint(normalized_batch)
        recovered_result = self._recover_checkpoint(
            checkpoint_path=checkpoint_path,
            update_fingerprint=update_fingerprint,
            updated_samples=len(samples),
        )
        if recovered_result is not None:
            return recovered_result

        model, tokenizer = self._load_checkpoint(self.model_path)
        teacher_tensor = self._teacher_embedding_tensor(
            model=model,
            teacher_embeddings=teacher_embeddings,
        )
        active_teacher_tensor = teacher_tensor if teacher_weight > 0.0 else None
        encoded, labels = self._encode_training_samples(
            model=model,
            tokenizer=tokenizer,
            smiles=smiles,
        )

        ewc_regularizer, base_calibration = self._load_continual_learning(
            model,
            tokenizer,
            checkpoint_path=Path(self.model_path),
        )
        mean_strength = math.fsum(strengths) / len(strengths)
        effective_learning_rate = self.learning_rate * mean_strength
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=effective_learning_rate,
        )
        model.train()
        teacher_loss = distillation_loss = ewc_loss = total_loss = None
        for _ in range(self.update_steps):
            optimizer.zero_grad()
            teacher_loss, _, _, distillation_loss = self._training_losses(
                model=model,
                encoded=encoded,
                labels=labels,
                directions=directions,
                sample_weights=strengths,
                teacher_embeddings=active_teacher_tensor,
            )
            ewc_loss = ewc_regularizer.ewc_loss()
            supervised_loss = (
                teacher_loss
                if active_teacher_tensor is None
                else (
                    (1.0 - teacher_weight) * teacher_loss
                    + teacher_weight * distillation_loss
                )
            )
            total_loss = supervised_loss + self.ewc_weight * ewc_loss
            if not torch.isfinite(total_loss):
                raise ValueError("ICLM update loss must be finite")
            total_loss.backward()
            optimizer.step()

        model.eval()
        (
            _,
            consolidation_task_losses,
            consolidation_weights,
            _,
        ) = self._training_losses(
            model=model,
            encoded=encoded,
            labels=labels,
            directions=directions,
            sample_weights=strengths,
        )
        ewc_regularizer.consolidate(
            consolidation_task_losses,
            sample_weights=consolidation_weights,
        )
        model.zero_grad(set_to_none=True)

        if (
            teacher_loss is None
            or distillation_loss is None
            or ewc_loss is None
            or total_loss is None
        ):
            raise RuntimeError("ICLM update did not execute an optimization step")
        continual_state = {
            "schema_version": _CONTINUAL_STATE_SCHEMA,
            "strategy": "ewc",
            "base_calibration": base_calibration,
            "ewc": ewc_regularizer.state_dict(),
        }
        validation_model_metadata = _load_validation_model_metadata(
            Path(self.model_path)
        )

        result = {
            "checkpoint_path": str(checkpoint_path),
            "updated_samples": len(samples),
            "teacher_loss": float(teacher_loss.detach().cpu().item()),
            "distillation_loss": float(
                distillation_loss.detach().cpu().item()
            ),
            "ewc_loss": float(ewc_loss.detach().cpu().item()),
            "total_loss": float(total_loss.detach().cpu().item()),
            "effective_learning_rate": effective_learning_rate,
        }
        try:
            written_checkpoint_path = self._write_checkpoint(
                model=model,
                tokenizer=tokenizer,
                checkpoint_root=checkpoint_root,
                target_version=target_version,
                update_fingerprint=update_fingerprint,
                result=result,
                continual_state=continual_state,
                validation_model_metadata=validation_model_metadata,
            )
        except FileExistsError:
            recovered_result = self._recover_checkpoint(
                checkpoint_path=checkpoint_path,
                update_fingerprint=update_fingerprint,
                updated_samples=len(samples),
            )
            if recovered_result is not None:
                return recovered_result
            raise
        result["checkpoint_path"] = str(written_checkpoint_path)
        return result

    @staticmethod
    def _encode_training_samples(
        *,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        smiles: list[str],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        eos_token = tokenizer.eos_token or ""
        training_text = [f"{value} {eos_token}".strip() for value in smiles]
        tokenizer_options: dict[str, object] = {}
        max_length = getattr(model.config, "max_position_embeddings", None)
        if isinstance(max_length, int) and max_length > 0:
            tokenizer_options.update(
                {
                    "truncation": True,
                    "max_length": max_length,
                }
            )
        encoded = tokenizer(
            training_text,
            return_tensors="pt",
            padding=True,
            **tokenizer_options,
        )
        encoded.pop("token_type_ids", None)
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        labels = encoded["input_ids"].clone()
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            labels = labels.masked_fill(attention_mask == 0, -100)
        return encoded, labels

    @staticmethod
    def _training_losses(
        *,
        model: PreTrainedModel,
        encoded: Mapping[str, torch.Tensor],
        labels: torch.Tensor,
        directions: list[str],
        sample_weights: list[float],
        teacher_embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = model(
            **encoded,
            output_hidden_states=teacher_embeddings is not None,
            return_dict=True,
        )
        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        likelihood_token_losses = functional.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(shift_labels.shape)
        valid_tokens = shift_labels.ne(-100)
        tokens_per_sample = valid_tokens.sum(dim=1)
        if torch.any(tokens_per_sample == 0):
            raise ValueError("ICLM update samples must contain a causal prediction target")
        safe_labels = shift_labels.masked_fill(~valid_tokens, 0)
        target_probabilities = functional.softmax(shift_logits, dim=-1).gather(
            dim=-1,
            index=safe_labels.unsqueeze(-1),
        ).squeeze(-1)
        probability_limit = 1.0 - torch.finfo(target_probabilities.dtype).eps
        unlikelihood_token_losses = -torch.log1p(
            -target_probabilities.clamp(max=probability_limit)
        )
        likelihood_sample_losses = (
            likelihood_token_losses * valid_tokens
        ).sum(dim=1) / tokens_per_sample
        unlikelihood_sample_losses = (
            unlikelihood_token_losses * valid_tokens
        ).sum(dim=1) / tokens_per_sample
        if len(directions) != labels.shape[0] or any(
            direction not in {"likelihood", "unlikelihood"}
            for direction in directions
        ):
            raise ValueError("ICLM update directions must match training samples")
        unlikelihood_mask = torch.tensor(
            [direction == "unlikelihood" for direction in directions],
            device=likelihood_sample_losses.device,
            dtype=torch.bool,
        )
        sample_task_losses = torch.where(
            unlikelihood_mask,
            unlikelihood_sample_losses,
            likelihood_sample_losses,
        )
        weights = torch.tensor(
            sample_weights,
            device=sample_task_losses.device,
            dtype=sample_task_losses.dtype,
        )
        if (
            weights.shape != sample_task_losses.shape
            or not torch.isfinite(weights).all()
            or torch.any(weights < 0)
            or weights.sum() <= 0
        ):
            raise ValueError("ICLM update sample weights must be finite and actionable")
        teacher_loss = (sample_task_losses * weights).mean()
        distillation_loss = teacher_loss.new_zeros(())
        if teacher_embeddings is not None:
            hidden_states = outputs.hidden_states
            if not hidden_states:
                raise RuntimeError("ICLM model did not return hidden states for distillation")
            student_hidden = hidden_states[-1]
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones(
                    student_hidden.shape[:2],
                    device=student_hidden.device,
                    dtype=student_hidden.dtype,
                )
            else:
                attention_mask = attention_mask.to(
                    device=student_hidden.device,
                    dtype=student_hidden.dtype,
                )
            mask = attention_mask.unsqueeze(-1)
            student_embeddings = (student_hidden * mask).sum(dim=1) / mask.sum(
                dim=1
            ).clamp_min(1.0)
            detached_teacher = teacher_embeddings.to(
                device=student_embeddings.device,
                dtype=student_embeddings.dtype,
            ).detach()
            if detached_teacher.shape != student_embeddings.shape:
                raise ValueError(
                    "teacher embeddings must match pooled student embeddings"
                )
            distillation_loss = functional.mse_loss(
                student_embeddings,
                detached_teacher,
            )
        return teacher_loss, sample_task_losses, weights, distillation_loss

    @staticmethod
    def _teacher_embedding_tensor(
        *,
        model: PreTrainedModel,
        teacher_embeddings: list[list[float]] | None,
    ) -> torch.Tensor | None:
        if teacher_embeddings is None:
            return None
        input_embeddings = model.get_input_embeddings()
        student_dimension = getattr(input_embeddings, "embedding_dim", None)
        if (
            isinstance(student_dimension, bool)
            or not isinstance(student_dimension, int)
            or student_dimension <= 0
        ):
            raise RuntimeError("ICLM model input embedding dimension is unavailable")
        teacher_dimension = len(teacher_embeddings[0])
        if teacher_dimension != student_dimension:
            raise ValueError(
                f"teacher embedding dimension {teacher_dimension} does not match "
                f"student embedding dimension {student_dimension}"
            )
        embedding_weight = getattr(input_embeddings, "weight", None)
        if not isinstance(embedding_weight, torch.Tensor):
            raise RuntimeError("ICLM model input embedding weights are unavailable")
        return torch.tensor(
            teacher_embeddings,
            device=embedding_weight.device,
            dtype=embedding_weight.dtype,
        ).detach()

    def _load_continual_learning(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        checkpoint_path: Path,
    ) -> tuple[EWCRegularizer, dict[str, object]]:
        ewc_regularizer = EWCRegularizer(model)
        state_path = checkpoint_path / _CONTINUAL_STATE_FILE
        if not checkpoint_path.is_dir() or not state_path.exists():
            replay = _load_ewc_replay(checkpoint_path / _EWC_REPLAY_FILE)
            encoded, labels = self._encode_training_samples(
                model=model,
                tokenizer=tokenizer,
                smiles=replay.smiles,
            )
            model.eval()
            _, replay_losses, replay_weights, _ = self._training_losses(
                model=model,
                encoded=encoded,
                labels=labels,
                directions=["likelihood"] * len(replay.smiles),
                sample_weights=replay.weights,
            )
            ewc_regularizer.consolidate(
                replay_losses,
                sample_weights=replay_weights,
            )
            model.zero_grad(set_to_none=True)
            return ewc_regularizer, replay.calibration
        try:
            state = torch.load(
                state_path,
                map_location="cpu",
                weights_only=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"ICLM continual-learning state is not loadable: {state_path}"
            ) from exc
        if not isinstance(state, Mapping) or state.get("schema_version") not in {
            _CONTINUAL_STATE_SCHEMA,
            _PREVIOUS_CONTINUAL_STATE_SCHEMA,
            _LEGACY_CONTINUAL_STATE_SCHEMA,
        }:
            raise RuntimeError("ICLM continual-learning state schema is invalid")
        if (
            state.get("schema_version") == _CONTINUAL_STATE_SCHEMA
            and state.get("strategy") != "ewc"
        ):
            raise RuntimeError("ICLM continual-learning strategy must be ewc")
        try:
            ewc_regularizer.load_state_dict(state.get("ewc"))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("ICLM continual-learning state is invalid") from exc
        if state.get("schema_version") == _CONTINUAL_STATE_SCHEMA:
            base_calibration = _validate_base_calibration(state.get("base_calibration"))
        else:
            base_calibration = {
                "schema_version": _EWC_CALIBRATION_SCHEMA,
                "source": "legacy_continual_state",
                "dataset_id": "legacy-unavailable",
                "sample_count": 0,
                "replay_checksum": f"sha256:{_sha256_file(state_path)}",
            }
        return ewc_regularizer, base_calibration

    @staticmethod
    def _write_checkpoint(
        *,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        checkpoint_root: Path,
        target_version: str,
        update_fingerprint: str,
        result: Mapping[str, object],
        continual_state: Mapping[str, object],
        validation_model_metadata: Mapping[str, object] | None,
    ) -> Path:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_root / target_version
        if checkpoint_path.exists():
            raise FileExistsError(f"ICLM checkpoint already exists: {checkpoint_path}")
        temporary_path = Path(
            tempfile.mkdtemp(
                prefix=f".{target_version}.",
                dir=checkpoint_root,
            )
        )
        try:
            model.save_pretrained(temporary_path)
            tokenizer.save_pretrained(temporary_path)
            if validation_model_metadata is not None:
                _write_json_artifact(
                    temporary_path / _VALIDATION_MODEL_METADATA_FILE,
                    validation_model_metadata,
                )
            continual_state_path = temporary_path / _CONTINUAL_STATE_FILE
            with continual_state_path.open("wb") as handle:
                torch.save(dict(continual_state), handle)
                handle.flush()
                os.fsync(handle.fileno())
            metadata_path = temporary_path / _UPDATE_METADATA_FILE
            with metadata_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "schema_version": _UPDATE_METADATA_SCHEMA,
                        "update_fingerprint": update_fingerprint,
                        "result": dict(result),
                    },
                    handle,
                    sort_keys=True,
                    allow_nan=False,
                )
                handle.flush()
                os.fsync(handle.fileno())
            _write_checkpoint_manifest(temporary_path)
            _fsync_checkpoint_tree(temporary_path)
            temporary_path.replace(checkpoint_path)
            _fsync_directory(checkpoint_root)
        finally:
            if temporary_path.exists():
                shutil.rmtree(temporary_path)
        return checkpoint_path

    def _update_fingerprint(self, batch: Mapping[str, object]) -> str:
        fingerprint_payload = {
            "model_path": str(Path(self.model_path).resolve()),
            "base_checkpoint_checksum": _checkpoint_tree_sha256(Path(self.model_path)),
            "learning_rate": self.learning_rate,
            "ewc_weight": self.ewc_weight,
            "update_steps": self.update_steps,
            "batch": {
                key: value
                for key, value in batch.items()
                if key
                not in {
                    "device",
                    "model_path",
                }
            },
        }
        serialized = json.dumps(
            fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    def _recover_checkpoint(
        self,
        *,
        checkpoint_path: Path,
        update_fingerprint: str,
        updated_samples: int,
    ) -> dict[str, object] | None:
        if not checkpoint_path.exists():
            return None
        metadata_path = checkpoint_path / _UPDATE_METADATA_FILE
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"existing ICLM checkpoint is not recoverable: {checkpoint_path}"
            ) from exc
        if (
            not isinstance(metadata, dict)
            or metadata.get("schema_version") != _UPDATE_METADATA_SCHEMA
            or metadata.get("update_fingerprint") != update_fingerprint
        ):
            raise RuntimeError("existing ICLM checkpoint does not match the requested update")
        result = metadata.get("result")
        if not isinstance(result, dict) or result.get("updated_samples") != updated_samples:
            raise RuntimeError("existing ICLM checkpoint metadata is invalid")
        for metric in (
            "teacher_loss",
            "ewc_loss",
            "total_loss",
            "effective_learning_rate",
        ):
            value = result.get(metric)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise RuntimeError("existing ICLM checkpoint metadata is invalid")
        if "distillation_loss" in result:
            distillation_loss = result["distillation_loss"]
            if (
                isinstance(distillation_loss, bool)
                or not isinstance(distillation_loss, (int, float))
                or not math.isfinite(float(distillation_loss))
            ):
                raise RuntimeError("existing ICLM checkpoint metadata is invalid")
        try:
            recovered_model, recovered_tokenizer = self._load_checkpoint(checkpoint_path)
            if not (checkpoint_path / _CONTINUAL_STATE_FILE).is_file():
                raise RuntimeError("ICLM continual-learning state is missing")
            self._load_continual_learning(
                recovered_model,
                recovered_tokenizer,
                checkpoint_path=checkpoint_path,
            )
        except Exception as exc:
            raise RuntimeError(
                f"existing ICLM checkpoint is not loadable: {checkpoint_path}"
            ) from exc
        _verify_checkpoint_manifest(checkpoint_path)
        _fsync_checkpoint_tree(checkpoint_path)
        _fsync_directory(checkpoint_path.parent)
        return {
            **result,
            "checkpoint_path": str(checkpoint_path),
        }

    def _load(self) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer
        model, tokenizer = self._load_checkpoint(self.model_path)
        self._model = model
        self._tokenizer = tokenizer
        return model, tokenizer

    def embedding_dimension(self) -> int:
        model, _ = self._load()
        embeddings = model.get_input_embeddings()
        dimension = getattr(embeddings, "embedding_dim", None)
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise RuntimeError("ICLM model input embedding dimension is unavailable")
        return dimension

    def _load_checkpoint(
        self,
        checkpoint_path: str | Path,
    ) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
        device = torch.device(
            self.device if torch.cuda.is_available() or self.device == "cpu" else "cpu"
        )
        model.to(device)
        model.eval()
        return model, tokenizer


def validate_ewc_replay(path: str | Path) -> dict[str, object]:
    return dict(_load_ewc_replay(Path(path)).calibration)


def validate_continual_checkpoint(path: str | Path) -> dict[str, object]:
    checkpoint_path = Path(path).expanduser().resolve()
    runner = HuggingFaceCausalLMRunner(model_path=str(checkpoint_path))
    model, tokenizer = runner._load_checkpoint(checkpoint_path)
    _regularizer, calibration = runner._load_continual_learning(
        model,
        tokenizer,
        checkpoint_path=checkpoint_path,
    )
    return dict(calibration)


def bootstrap_validation_checkpoint(checkpoint_path: str | Path) -> Path:
    target_path = Path(checkpoint_path).expanduser().resolve()
    if target_path.exists():
        _validate_bootstrap_target(target_path)
        return target_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(
        tempfile.mkdtemp(
            prefix=f".{target_path.name}.",
            dir=target_path.parent,
        )
    )
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import WhitespaceSplit
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

        pad_marker = "[PAD]"
        eos_marker = "[EOS]"
        unknown_marker = "[UNK]"
        vocabulary = {
            pad_marker: 0,
            eos_marker: 1,
            unknown_marker: 2,
            "C": 3,
            "CC": 4,
            "CCO": 5,
            "CCN": 6,
            "CCC": 7,
            "O": 8,
            "N": 9,
        }
        tokenizer_backend = Tokenizer(
            WordLevel(vocab=vocabulary, unk_token=unknown_marker)
        )
        tokenizer_backend.pre_tokenizer = WhitespaceSplit()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_backend,
            pad_token=pad_marker,
            eos_token=eos_marker,
            unk_token=unknown_marker,
        )
        with torch.random.fork_rng():
            torch.manual_seed(7)
            model = GPT2LMHeadModel(
                GPT2Config(
                    vocab_size=len(tokenizer),
                    n_positions=128,
                    n_ctx=128,
                    n_embd=8,
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
        tokenizer.save_pretrained(temporary_path)
        model.save_pretrained(temporary_path)
        _write_json_artifact(
            temporary_path / _EWC_REPLAY_FILE,
            {
                "schema_version": _EWC_REPLAY_SCHEMA,
                "dataset_id": "moleculeforge-validation-smiles-v1",
                "samples": [
                    {"smiles": "CCO", "weight": 1.0},
                    {"smiles": "CCN", "weight": 1.0},
                    {"smiles": "CCC", "weight": 1.0},
                ],
            },
        )
        _write_json_artifact(
            temporary_path / _VALIDATION_MODEL_METADATA_FILE,
            {
                "schema_version": _VALIDATION_MODEL_SCHEMA,
                "purpose": "synthetic_pipeline_validation_only",
                "seed": 7,
            },
        )
        _write_checkpoint_manifest(temporary_path)
        _fsync_checkpoint_tree(temporary_path)
        temporary_path.replace(target_path)
        _fsync_directory(target_path.parent)
    finally:
        if temporary_path.exists():
            shutil.rmtree(temporary_path)
    _validate_bootstrap_target(target_path)
    return target_path


def _validate_bootstrap_target(checkpoint_path: Path) -> None:
    if not checkpoint_path.is_dir():
        raise RuntimeError(
            f"ICLM validation checkpoint must be a directory: {checkpoint_path}"
        )
    HuggingFaceCausalLMRunner(model_path=str(checkpoint_path))._load_checkpoint(
        checkpoint_path
    )
    _load_ewc_replay(checkpoint_path / _EWC_REPLAY_FILE)
    manifest_path = checkpoint_path / _CHECKPOINT_MANIFEST_FILE
    if manifest_path.exists():
        _verify_checkpoint_manifest(checkpoint_path)


def _positive_generation_integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be a positive integer")
    if normalized <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return normalized


def _finite_unit_interval(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return normalized


def _normalized_teacher_embeddings(
    value: object,
    *,
    expected_rows: int,
) -> list[list[float]] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("teacher_embeddings must be a list")
    if not value:
        return None
    if len(value) != expected_rows:
        raise ValueError("teacher_embeddings rows must match ICLM update samples")
    dimension: int | None = None
    normalized_rows: list[list[float]] = []
    for row in value:
        if not isinstance(row, list):
            raise ValueError("teacher_embeddings rows must be lists")
        if dimension is None:
            dimension = len(row)
        elif len(row) != dimension:
            raise ValueError("teacher_embeddings must be rectangular")
        normalized_row: list[float] = []
        for raw_embedding_value in row:
            if isinstance(raw_embedding_value, bool) or not isinstance(
                raw_embedding_value,
                int | float,
            ):
                raise ValueError("teacher_embeddings values must be numeric")
            embedding_value = float(raw_embedding_value)
            if not math.isfinite(embedding_value):
                raise ValueError("teacher_embeddings values must be finite")
            normalized_row.append(embedding_value)
        normalized_rows.append(normalized_row)
    if dimension == 0:
        return None
    return normalized_rows


def _optional_sampling_seed(kwargs: Mapping[str, object]) -> int | None:
    raw_seed = kwargs.get("sampling_seed", kwargs.get("seed"))
    if raw_seed in (None, ""):
        return None
    if isinstance(raw_seed, bool):
        raise ValueError("sampling_seed must be a non-negative integer")
    try:
        seed = int(raw_seed)
    except (TypeError, ValueError) as exc:
        raise ValueError("sampling_seed must be a non-negative integer") from exc
    if isinstance(raw_seed, float) and not raw_seed.is_integer():
        raise ValueError("sampling_seed must be a non-negative integer")
    if seed < 0:
        raise ValueError("sampling_seed must be a non-negative integer")
    if seed > 2**63 - 1:
        raise ValueError("sampling_seed must be no greater than 2^63 - 1")
    return seed


def _model_context_limit(model: object) -> int | None:
    config = getattr(model, "config", None)
    for attribute in ("max_position_embeddings", "n_positions"):
        value = getattr(config, attribute, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _load_validation_model_metadata(
    checkpoint_path: Path,
) -> dict[str, object] | None:
    metadata_path = checkpoint_path / _VALIDATION_MODEL_METADATA_FILE
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("ICLM validation model metadata is invalid") from exc
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema_version") != _VALIDATION_MODEL_SCHEMA
        or metadata.get("purpose") != "synthetic_pipeline_validation_only"
        or metadata.get("seed") != 7
    ):
        raise RuntimeError("ICLM validation model metadata is invalid")
    return metadata


def _write_json_artifact(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            dict(payload),
            handle,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_ewc_replay(replay_path: Path) -> _EWCReplay:
    if not replay_path.is_file():
        raise RuntimeError(
            f"versioned EWC replay is required for the initial ICLM update: {replay_path}"
        )
    try:
        replay_bytes = replay_path.read_bytes()
        payload = json.loads(replay_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"ICLM EWC replay is not loadable: {replay_path}") from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema_version", "dataset_id", "samples"}
        or payload.get("schema_version") != _EWC_REPLAY_SCHEMA
    ):
        raise RuntimeError("ICLM EWC replay schema is invalid")
    dataset_id = payload.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise RuntimeError("ICLM EWC replay dataset_id is required")
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise RuntimeError("ICLM EWC replay samples must be a non-empty list")
    smiles: list[str] = []
    weights: list[float] = []
    for sample in raw_samples:
        if not isinstance(sample, dict) or set(sample) != {"smiles", "weight"}:
            raise RuntimeError("ICLM EWC replay sample schema is invalid")
        sample_smiles = sample.get("smiles")
        if not isinstance(sample_smiles, str) or not sample_smiles.strip():
            raise RuntimeError("ICLM EWC replay sample smiles is required")
        weight = sample.get("weight")
        if (
            isinstance(weight, bool)
            or not isinstance(weight, int | float)
            or not math.isfinite(float(weight))
            or float(weight) <= 0.0
        ):
            raise RuntimeError("ICLM EWC replay sample weight must be finite and positive")
        try:
            canonical_smiles = _canonical_training_smiles(sample_smiles.strip())
        except ValueError as exc:
            raise RuntimeError("ICLM EWC replay sample smiles is invalid") from exc
        if canonical_smiles in smiles:
            raise RuntimeError("ICLM EWC replay contains duplicate canonical smiles")
        smiles.append(canonical_smiles)
        weights.append(float(weight))
    replay_checksum = hashlib.sha256(replay_bytes).hexdigest()
    return _EWCReplay(
        smiles=smiles,
        weights=weights,
        calibration={
            "schema_version": _EWC_CALIBRATION_SCHEMA,
            "source": "versioned_replay",
            "dataset_id": dataset_id.strip(),
            "sample_count": len(smiles),
            "replay_checksum": f"sha256:{replay_checksum}",
        },
    )


def _validate_base_calibration(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "source",
        "dataset_id",
        "sample_count",
        "replay_checksum",
    }:
        raise RuntimeError("ICLM EWC base calibration is invalid")
    source = value.get("source")
    dataset_id = value.get("dataset_id")
    sample_count = value.get("sample_count")
    replay_checksum = value.get("replay_checksum")
    if (
        value.get("schema_version") != _EWC_CALIBRATION_SCHEMA
        or source not in {"versioned_replay", "legacy_continual_state"}
        or not isinstance(dataset_id, str)
        or not dataset_id.strip()
        or isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 0
        or (source == "versioned_replay" and sample_count == 0)
        or not isinstance(replay_checksum, str)
        or not replay_checksum.startswith("sha256:")
        or len(replay_checksum) != 71
        or any(character not in "0123456789abcdef" for character in replay_checksum[7:])
    ):
        raise RuntimeError("ICLM EWC base calibration is invalid")
    return dict(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_tree_sha256(checkpoint_path: Path) -> str:
    expanded_path = checkpoint_path.expanduser()
    if expanded_path.is_symlink():
        raise RuntimeError("ICLM base checkpoint does not allow symbolic links")
    resolved_path = expanded_path.resolve()
    if not resolved_path.exists():
        raise RuntimeError(f"ICLM base checkpoint is not available: {resolved_path}")
    if resolved_path.is_file():
        return f"sha256:{_sha256_file(resolved_path)}"
    checkpoint_entries = list(resolved_path.rglob("*"))
    if any(path.is_symlink() for path in checkpoint_entries):
        raise RuntimeError("ICLM base checkpoint does not allow symbolic links")
    artifacts = sorted(
        (path for path in checkpoint_entries if path.is_file()),
        key=lambda path: path.relative_to(resolved_path).as_posix(),
    )
    if not artifacts:
        raise RuntimeError("ICLM base checkpoint does not contain artifacts")
    digest = hashlib.sha256()
    for artifact_path in artifacts:
        relative_path = artifact_path.relative_to(resolved_path).as_posix().encode("utf-8")
        digest.update(len(relative_path).to_bytes(8, "big"))
        digest.update(relative_path)
        digest.update(artifact_path.stat().st_size.to_bytes(8, "big"))
        with artifact_path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _extract_valid_smiles(text: str) -> str | None:
    candidates = []
    for token in text.replace("\n", " ").split():
        candidates.append(token)
    candidates.append(text.strip())
    for candidate in candidates:
        cleaned = _clean_smiles(candidate)
        if cleaned and _is_valid_smiles(cleaned):
            return cleaned
    return None


def _clean_smiles(value: str) -> str:
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789[]()=#@+-/\\\\.%")
    return "".join(char for char in value.strip() if char in allowed)


def _is_valid_smiles(smiles: str) -> bool:
    if not smiles or Chem is None:
        return False
    return Chem.MolFromSmiles(smiles) is not None


def _canonical_training_smiles(smiles: str) -> str:
    if Chem is None:
        raise RuntimeError("RDKit is required to validate ICLM training molecules")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("ICLM update sample smiles must be a valid molecule")
    return str(Chem.MolToSmiles(molecule, canonical=True))


def _write_checkpoint_manifest(checkpoint_path: Path) -> None:
    manifest_path = checkpoint_path / _CHECKPOINT_MANIFEST_FILE
    payload = {
        "schema_version": _CHECKPOINT_MANIFEST_SCHEMA,
        "artifacts": _checkpoint_manifest_artifacts(checkpoint_path),
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _verify_checkpoint_manifest(checkpoint_path: Path) -> None:
    manifest_path = checkpoint_path / _CHECKPOINT_MANIFEST_FILE
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("existing ICLM checkpoint manifest is invalid") from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema_version", "artifacts"}
        or payload.get("schema_version") != _CHECKPOINT_MANIFEST_SCHEMA
        or not isinstance(payload.get("artifacts"), list)
    ):
        raise RuntimeError("existing ICLM checkpoint manifest is invalid")
    expected_artifacts = payload["artifacts"]
    actual_artifacts = _checkpoint_manifest_artifacts(checkpoint_path)
    if expected_artifacts != actual_artifacts:
        raise RuntimeError("existing ICLM checkpoint manifest does not match artifacts")


def _checkpoint_manifest_artifacts(checkpoint_path: Path) -> list[dict[str, object]]:
    artifacts = []
    for artifact_path in sorted(
        (
            path
            for path in checkpoint_path.rglob("*")
            if path.is_file()
            and path != checkpoint_path / _CHECKPOINT_MANIFEST_FILE
        ),
        key=lambda path: path.relative_to(checkpoint_path).as_posix(),
    ):
        if artifact_path.is_symlink():
            raise RuntimeError("ICLM checkpoint manifest does not allow symbolic links")
        relative_path = artifact_path.relative_to(checkpoint_path).as_posix()
        digest = hashlib.sha256()
        with artifact_path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        artifacts.append(
            {
                "path": relative_path,
                "size": artifact_path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    if not artifacts:
        raise RuntimeError("ICLM checkpoint manifest requires artifacts")
    return artifacts


def _fsync_checkpoint_tree(checkpoint_path: Path) -> None:
    for artifact_path in sorted(
        (path for path in checkpoint_path.rglob("*") if path.is_file()),
        key=lambda path: str(path),
    ):
        with artifact_path.open("rb") as handle:
            os.fsync(handle.fileno())
    directories = [path for path in checkpoint_path.rglob("*") if path.is_dir()]
    for directory in sorted(
        directories,
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        _fsync_directory(directory)
    _fsync_directory(checkpoint_path)


def _fsync_directory(directory: Path) -> None:
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, directory_flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _run_json_command() -> None:
    payload = json.load(sys.stdin)
    if not isinstance(payload, dict):
        raise ValueError("ICLM update command input must be a JSON object")
    model_path = str(payload.get("model_path", "")).strip()
    if not model_path:
        raise ValueError("ICLM update command requires model_path")
    device = str(payload.get("device", "")).strip()
    if not device:
        raise ValueError("ICLM update command requires device")
    result = HuggingFaceCausalLMRunner(
        model_path=model_path,
        device=device,
    ).update(payload)
    json.dump(result, sys.stdout, sort_keys=True, allow_nan=False)
    sys.stdout.write("\n")


def _main(argv: list[str]) -> None:
    if argv:
        if len(argv) != 2 or argv[0] != "--bootstrap-validation-model":
            raise ValueError("unsupported ICLM runner command")
        path = bootstrap_validation_checkpoint(argv[1])
        sys.stdout.write(f"{path}\n")
        return
    _run_json_command()


if __name__ == "__main__":
    _main(sys.argv[1:])
