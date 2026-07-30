"""Low-level task learner for incremental CLM."""

import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

import torch


class OnlineLearner:
    def __init__(
        self,
        model,
        learning_rate=1e-4,
        *,
        checkpoint_directory: str | Path,
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.checkpoint_directory = Path(checkpoint_directory)
        self.last_task_loss = 0.0
        self.last_total_loss = 0.0

    def update(self, batch):
        samples, target_version = self._checkpoint_contract(batch)
        if any(
            self._batch_value(batch, field) not in (None, [])
            for field in ("teacher_embeddings", "kd_teacher_embeddings")
        ):
            raise ValueError(
                "teacher supervision requires HuggingFaceCausalLMRunner"
            )
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(batch)
        if isinstance(output, tuple):
            task_loss = output[0]
        else:
            task_loss = output
        if (
            not torch.is_tensor(task_loss)
            or task_loss.numel() != 1
            or not math.isfinite(float(task_loss.detach().cpu().item()))
        ):
            raise ValueError("online learner model must return one finite task loss")
        total_loss = task_loss
        total_loss.backward()
        self.optimizer.step()
        self.last_task_loss = float(task_loss.detach().cpu().item())
        self.last_total_loss = float(total_loss.detach().cpu().item())
        checkpoint_path = self._write_checkpoint(target_version)
        return {
            "checkpoint_path": str(checkpoint_path),
            "updated_samples": len(samples),
            "task_loss": self.last_task_loss,
            "total_loss": self.last_total_loss,
        }

    @staticmethod
    def _checkpoint_contract(batch) -> tuple[list[object], str]:
        if not isinstance(batch, Mapping):
            raise ValueError("online learner batch must be a mapping")
        samples = batch.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError("online learner batch requires non-empty samples")
        target_version = batch.get("target_checkpoint_version")
        if not isinstance(target_version, str) or not target_version.strip():
            raise ValueError("online learner batch requires target_checkpoint_version")
        target_version = target_version.strip()
        if target_version in {".", ".."} or "/" in target_version or "\\" in target_version:
            raise ValueError("target_checkpoint_version must be a file-safe name")
        return samples, target_version

    def _write_checkpoint(self, target_version: str) -> Path:
        self.checkpoint_directory.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.checkpoint_directory / f"{target_version}.pt"
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "target_checkpoint_version": target_version,
        }
        file_descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{checkpoint_path.name}.",
            suffix=".tmp",
            dir=self.checkpoint_directory,
        )
        try:
            with os.fdopen(file_descriptor, "wb") as handle:
                torch.save(checkpoint, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, checkpoint_path)
            directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            directory_descriptor = os.open(
                self.checkpoint_directory,
                directory_flags,
            )
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            Path(temporary_path).unlink(missing_ok=True)
        return checkpoint_path

    @staticmethod
    def _batch_value(batch, key):
        if isinstance(batch, Mapping):
            return batch.get(key)
        return getattr(batch, key, None)
