"""Online learner for incremental CLM."""
from collections.abc import Mapping

import torch
import torch.nn.functional as F


class OnlineLearner:
    def __init__(self, model, learning_rate=1e-4):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.last_task_loss = 0.0
        self.last_kd_loss = 0.0
        self.last_total_loss = 0.0

    def update(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(batch)
        student_embeddings = None
        if isinstance(output, tuple):
            task_loss = output[0]
            if len(output) > 1:
                student_embeddings = output[1]
        else:
            task_loss = output
        kd_loss = self._teacher_embedding_loss(batch, student_embeddings)
        total_loss = task_loss + self._kd_weight(batch) * kd_loss
        total_loss.backward()
        self.optimizer.step()
        self.last_task_loss = float(task_loss.detach().cpu().item())
        self.last_kd_loss = float(kd_loss.detach().cpu().item())
        self.last_total_loss = float(total_loss.detach().cpu().item())
        return self.last_total_loss

    def _teacher_embedding_loss(self, batch, student_embeddings):
        teacher_embeddings = self._batch_value(batch, "kd_teacher_embeddings")
        kd_weight = self._kd_weight(batch)
        if kd_weight <= 0.0 or teacher_embeddings in (None, []):
            return torch.zeros((), dtype=torch.float32, device=self._loss_device())
        if student_embeddings is None:
            raise ValueError("model must return student embeddings when KD is requested")
        student = student_embeddings
        if not torch.is_tensor(student):
            student = torch.tensor(student, dtype=torch.float32, device=self._loss_device())
        teacher = torch.tensor(
            teacher_embeddings,
            dtype=student.dtype,
            device=student.device,
        )
        if teacher.shape != student.shape:
            raise ValueError(
                "kd_teacher_embeddings shape must match model student embeddings"
            )
        return F.mse_loss(student, teacher)

    @staticmethod
    def _batch_value(batch, key):
        if isinstance(batch, Mapping):
            return batch.get(key)
        return getattr(batch, key, None)

    def _kd_weight(self, batch) -> float:
        value = self._batch_value(batch, "kd_weight")
        if value in (None, ""):
            return 0.0
        weight = float(value)
        if weight < 0.0:
            raise ValueError("kd_weight must be non-negative")
        return weight

    def _loss_device(self):
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")
