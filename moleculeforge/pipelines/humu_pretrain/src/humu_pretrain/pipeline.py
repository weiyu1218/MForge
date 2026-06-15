"""HUMU joint manifold pretraining pipeline with real training loop and checkpointing."""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from statistics import median

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as functional
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Retrieval heads whose positive is identical to the anchor (SimCSE-style:
# anchor and positive encode the same input, differing only by dropout). In eval
# mode dropout is disabled, so their top-1 retrieval is trivially 1.0; they are
# reported individually but excluded from the aggregated retrieval_top1.
_DEGENERATE_RETRIEVAL_KEYS = {
    "mol_self_retrieval_top1",
    "protac_component_library_retrieval_top1",
}

_DDP_DUMMY_POCKET = {
    "pdb_id": "__ddp_dummy_pocket__",
    "coords": [[0.0, 0.0, 0.0]],
    "elements": ["C"],
    "residue_types": ["ALA"],
    "protein_sequence": "A",
}
_DDP_DUMMY_ROUTE = {
    "id": "__ddp_dummy_route__",
    "reactions": ["C>>C"],
    "steps": 1,
    "intermediates": [],
    "score": 0.0,
}


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int


@dataclass(frozen=True)
class TrainingResumeState:
    start_epoch: int
    epoch_step: int
    best_loss: float
    epoch_loss_sum: float = 0.0
    epoch_loss_count: int = 0


@dataclass(frozen=True)
class ActivityRecord:
    ligand_smiles: str
    target_id: str
    activity_value: float
    activity_type: str
    assay_id: str


async def run(config: dict) -> dict:
    """Execute the full HUMU pre-training pipeline with real training."""
    cfg = _validate_config(config)
    distributed = _distributed_context_from_env()
    device = _get_device(cfg, distributed)
    _configure_cuda_backends(cfg, device)
    _setup_distributed(distributed, device, cfg)

    # Create encoders
    encoders = _build_encoders(cfg, device)
    encoders = _wrap_distributed(
        encoders,
        distributed,
        device,
        find_unused_parameters=bool(cfg.get("ddp_find_unused_parameters", False)),
    )
    models = list(encoders.values())

    start_epoch = 0
    best_loss = float("inf")
    output_dir = Path(cfg.get("output_dir", "checkpoints/humu/"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data loaders
    from humu_pretrain.data_loader import create_dataloaders
    loaders = create_dataloaders(cfg)
    loaders = _prepare_distributed_loaders(loaders, distributed)

    # Optimizer and scheduler
    optimizer = AdamW(
        [p for m in models for p in m.parameters() if p.requires_grad],
        lr=cfg["learning_rate"],
        weight_decay=cfg.get("weight_decay", 1e-5),
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    resume_from = cfg.get("resume_from")
    resume_epoch_step = 0
    resume_epoch_loss_sum = 0.0
    resume_epoch_loss_count = 0
    if resume_from:
        if not os.path.exists(resume_from):
            raise FileNotFoundError(f"resume_from checkpoint does not exist: {resume_from}")
        resume_state = _load_training_checkpoint(
            encoders,
            resume_from,
            device,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        start_epoch = resume_state.start_epoch
        best_loss = resume_state.best_loss
        resume_epoch_step = resume_state.epoch_step
        resume_epoch_loss_sum = resume_state.epoch_loss_sum
        resume_epoch_loss_count = resume_state.epoch_loss_count
        if _is_main_process(distributed):
            print(
                f"Resumed checkpoint {resume_from}: "
                f"starting_epoch={start_epoch + 1}, "
                f"starting_batch={resume_epoch_step + 1}, "
                f"best_loss={best_loss:.4f}"
            )

    # Mixed precision
    use_amp = bool(cfg.get("use_amp", False)) and device.type == "cuda"
    amp_dtype = _amp_dtype_from_config(cfg)
    scaler = (
        torch.amp.GradScaler("cuda")
        if use_amp and amp_dtype is torch.float16
        else None
    )

    epochs = cfg["epochs"]
    save_every = cfg.get("save_every_n_epochs", 5)
    save_every_steps = int(cfg.get("save_every_n_steps", 0) or 0)
    gradient_clip = cfg.get("gradient_clip_norm", 1.0)
    warmup_steps = int(cfg.get("warmup_steps", 0) or 0)
    base_lr = cfg["learning_rate"]
    logging_cfg = cfg.get("logging", {})
    log_every = max(1, int(logging_cfg.get("log_every_n_steps", 50)))
    preserve_step_logs = bool(logging_cfg.get("preserve_step_logs", False))
    skip_bad_batches = bool(cfg.get("skip_bad_batches", True))
    max_skipped_batches = int(cfg.get("max_skipped_batches", 1000) or 0)
    skipped_batches = 0
    early_stopping = _early_stopping_state(cfg)
    stopped_early = False
    early_stop_reason = None
    completed_epochs = start_epoch

    for epoch in range(start_epoch, epochs):
        should_stop_early = False
        _set_sampler_epoch(loaders, epoch)
        epoch_start = time.time()
        paired_loader = loaders.get("paired")
        n_batches = len(paired_loader) if paired_loader else 0
        if n_batches == 0:
            break

        epoch_resume_step = resume_epoch_step if epoch == start_epoch else 0
        if epoch_resume_step > n_batches:
            raise ValueError(
                "resume checkpoint epoch_step exceeds current epoch batch count: "
                f"epoch_step={epoch_resume_step}, n_batches={n_batches}"
            )
        epoch_loss_sum = resume_epoch_loss_sum if epoch == start_epoch else 0.0
        epoch_loss_count = resume_epoch_loss_count if epoch == start_epoch else 0
        paired_iter = iter(paired_loader)
        if epoch_resume_step > 0:
            for _ in range(epoch_resume_step):
                try:
                    next(paired_iter)
                except StopIteration as exc:
                    raise RuntimeError(
                        "resume checkpoint cannot skip completed batches"
                    ) from exc
            if _is_main_process(distributed):
                print(
                    "Skipped previously completed HUMU batches "
                    f"epoch={epoch + 1} "
                    f"batches={epoch_resume_step}/{n_batches}",
                    flush=True,
                )

        for step in range(epoch_resume_step, n_batches):
            step_start = time.time()
            dataloader_start = time.time()
            local_batch_error = None
            losses = None
            step_loss = None
            forward_time = 0.0
            try:
                paired_batch = next(paired_iter)
            except Exception as exc:  # noqa: BLE001
                paired_batch = None
                local_batch_error = exc
            dataloader_time = time.time() - dataloader_start
            optimizer.zero_grad()

            if local_batch_error is None:
                forward_start = time.time()
                try:
                    with (
                        torch.amp.autocast("cuda", dtype=amp_dtype)
                        if use_amp
                        else _null_context()
                    ):
                        losses = _forward_paired_batch(encoders, paired_batch, cfg)
                        step_loss = losses["total"]
                except Exception as exc:  # noqa: BLE001
                    local_batch_error = exc
                forward_time = time.time() - forward_start

            batch_failed = _distributed_batch_failed(
                distributed,
                device,
                local_failed=local_batch_error is not None,
            )
            if batch_failed:
                skipped_batches += 1
                optimizer.zero_grad(set_to_none=True)
                if local_batch_error is not None:
                    _log_batch_skip(
                        epoch,
                        step + 1,
                        n_batches,
                        distributed,
                        local_batch_error,
                        skipped_batches,
                    )
                if not skip_bad_batches:
                    raise RuntimeError(
                        "HUMU batch failed on at least one distributed rank"
                    ) from local_batch_error
                if max_skipped_batches > 0 and skipped_batches > max_skipped_batches:
                    raise RuntimeError(
                        f"HUMU skipped {skipped_batches} batches, "
                        f"exceeding max_skipped_batches={max_skipped_batches}"
                    ) from local_batch_error
                continue

            warmup_active = _apply_warmup_lr(
                optimizer,
                epoch=epoch,
                step=step,
                n_batches=n_batches,
                warmup_steps=warmup_steps,
                base_lr=base_lr,
            )
            backward_start = time.time()
            if scaler:
                scaler.scale(step_loss).backward()
                if gradient_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        [p for m in models for p in m.parameters() if p.requires_grad],
                        gradient_clip,
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                step_loss.backward()
                if gradient_clip > 0:
                    nn.utils.clip_grad_norm_(
                        [p for m in models for p in m.parameters() if p.requires_grad],
                        gradient_clip,
                    )
                optimizer.step()
            backward_time = time.time() - backward_start

            if not warmup_active:
                _apply_lr_schedule(
                    scheduler,
                    epoch=epoch,
                    step=step,
                    n_batches=n_batches,
                )
            step_loss_value = _loss_float(losses["total"])
            epoch_loss_sum += step_loss_value
            epoch_loss_count += 1

            if step % log_every == 0 and _is_main_process(distributed):
                _log_step(
                    epoch,
                    step + 1,
                    n_batches,
                    losses,
                    optimizer.param_groups[0]["lr"],
                    rank=distributed.rank,
                    step_time=time.time() - step_start,
                    dataloader_time=dataloader_time,
                    forward_time=forward_time,
                    backward_time=backward_time,
                    gpu_stats=_gpu_stats(device),
                    preserve=preserve_step_logs,
                )

            global_step = epoch * n_batches + step + 1
            if (
                save_every_steps > 0
                and global_step % save_every_steps == 0
                and _is_main_process(distributed)
            ):
                _save_checkpoint(
                    encoders,
                    optimizer,
                    scheduler,
                    epoch,
                    step_loss_value,
                    output_dir / f"checkpoint_step_{global_step:08d}.pt",
                    checkpoint_type="step",
                    global_step=global_step,
                    epoch_step=step + 1,
                    n_batches=n_batches,
                    best_loss=best_loss,
                    epoch_loss_sum=epoch_loss_sum,
                    epoch_loss_count=epoch_loss_count,
                )

        # Epoch summary
        avg_loss = epoch_loss_sum / max(epoch_loss_count, 1)
        if _is_main_process(distributed):
            if not preserve_step_logs:
                _clear_progress_line()
            print(
                f"Epoch {epoch + 1}/{epochs}: train_loss={avg_loss:.4f}, "
                f"time={time.time() - epoch_start:.1f}s"
            )

        if avg_loss < best_loss and _is_main_process(distributed):
            best_loss = avg_loss
            _save_checkpoint(
                encoders,
                optimizer,
                scheduler,
                epoch,
                avg_loss,
                output_dir / "best_model.pt",
                checkpoint_type="epoch",
                best_loss=best_loss,
                epoch_loss_sum=epoch_loss_sum,
                epoch_loss_count=epoch_loss_count,
            )

        if _should_validate_epoch(epoch, cfg, loaders):
            validation_metrics = _validate_epoch(
                encoders,
                loaders["validation"],
                cfg,
                device,
                distributed,
            )
            if _is_main_process(distributed):
                metrics_path = _write_validation_metrics(
                    output_dir,
                    epoch,
                    validation_metrics,
                )
                print(_format_validation_summary(epoch, epochs, validation_metrics, metrics_path))
            should_stop_early, early_stopping = _update_early_stopping(
                early_stopping,
                validation_metrics,
                epoch,
            )
            if should_stop_early:
                stopped_early = True
                early_stop_reason = _format_early_stop_reason(early_stopping)
                if _is_main_process(distributed):
                    print(early_stop_reason)

        if (epoch + 1) % save_every == 0 and _is_main_process(distributed):
            _save_checkpoint(
                encoders,
                optimizer,
                scheduler,
                epoch,
                avg_loss,
                output_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt",
                checkpoint_type="epoch",
                best_loss=best_loss,
                epoch_loss_sum=epoch_loss_sum,
                epoch_loss_count=epoch_loss_count,
            )
            _rotate_checkpoints(output_dir, cfg.get("keep_last_n", 3))
        completed_epochs = epoch + 1
        if should_stop_early:
            break

    # Save final model
    if _is_main_process(distributed):
        _save_checkpoint(
            encoders,
            optimizer,
            scheduler,
            completed_epochs,
            best_loss,
            output_dir / "final_model.pt",
            checkpoint_type="final",
            best_loss=best_loss,
        )
    _cleanup_distributed(distributed)

    return {
        "pipeline": "humu_pretrain",
        "status": "completed",
        "epochs_completed": completed_epochs,
        "best_loss": best_loss,
        "output_dir": str(output_dir),
        "stopped_early": stopped_early,
        "early_stop_reason": early_stop_reason,
    }


def combined_loss(mol_emb, pocket_emb, route_emb, lambda_curvature=0.01) -> dict:
    """Compute the three-tower joint loss with curvature regularization."""
    losses = _compute_losses(
        mol_emb,
        pocket_emb,
        route_emb,
        {
            "mol_pocket": 1.0,
            "mol_route": 1.0,
            "pocket_route": 1.0,
            "curvature_reg": lambda_curvature,
        },
        {"temperature": 0.07, "negative_sampling": "in_batch"},
        route_mol_emb=mol_emb,
        pocket_route_pocket_emb=pocket_emb,
        pocket_route_route_emb=route_emb,
    )
    return {key: _loss_float(value) for key, value in losses.items()}


# ── Stub async wrappers (kept for backward compat) ──────────────────────────

async def pretrain_molecule_encoder(cfg: dict) -> dict:
    raise RuntimeError("Use humu_pretrain.pipeline.run with configured molecule data")


async def pretrain_pocket_encoder(cfg: dict) -> dict:
    raise RuntimeError("Use humu_pretrain.pipeline.run with configured pocket data")


async def pretrain_route_encoder(cfg: dict) -> dict:
    raise RuntimeError("Use humu_pretrain.pipeline.run with configured route data")


# ── Internal helpers ────────────────────────────────────────────────────────

def _build_encoders(cfg: dict, device: torch.device) -> dict[str, nn.Module]:
    """Build or import HUMU encoder towers and auxiliary source projectors."""
    dim = cfg.get("embed_dim", 129) - 1
    curvature = cfg.get("curvature", 1.0)
    learnable_curvature = bool(cfg.get("learnable_curvature", False))

    from mf_encoders.humu_mol.encoder import HUMUMoleculeEncoder
    mol_cfg = cfg.get("encoders", {}).get("mol", {})
    mol = HUMUMoleculeEncoder(
        dim=dim,
        curvature=curvature,
        learnable_curvature=learnable_curvature,
        hidden_dim=mol_cfg.get("hidden_dim"),
        n_layers=int(mol_cfg.get("n_layers", 2)),
        n_heads=int(mol_cfg.get("n_heads", 8)),
        dropout=float(mol_cfg.get("dropout", 0.0)),
        use_3d_geometry=bool(mol_cfg.get("use_3d_geometry", True)),
    )
    mol = _wrap_as_module(mol, dim, device, curvature)

    from mf_encoders.humu_pocket.encoder import HUMUPocketEncoder
    pocket_cfg = cfg.get("encoders", {}).get("pocket", {})
    pocket = HUMUPocketEncoder(
        dim=dim,
        curvature=curvature,
        learnable_curvature=learnable_curvature,
        hidden_dim=pocket_cfg.get("hidden_dim"),
        n_layers=int(pocket_cfg.get("n_layers", 1)),
        n_heads=int(pocket_cfg.get("n_heads", 8)),
        dropout=float(pocket_cfg.get("dropout", 0.0)),
        radius_angstrom=float(pocket_cfg.get("radius_angstrom", 20.0)),
        max_neighbors=(
            int(pocket_cfg["max_neighbors"])
            if pocket_cfg.get("max_neighbors") is not None
            else None
        ),
        use_3d_geometry=bool(pocket_cfg.get("use_3d_geometry", True)),
        use_esm2=bool(pocket_cfg.get("use_esm2", False)),
        esm2_checkpoint=pocket_cfg.get("esm2_checkpoint"),
        esm2_layer=int(pocket_cfg.get("esm2_layer", 33)),
        esm2_dim=int(pocket_cfg.get("esm2_dim", 1280)),
        esm2_batch_tokens=int(pocket_cfg.get("esm2_batch_tokens", 8192)),
        esm2_max_sequence_length=(
            int(pocket_cfg["esm2_max_sequence_length"])
            if pocket_cfg.get("esm2_max_sequence_length") is not None
            else None
        ),
        esm2_required_sources=pocket_cfg.get("esm2_required_sources"),
    )
    pocket = _wrap_as_module(pocket, dim, device, curvature)

    from mf_encoders.humu_route.encoder import HUMURouteEncoder
    route_cfg = cfg.get("encoders", {}).get("route", {})
    route = HUMURouteEncoder(
        dim=dim,
        curvature=curvature,
        learnable_curvature=learnable_curvature,
        hidden_dim=route_cfg.get("hidden_dim"),
        n_layers=int(route_cfg.get("n_layers", 2)),
        n_heads=int(route_cfg.get("n_heads", 8)),
        dropout=float(route_cfg.get("dropout", 0.0)),
        use_tree_pooling=bool(route_cfg.get("use_tree_pooling", True)),
    )
    route = _wrap_as_module(route, dim, device, curvature)

    feature_cfg = cfg.get("encoders", {}).get("protac_feature", {})
    protac_feature = _FeatureVectorEncoder(
        input_dim=int(feature_cfg.get("protac_dim", 167)),
        dim=dim,
        hidden_dim=int(feature_cfg.get("hidden_dim", mol_cfg.get("hidden_dim") or dim)),
        dropout=float(feature_cfg.get("dropout", 0.0)),
        curvature=curvature,
    ).to(device)
    protac_context_feature = _FeatureVectorEncoder(
        input_dim=int(feature_cfg.get("context_dim", 30)),
        dim=dim,
        hidden_dim=int(feature_cfg.get("hidden_dim", mol_cfg.get("hidden_dim") or dim)),
        dropout=float(feature_cfg.get("dropout", 0.0)),
        curvature=curvature,
    ).to(device)

    return {
        "mol": mol,
        "pocket": pocket,
        "route": route,
        "protac_feature": protac_feature,
        "protac_context_feature": protac_context_feature,
    }


class _FeatureVectorEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        dim: int,
        hidden_dim: int,
        dropout: float,
        curvature: float,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim + 1),
        )
        from mf_humu.manifold.lorentz import LorentzManifold
        self.manifold = LorentzManifold(curvature=curvature)

    def forward(self, features) -> torch.Tensor:
        tensor = torch.as_tensor(
            features,
            dtype=torch.float32,
            device=next(self.parameters()).device,
        )
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[-1] != self.input_dim:
            raise ValueError(
                f"feature vector encoder expected shape (*, {self.input_dim})"
            )
        return self.manifold._project(self.net(tensor))


def _wrap_as_module(
    encoder_obj,
    dim: int,
    device: torch.device,
    curvature: float = 1.0,
) -> nn.Module:
    """Wrap a class-based encoder as a trainable nn.Module with a learnable projection."""
    class _EncoderWrapper(nn.Module):
        def __init__(self, inner, dim, dev):
            super().__init__()
            self.inner = inner
            self.proj = nn.Linear(dim + 1, dim + 1)
            self.device = dev
            from mf_humu.manifold.lorentz import LorentzManifold
            self._manifold = LorentzManifold(curvature=curvature)
            self.to(dev)

        def forward(self, smiles_or_data):
            if isinstance(smiles_or_data, str):
                emb = self.inner.encode(smiles_or_data)
            elif isinstance(smiles_or_data, list):
                emb = self.inner.encode_batch(smiles_or_data)
            else:
                emb = self.inner.encode(smiles_or_data)
            if emb.device != self.device:
                emb = emb.to(self.device)
            out = self.proj(emb)
            return self._manifold._project(out)

        def encode_batch(self, smiles_list):
            return self.forward(smiles_list)

        def encode(self, data):
            return self.forward(data)

    return _EncoderWrapper(encoder_obj, dim, device)


def _distributed_context_from_env() -> DistributedContext:
    """Read torchrun distributed settings without initializing process groups."""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return DistributedContext(
        enabled=world_size > 1,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )


def _get_device(cfg: dict, distributed: DistributedContext | None = None) -> torch.device:
    device_str = cfg.get("device", "cpu")
    if device_str == "cuda" and torch.cuda.is_available():
        if distributed and distributed.enabled:
            torch.cuda.set_device(distributed.local_rank)
            return torch.device("cuda", distributed.local_rank)
        return torch.device("cuda")
    return torch.device("cpu")


def _setup_distributed(
    distributed: DistributedContext,
    device: torch.device,
    cfg: dict,
) -> None:
    if not distributed.enabled or dist.is_initialized():
        return
    default_backend = "nccl" if device.type == "cuda" else "gloo"
    timeout_seconds = int(cfg.get("distributed_timeout_seconds", 3600) or 3600)
    if timeout_seconds <= 0:
        raise ValueError("distributed_timeout_seconds must be positive")
    dist.init_process_group(
        backend=cfg.get("distributed_backend", default_backend),
        rank=distributed.rank,
        world_size=distributed.world_size,
        timeout=timedelta(seconds=timeout_seconds),
    )


def _cleanup_distributed(distributed: DistributedContext) -> None:
    if distributed.enabled and dist.is_initialized():
        dist.destroy_process_group()


def _wrap_distributed(
    encoders: dict[str, nn.Module],
    distributed: DistributedContext,
    device: torch.device,
    find_unused_parameters: bool = False,
) -> dict[str, nn.Module]:
    if not distributed.enabled:
        return encoders
    wrapped = {}
    device_ids = [distributed.local_rank] if device.type == "cuda" else None
    for name, model in encoders.items():
        wrapped[name] = DistributedDataParallel(
            model,
            device_ids=device_ids,
            find_unused_parameters=find_unused_parameters,
        )
    return wrapped


def _prepare_distributed_loaders(loaders: dict, distributed: DistributedContext) -> dict:
    if not distributed.enabled:
        return loaders
    from torch.utils.data import DataLoader, RandomSampler
    from torch.utils.data.distributed import DistributedSampler

    prepared = {}
    for name, loader in loaders.items():
        if getattr(loader, "batch_size", None) is None:
            batch_sampler = getattr(loader, "batch_sampler", None)
            try:
                from humu_pretrain.data_loader import TargetRatioMultiSourceBatchSampler
            except ImportError:
                TargetRatioMultiSourceBatchSampler = None
            if TargetRatioMultiSourceBatchSampler is not None and isinstance(
                batch_sampler,
                TargetRatioMultiSourceBatchSampler,
            ):
                prepared[name] = DataLoader(
                    loader.dataset,
                    batch_sampler=TargetRatioMultiSourceBatchSampler(
                        loader.dataset,
                        batch_size=batch_sampler.batch_size,
                        objective_ratios=batch_sampler.objective_ratios,
                        steps_per_epoch=len(batch_sampler),
                        alpha=batch_sampler.alpha,
                        rank=distributed.rank,
                        world_size=distributed.world_size,
                        seed=batch_sampler.seed,
                    ),
                    num_workers=loader.num_workers,
                    pin_memory=getattr(loader, "pin_memory", False),
                    collate_fn=loader.collate_fn,
                )
            else:
                prepared[name] = loader
            continue
        shuffle = isinstance(getattr(loader, "sampler", None), RandomSampler)
        sampler = DistributedSampler(
            loader.dataset,
            num_replicas=distributed.world_size,
            rank=distributed.rank,
            shuffle=shuffle,
            drop_last=False,
        )
        prepared[name] = DataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            sampler=sampler,
            num_workers=loader.num_workers,
            pin_memory=getattr(loader, "pin_memory", False),
            collate_fn=loader.collate_fn,
        )
    return prepared


def _apply_lr_schedule(
    scheduler,
    *,
    epoch: int,
    step: int,
    n_batches: int,
) -> None:
    """Advance the cosine schedule after the warmup window."""
    scheduler.step(epoch + step / max(n_batches, 1))


def _apply_warmup_lr(
    optimizer,
    *,
    epoch: int,
    step: int,
    n_batches: int,
    warmup_steps: int,
    base_lr: float,
) -> bool:
    global_step = epoch * n_batches + step
    if warmup_steps <= 0 or global_step >= warmup_steps:
        return False
    lr = base_lr * float(global_step + 1) / float(warmup_steps)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return True


def _set_sampler_epoch(loaders: dict, epoch: int) -> None:
    for loader in loaders.values():
        for attr in ("sampler", "batch_sampler"):
            candidate = getattr(loader, attr, None)
            if hasattr(candidate, "set_epoch"):
                candidate.set_epoch(epoch)


def _next_or_restart(loader, iterator):
    """Return the next batch, restarting a non-empty loader after exhaustion."""
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _is_main_process(distributed: DistributedContext) -> bool:
    return not distributed.enabled or distributed.rank == 0


def _distributed_batch_failed(
    distributed: DistributedContext,
    device: torch.device,
    local_failed: bool,
) -> bool:
    if not distributed.enabled or not dist.is_initialized():
        return local_failed
    tensor_device = device if device.type == "cuda" else torch.device("cpu")
    failed = torch.tensor(
        [1 if local_failed else 0],
        dtype=torch.int32,
        device=tensor_device,
    )
    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    return bool(failed.item())


def _log_batch_skip(
    epoch: int,
    step: int,
    n_batches: int,
    distributed: DistributedContext,
    error: Exception,
    skipped_batches: int,
) -> None:
    print(
        "Skipped HUMU batch "
        f"rank={distributed.rank} "
        f"epoch={epoch + 1} "
        f"batch={step}/{n_batches} "
        f"skipped_batches={skipped_batches} "
        f"error={type(error).__name__}: {error}",
        flush=True,
    )


def _module(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _encode_model(model: nn.Module, data) -> torch.Tensor:
    return model(data)


def _forward_paired_batch(
    encoders: dict[str, nn.Module],
    paired_batch: dict,
    cfg: dict,
    return_context: bool = False,
):
    mol_encoder = encoders["mol"]
    pocket_encoder = encoders["pocket"]
    route_encoder = encoders["route"]
    smiles_list = paired_batch.get("ligand_smiles", [])
    pair_types = paired_batch.get("pair_type", [])
    protac_ternary_indices = [
        i for i, pair_type in enumerate(pair_types)
        if pair_type == "protac_ternary"
    ]
    protac_ternary_feature_indices = _indices_with_batch_value(
        paired_batch,
        protac_ternary_indices,
        "protac_feature",
    )
    protac_ternary_feature_index_set = set(protac_ternary_feature_indices)
    protac_ternary_smiles_indices = [
        index
        for index in protac_ternary_indices
        if index not in protac_ternary_feature_index_set
    ]
    pdc_component_indices = [
        i for i, pair_type in enumerate(pair_types)
        if pair_type == "pdc_component"
    ]
    protac_component_library_indices = [
        i for i, pair_type in enumerate(pair_types)
        if pair_type == "protac_component_library"
    ]
    pdc_component_molecule_indices = _indices_with_batch_value(
        paired_batch,
        pdc_component_indices,
        "ligand_smiles",
    )
    pdc_component_molecule_index_set = set(pdc_component_molecule_indices)
    pdc_component_peptide_indices = [
        index
        for index in pdc_component_indices
        if index not in pdc_component_molecule_index_set
    ]
    mol_required_indices = [
        i for i, pair_type in enumerate(pair_types)
        if pair_type
        in {
            "mol_self",
            "mol_pocket",
            "mol_route",
            "mol_pocket_route",
            "activity_pair",
            "protac_component",
        }
    ] + protac_ternary_smiles_indices + pdc_component_molecule_indices
    mol_emb = _encode_molecule_indices(mol_encoder, smiles_list, mol_required_indices)
    pocket_indices = [
        i for i, pair_type in enumerate(pair_types)
        if pair_type in {"mol_pocket", "mol_pocket_route"}
    ]
    route_indices = [
        i for i, pair_type in enumerate(pair_types)
        if pair_type in {"mol_route", "mol_pocket_route"}
    ]
    protac_component_indices = [
        i for i, pair_type in enumerate(pair_types)
        if pair_type == "protac_component"
    ]
    mol_self_indices = [
        i for i, pair_type in enumerate(pair_types)
        if pair_type == "mol_self"
    ]
    activity_pair_indices = [
        i for i, pair_type in enumerate(pair_types)
        if pair_type == "activity_pair"
    ]
    route_template_indices = [
        i for i, pair_type in enumerate(pair_types)
        if pair_type == "route_template"
    ]
    protein_interface_indices = [
        i for i, pair_type in enumerate(pair_types)
        if pair_type == "protein_interface"
    ]
    interface_mutation_indices = [
        i for i, pair_type in enumerate(pair_types)
        if pair_type == "interface_mutation"
    ]
    pocket_route_indices = [
        i for i, pair_type in enumerate(pair_types)
        if pair_type == "mol_pocket_route"
    ]
    mol_route_indices = route_indices
    route_encode_indices = route_indices + route_template_indices
    mol_pocket_emb = _select_encoded_rows(mol_emb, mol_required_indices, pocket_indices)
    mol_route_emb = _select_encoded_rows(mol_emb, mol_required_indices, mol_route_indices)
    protac_anchor_emb = _select_encoded_rows(
        mol_emb,
        mol_required_indices,
        protac_component_indices,
    )
    protac_component_emb = None
    if protac_component_indices:
        component_smiles_batch = paired_batch.get("component_smiles", [])
        component_smiles = []
        for index in protac_component_indices:
            component_smiles_value = component_smiles_batch[index]
            if not component_smiles_value:
                raise ValueError("PROTAC component batch requires component_smiles")
            component_smiles.append(component_smiles_value)
        protac_component_emb = _normalize_batch_embeddings(
            _encode_model(mol_encoder, component_smiles)
        )
    protac_component_library_anchor_emb = _encode_smiles_groups(
        mol_encoder,
        paired_batch,
        protac_component_library_indices,
        ("component_smiles",),
        pair_type="protac_component_library",
    )
    protac_component_library_positive_emb = _encode_smiles_groups(
        mol_encoder,
        paired_batch,
        protac_component_library_indices,
        ("component_smiles",),
        pair_type="protac_component_library",
    )
    mol_self_anchor_emb = _select_encoded_rows(mol_emb, mol_required_indices, mol_self_indices)
    mol_self_positive_emb = _encode_positive_smiles(
        mol_encoder,
        paired_batch,
        mol_self_indices,
        "mol_self",
    )
    activity_anchor_emb = _select_encoded_rows(
        mol_emb,
        mol_required_indices,
        activity_pair_indices,
    )
    activity_positive_emb = _encode_positive_smiles(
        mol_encoder,
        paired_batch,
        activity_pair_indices,
        "activity_pair",
    )
    activity_delta = _activity_delta_tensor(
        paired_batch,
        activity_pair_indices,
        activity_anchor_emb,
    )
    pocket_emb, pocket_zero_loss = _encode_items_at_indices(
        pocket_encoder,
        paired_batch.get("pocket", []),
        pocket_indices,
        _DDP_DUMMY_POCKET,
    )
    route_emb, route_zero_loss = _encode_items_at_indices(
        route_encoder,
        paired_batch.get("route", []),
        route_encode_indices,
        _DDP_DUMMY_ROUTE,
    )
    pocket_route_pocket_emb = _select_encoded_rows(
        pocket_emb,
        pocket_indices,
        pocket_route_indices,
    )
    pocket_route_route_emb = _select_encoded_rows(
        route_emb,
        route_encode_indices,
        pocket_route_indices,
    )
    route_template_route_emb = _select_encoded_rows(
        route_emb,
        route_encode_indices,
        route_template_indices,
    )
    route_template_anchor_emb = _encode_route_template_views(
        route_encoder,
        paired_batch,
        route_template_indices,
    )
    protac_ternary_anchor_emb = _select_encoded_rows(
        mol_emb,
        mol_required_indices,
        protac_ternary_smiles_indices,
    )
    protac_ternary_smiles_emb = _encode_smiles_groups(
        mol_encoder,
        paired_batch,
        protac_ternary_smiles_indices,
        ("target_ligand_smiles", "e3_ligand_smiles"),
        pair_type="protac_ternary",
    )
    protac_ternary_pocket_emb = _encode_optional_item_groups(
        pocket_encoder,
        paired_batch,
        protac_ternary_smiles_indices,
        ("target_pocket", "e3_pocket"),
    )
    protac_ternary_positive_emb = _average_optional_embeddings(
        protac_ternary_smiles_emb,
        protac_ternary_pocket_emb,
    )
    protac_feature_anchor_emb = _encode_feature_vectors(
        encoders.get("protac_feature"),
        paired_batch,
        protac_ternary_feature_indices,
        "protac_feature",
        pair_type="protac_ternary",
    )
    protac_target_feature_emb = _encode_feature_vectors(
        encoders.get("protac_context_feature"),
        paired_batch,
        protac_ternary_feature_indices,
        "target_feature",
        pair_type="protac_ternary",
    )
    protac_e3_feature_emb = _encode_feature_vectors(
        encoders.get("protac_context_feature"),
        paired_batch,
        protac_ternary_feature_indices,
        "e3_feature",
        pair_type="protac_ternary",
    )
    protac_feature_positive_emb = _average_optional_embeddings(
        protac_target_feature_emb,
        protac_e3_feature_emb,
    )
    protac_ternary_anchor_emb = _concat_optional_embeddings(
        protac_ternary_anchor_emb,
        protac_feature_anchor_emb,
    )
    protac_ternary_positive_emb = _concat_optional_embeddings(
        protac_ternary_positive_emb,
        protac_feature_positive_emb,
    )
    protein_interface_anchor_emb, protein_interface_positive_emb = _encode_paired_items(
        pocket_encoder,
        paired_batch,
        protein_interface_indices,
        "interface_anchor",
        "interface_positive",
    )
    interface_mutation_anchor_emb, interface_mutation_positive_emb = _encode_paired_items(
        pocket_encoder,
        paired_batch,
        interface_mutation_indices,
        "interface_anchor",
        "interface_positive",
    )
    interface_affinity_delta = _interface_affinity_delta_tensor(
        paired_batch,
        interface_mutation_indices,
        interface_mutation_anchor_emb,
    )
    pdc_molecule_anchor_emb = _select_encoded_rows(
        mol_emb,
        mol_required_indices,
        pdc_component_molecule_indices,
    )
    pdc_molecule_component_emb = _encode_smiles_groups(
        mol_encoder,
        paired_batch,
        pdc_component_molecule_indices,
        ("component_smiles",),
        pair_type="pdc_component",
    )
    pdc_peptide_anchor_emb, _pdc_zero_loss = _encode_items_at_indices(
        pocket_encoder,
        paired_batch.get("peptide_pocket", []),
        pdc_component_peptide_indices,
    )
    pdc_peptide_component_emb = _encode_smiles_groups(
        mol_encoder,
        paired_batch,
        pdc_component_peptide_indices,
        ("component_smiles",),
        pair_type="pdc_component",
    )
    pdc_anchor_emb = _concat_optional_embeddings(
        pdc_molecule_anchor_emb,
        pdc_peptide_anchor_emb,
    )
    pdc_component_emb = _concat_optional_embeddings(
        pdc_molecule_component_emb,
        pdc_peptide_component_emb,
    )

    losses = _compute_losses(
        mol_pocket_emb,
        pocket_emb,
        route_emb,
        cfg.get("loss_weights", {}),
        cfg.get("contrastive", {}),
        cfg.get("curvature", 1.0),
        route_mol_emb=mol_route_emb,
        pocket_route_pocket_emb=pocket_route_pocket_emb,
        pocket_route_route_emb=pocket_route_route_emb,
        protac_anchor_emb=protac_anchor_emb,
        protac_component_emb=protac_component_emb,
        protac_component_library_anchor_emb=protac_component_library_anchor_emb,
        protac_component_library_positive_emb=protac_component_library_positive_emb,
        mol_self_anchor_emb=mol_self_anchor_emb,
        mol_self_positive_emb=mol_self_positive_emb,
        activity_anchor_emb=activity_anchor_emb,
        activity_positive_emb=activity_positive_emb,
        activity_delta=activity_delta,
        route_template_mol_emb=route_template_anchor_emb,
        route_template_route_emb=route_template_route_emb,
        protac_ternary_anchor_emb=protac_ternary_anchor_emb,
        protac_ternary_positive_emb=protac_ternary_positive_emb,
        protein_interface_anchor_emb=protein_interface_anchor_emb,
        protein_interface_positive_emb=protein_interface_positive_emb,
        interface_mutation_anchor_emb=interface_mutation_anchor_emb,
        interface_mutation_positive_emb=interface_mutation_positive_emb,
        interface_affinity_delta=interface_affinity_delta,
        pdc_anchor_emb=pdc_anchor_emb,
        pdc_component_emb=pdc_component_emb,
    )
    for key in (
        "pair_type_counts",
        "source_counts",
        "unique_source_coverage",
        "source_repeat_rate",
    ):
        if key in paired_batch:
            losses[key] = paired_batch[key]
    auxiliary_zero_loss = _sum_zero_losses(
        pocket_zero_loss,
        route_zero_loss,
        _pdc_zero_loss,
    )
    if auxiliary_zero_loss is not None:
        losses["total"] = losses["total"] + auxiliary_zero_loss
    if not return_context:
        return losses
    context = {
        "mol_emb": mol_emb,
        "ligand_smiles": [
            smiles_list[index]
            for index in mol_required_indices
            if index < len(smiles_list) and smiles_list[index]
        ],
        "route_emb": route_emb,
        "route_items": [
            paired_batch.get("route", [])[index]
            for index in route_encode_indices
            if paired_batch.get("route", [])[index] is not None
        ],
        "protac_component_emb": protac_component_emb,
        "protac_component_library_positive_emb": protac_component_library_positive_emb,
        "protac_ternary_positive_emb": protac_ternary_positive_emb,
        "protein_interface_positive_emb": protein_interface_positive_emb,
        "interface_mutation_positive_emb": interface_mutation_positive_emb,
        "pdc_component_emb": pdc_component_emb,
        "pair_type_counts": paired_batch.get("pair_type_counts", {}),
        "source_counts": paired_batch.get("source_counts", {}),
    }
    return losses, context


def _encode_molecule_indices(
    mol_encoder: nn.Module,
    smiles_list: list,
    indices: list[int],
) -> torch.Tensor | None:
    if not indices:
        return None
    smiles = []
    for index in indices:
        value = smiles_list[index] if index < len(smiles_list) else None
        if not value:
            raise ValueError("HUMU molecule objective requires ligand_smiles")
        smiles.append(value)
    return _normalize_batch_embeddings(_encode_model(mol_encoder, smiles))


def _encode_route_template_views(
    route_encoder: nn.Module,
    paired_batch: dict,
    indices: list[int],
) -> torch.Tensor | None:
    if not indices:
        return None
    routes = paired_batch.get("route", [])
    views = []
    for index in indices:
        route = routes[index] if index < len(routes) else None
        if route is None:
            raise ValueError("route_template batch requires route payload")
        template = route.get("template") or (route.get("reactions") or [""])[0]
        if not template:
            raise ValueError("route_template batch requires template")
        views.append(
            {
                "id": f"template:{route.get('id', index)}",
                "template": template,
                "reactions": [template],
                "steps": 1,
                "tree_depth": 1,
                "reaction_types": ["template"],
                "intermediates": [],
                "score": route.get("score", 0.0),
            }
        )
    return _normalize_batch_embeddings(_encode_model(route_encoder, views))


def _encode_positive_smiles(
    mol_encoder: nn.Module,
    paired_batch: dict,
    indices: list[int],
    pair_type: str,
) -> torch.Tensor | None:
    if not indices:
        return None
    values = paired_batch.get("positive_smiles", [])
    smiles = []
    for index in indices:
        value = values[index] if index < len(values) else None
        if not value:
            raise ValueError(f"{pair_type} batch requires positive_smiles")
        smiles.append(value)
    return _normalize_batch_embeddings(_encode_model(mol_encoder, smiles))


def _encode_smiles_groups(
    mol_encoder: nn.Module,
    paired_batch: dict,
    indices: list[int],
    keys: tuple[str, ...],
    *,
    pair_type: str,
) -> torch.Tensor | None:
    if not indices:
        return None
    groups = []
    flat_values = []
    for index in indices:
        values = []
        for key in keys:
            batch_values = paired_batch.get(key, [])
            value = batch_values[index] if index < len(batch_values) else None
            if value:
                values.append(value)
        if not values:
            raise ValueError(f"{pair_type} batch requires one of {', '.join(keys)}")
        groups.append((len(flat_values), len(values)))
        flat_values.extend(values)
    encoded = _normalize_batch_embeddings(_encode_model(mol_encoder, flat_values))
    rows = []
    for start, count in groups:
        rows.append(encoded[start : start + count].mean(dim=0))
    return torch.stack(rows, dim=0)


def _indices_with_batch_value(paired_batch: dict, indices: list[int], key: str) -> list[int]:
    values = paired_batch.get(key, [])
    return [
        index
        for index in indices
        if index < len(values) and values[index] is not None
    ]


def _encode_feature_vectors(
    feature_encoder: nn.Module | None,
    paired_batch: dict,
    indices: list[int],
    key: str,
    *,
    pair_type: str,
) -> torch.Tensor | None:
    if not indices:
        return None
    if feature_encoder is None:
        raise ValueError(f"{pair_type} batch requires {key} encoder")
    values = paired_batch.get(key, [])
    features = []
    for index in indices:
        value = values[index] if index < len(values) else None
        if value is None:
            raise ValueError(f"{pair_type} batch requires {key}")
        features.append(value)
    return _normalize_batch_embeddings(_encode_model(feature_encoder, features))


def _concat_optional_embeddings(*embeddings: torch.Tensor | None) -> torch.Tensor | None:
    tensors = [embedding for embedding in embeddings if embedding is not None]
    if not tensors:
        return None
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim=0)


def _encode_optional_item_groups(
    model: nn.Module,
    paired_batch: dict,
    indices: list[int],
    keys: tuple[str, ...],
) -> torch.Tensor | None:
    if not indices:
        return None
    groups = []
    flat_items = []
    for index in indices:
        items = []
        for key in keys:
            batch_values = paired_batch.get(key, [])
            value = batch_values[index] if index < len(batch_values) else None
            if value is not None:
                items.append(value)
        if not items:
            groups.append((None, 0))
            continue
        groups.append((len(flat_items), len(items)))
        flat_items.extend(items)
    if not flat_items:
        return None
    encoded = _normalize_batch_embeddings(_encode_model(model, flat_items))
    rows = []
    for start, count in groups:
        if start is None:
            rows.append(torch.zeros_like(encoded[0]))
        else:
            rows.append(encoded[start : start + count].mean(dim=0))
    return torch.stack(rows, dim=0)


def _average_optional_embeddings(*embeddings: torch.Tensor | None) -> torch.Tensor | None:
    tensors = [embedding for embedding in embeddings if embedding is not None]
    if not tensors:
        return None
    if len(tensors) == 1:
        return tensors[0]
    return torch.stack(tensors, dim=0).mean(dim=0)


def _encode_paired_items(
    model: nn.Module,
    paired_batch: dict,
    indices: list[int],
    anchor_key: str,
    positive_key: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not indices:
        return None, None
    anchor_emb, _ = _encode_items_at_indices(
        model,
        paired_batch.get(anchor_key, []),
        indices,
    )
    positive_emb, _ = _encode_items_at_indices(
        model,
        paired_batch.get(positive_key, []),
        indices,
    )
    return anchor_emb, positive_emb


def _activity_delta_tensor(
    paired_batch: dict,
    indices: list[int],
    anchor_emb: torch.Tensor | None,
) -> torch.Tensor | None:
    if not indices or anchor_emb is None:
        return None
    values = paired_batch.get("activity_delta", [])
    deltas = []
    for index in indices:
        value = values[index] if index < len(values) else None
        if value is None:
            raise ValueError("activity_pair batch requires activity_delta")
        deltas.append(float(value))
    return torch.tensor(deltas, dtype=torch.float32, device=anchor_emb.device)


def _interface_affinity_delta_tensor(
    paired_batch: dict,
    indices: list[int],
    anchor_emb: torch.Tensor | None,
) -> torch.Tensor | None:
    if not indices or anchor_emb is None:
        return None
    values = paired_batch.get("interface_affinity_delta", [])
    deltas = []
    for index in indices:
        value = values[index] if index < len(values) else None
        if value is None:
            raise ValueError("interface_mutation batch requires interface_affinity_delta")
        deltas.append(float(value))
    return torch.tensor(deltas, dtype=torch.float32, device=anchor_emb.device)


def _encode_items_at_indices(
    model: nn.Module,
    items: list,
    indices: list[int],
    padding_item: dict | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    selected = []
    for index in indices:
        item = items[index]
        if item is None:
            raise ValueError("Paired HUMU batch contains a missing tower payload")
        selected.append(item)
    if selected:
        return _normalize_batch_embeddings(_encode_model(model, selected)), None
    if padding_item is None:
        return None, None
    padding_emb = _normalize_batch_embeddings(_encode_model(model, [padding_item]))
    return None, padding_emb.sum() * 0.0


def _normalize_batch_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    if embeddings.ndim == 3:
        return embeddings.squeeze(1)
    return embeddings


def _select_encoded_rows(
    embeddings: torch.Tensor | None,
    source_indices: list[int],
    selected_indices: list[int],
) -> torch.Tensor | None:
    if embeddings is None or not selected_indices:
        return None
    row_by_index = {source_index: row for row, source_index in enumerate(source_indices)}
    rows = [row_by_index[index] for index in selected_indices if index in row_by_index]
    if not rows:
        return None
    return embeddings[rows]


def _sum_zero_losses(*losses: torch.Tensor | None) -> torch.Tensor | None:
    total = None
    for loss in losses:
        if loss is None:
            continue
        total = loss if total is None else total + loss
    return total


def _contrastive_loss(emb_a, emb_b, manifold) -> torch.Tensor:
    """Lorentz-aware contrastive loss between two embedding sets."""
    if emb_a is None or emb_b is None:
        device = emb_a.device if emb_a is not None else emb_b.device if emb_b is not None else "cpu"
        return torch.tensor(0.0, device=device)
    emb_a, emb_b = _align_batches(emb_a, emb_b)
    dist = manifold.distance(emb_a, emb_b)
    return dist.mean()


def _align_batches(emb_a: torch.Tensor, emb_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = min(emb_a.shape[0], emb_b.shape[0])
    if n == 0:
        raise ValueError("Cannot compute HUMU loss with an empty embedding batch")
    return emb_a[:n], emb_b[:n]


def _in_batch_contrastive_loss(
    anchor_emb: torch.Tensor,
    positive_emb: torch.Tensor,
    manifold,
    temperature: float,
) -> torch.Tensor:
    anchor_emb, positive_emb = _align_batches(anchor_emb, positive_emb)
    distances = _pairwise_lorentz_distance(anchor_emb, positive_emb, manifold)
    logits = -distances / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return functional.cross_entropy(logits, labels)


def _activity_supervised_loss(
    anchor_emb: torch.Tensor,
    positive_emb: torch.Tensor,
    activity_delta: torch.Tensor | None,
    manifold,
) -> torch.Tensor:
    anchor_emb, positive_emb = _align_batches(anchor_emb, positive_emb)
    distances = manifold.distance(anchor_emb, positive_emb).reshape(-1)
    if activity_delta is None:
        target = torch.zeros_like(distances)
    else:
        target = torch.log1p(activity_delta[: distances.shape[0]].to(distances.device))
    return functional.smooth_l1_loss(distances, target)


def _hard_negative_objective(
    pair_embeddings,
    manifold,
    margin: float,
) -> torch.Tensor:
    losses = []
    for anchor_emb, positive_emb in pair_embeddings:
        if anchor_emb is None or positive_emb is None:
            continue
        anchor_emb, positive_emb = _align_batches(anchor_emb, positive_emb)
        if anchor_emb.shape[0] < 2:
            continue
        distances = _pairwise_lorentz_distance(anchor_emb, positive_emb, manifold)
        positives = torch.diagonal(distances)
        mask = ~torch.eye(distances.shape[0], dtype=torch.bool, device=distances.device)
        hard_negatives = distances.masked_fill(~mask, float("inf")).min(dim=1).values
        losses.append(functional.relu(positives - hard_negatives + margin).mean())
    if losses:
        return torch.stack(losses).mean()
    for anchor_emb, positive_emb in pair_embeddings:
        if anchor_emb is not None:
            return anchor_emb.sum() * 0.0
        if positive_emb is not None:
            return positive_emb.sum() * 0.0
    return torch.tensor(0.0)


def _pairwise_lorentz_distance(
    anchor_emb: torch.Tensor,
    positive_emb: torch.Tensor,
    manifold,
) -> torch.Tensor:
    distances = manifold.distance(anchor_emb[:, None, :], positive_emb[None, :, :])
    if distances.ndim == 3:
        distances = distances.squeeze(-1)
    return distances


def _retrieval_top1_accuracy(
    anchor_emb: torch.Tensor,
    positive_emb: torch.Tensor,
    manifold,
) -> float:
    anchor_emb, positive_emb = _align_batches(anchor_emb, positive_emb)
    distances = _pairwise_lorentz_distance(anchor_emb, positive_emb, manifold)
    nearest = distances.argmin(dim=1)
    labels = torch.arange(distances.shape[0], device=distances.device)
    return float((nearest == labels).float().mean().cpu().item())


def _compute_losses(
    mol_emb,
    pocket_emb,
    route_emb,
    loss_weights: dict,
    contrastive_cfg: dict | None = None,
    curvature: float = 1.0,
    route_mol_emb=None,
    pocket_route_pocket_emb=None,
    pocket_route_route_emb=None,
    protac_anchor_emb=None,
    protac_component_emb=None,
    protac_component_library_anchor_emb=None,
    protac_component_library_positive_emb=None,
    mol_self_anchor_emb=None,
    mol_self_positive_emb=None,
    activity_anchor_emb=None,
    activity_positive_emb=None,
    activity_delta=None,
    route_template_mol_emb=None,
    route_template_route_emb=None,
    protac_ternary_anchor_emb=None,
    protac_ternary_positive_emb=None,
    protein_interface_anchor_emb=None,
    protein_interface_positive_emb=None,
    interface_mutation_anchor_emb=None,
    interface_mutation_positive_emb=None,
    interface_affinity_delta=None,
    pdc_anchor_emb=None,
    pdc_component_emb=None,
) -> dict:
    """Compute HUMU paired contrastive losses with in-batch negatives."""
    from mf_humu.manifold.lorentz import LorentzManifold
    manifold = LorentzManifold(curvature=curvature)
    contrastive_cfg = contrastive_cfg or {}
    negative_sampling = contrastive_cfg.get("negative_sampling", "in_batch")
    if negative_sampling not in {"in_batch", "hard_negative"}:
        raise ValueError(
            "contrastive.negative_sampling must be 'in_batch' or 'hard_negative' "
            "for HUMU pretraining"
        )
    temperature = float(contrastive_cfg.get("temperature", 0.07))
    if temperature <= 0:
        raise ValueError("contrastive.temperature must be > 0")
    hard_negative_margin = float(contrastive_cfg.get("hard_negative_margin", 0.2))

    w_mol_pocket = loss_weights.get("mol_pocket", 1.0)
    w_mol_route = loss_weights.get("mol_route", 0.5)
    w_pocket_route = loss_weights.get("pocket_route", 0.0)
    w_protac_component = loss_weights.get("protac_component", 0.0)
    w_protac_component_library = loss_weights.get("protac_component_library", 0.0)
    w_mol_self = loss_weights.get("mol_self", 0.0)
    w_activity = loss_weights.get("activity_supervised", 0.0)
    w_route_template = loss_weights.get("route_template", 0.0)
    w_protac_ternary = loss_weights.get("protac_ternary", 0.0)
    w_protein_interface = loss_weights.get("protein_interface", 0.0)
    w_interface_mutation = loss_weights.get("interface_mutation", 0.0)
    w_pdc_component = loss_weights.get("pdc_component", 0.0)
    w_hard_negative = loss_weights.get(
        "hard_negative",
        1.0 if negative_sampling == "hard_negative" else 0.0,
    )
    w_curvature = loss_weights.get("curvature_reg", 0.0)

    l_mol_pocket = _zero_loss(
        mol_emb,
        pocket_emb,
        route_emb,
        pocket_route_pocket_emb,
        pocket_route_route_emb,
        protac_anchor_emb,
        protac_component_emb,
        protac_component_library_anchor_emb,
        protac_component_library_positive_emb,
        mol_self_anchor_emb,
        mol_self_positive_emb,
        activity_anchor_emb,
        activity_positive_emb,
        route_template_mol_emb,
        route_template_route_emb,
        protac_ternary_anchor_emb,
        protac_ternary_positive_emb,
        protein_interface_anchor_emb,
        protein_interface_positive_emb,
        interface_mutation_anchor_emb,
        interface_mutation_positive_emb,
        pdc_anchor_emb,
        pdc_component_emb,
    )
    l_mol_route = _zero_loss(
        mol_emb,
        pocket_emb,
        route_emb,
        pocket_route_pocket_emb,
        pocket_route_route_emb,
        protac_anchor_emb,
        protac_component_emb,
        protac_component_library_anchor_emb,
        protac_component_library_positive_emb,
        mol_self_anchor_emb,
        mol_self_positive_emb,
        activity_anchor_emb,
        activity_positive_emb,
        route_template_mol_emb,
        route_template_route_emb,
        protac_ternary_anchor_emb,
        protac_ternary_positive_emb,
        protein_interface_anchor_emb,
        protein_interface_positive_emb,
        interface_mutation_anchor_emb,
        interface_mutation_positive_emb,
        pdc_anchor_emb,
        pdc_component_emb,
    )
    l_pocket_route = _zero_loss(
        mol_emb,
        pocket_emb,
        route_emb,
        pocket_route_pocket_emb,
        pocket_route_route_emb,
        protac_anchor_emb,
        protac_component_emb,
        protac_component_library_anchor_emb,
        protac_component_library_positive_emb,
        mol_self_anchor_emb,
        mol_self_positive_emb,
        activity_anchor_emb,
        activity_positive_emb,
        route_template_mol_emb,
        route_template_route_emb,
        protac_ternary_anchor_emb,
        protac_ternary_positive_emb,
        protein_interface_anchor_emb,
        protein_interface_positive_emb,
        interface_mutation_anchor_emb,
        interface_mutation_positive_emb,
        pdc_anchor_emb,
        pdc_component_emb,
    )
    l_protac_component = _zero_loss(
        mol_emb,
        pocket_emb,
        route_emb,
        pocket_route_pocket_emb,
        pocket_route_route_emb,
        protac_anchor_emb,
        protac_component_emb,
        protac_component_library_anchor_emb,
        protac_component_library_positive_emb,
        mol_self_anchor_emb,
        mol_self_positive_emb,
        activity_anchor_emb,
        activity_positive_emb,
        route_template_mol_emb,
        route_template_route_emb,
        protac_ternary_anchor_emb,
        protac_ternary_positive_emb,
        protein_interface_anchor_emb,
        protein_interface_positive_emb,
        interface_mutation_anchor_emb,
        interface_mutation_positive_emb,
        pdc_anchor_emb,
        pdc_component_emb,
    )
    l_protac_component_library = _zero_loss(
        mol_emb,
        pocket_emb,
        route_emb,
        protac_anchor_emb,
        protac_component_emb,
        protac_component_library_anchor_emb,
        protac_component_library_positive_emb,
        mol_self_anchor_emb,
        mol_self_positive_emb,
        activity_anchor_emb,
        activity_positive_emb,
        route_template_mol_emb,
        route_template_route_emb,
        protac_ternary_anchor_emb,
        protac_ternary_positive_emb,
        protein_interface_anchor_emb,
        protein_interface_positive_emb,
        interface_mutation_anchor_emb,
        interface_mutation_positive_emb,
        pdc_anchor_emb,
        pdc_component_emb,
    )
    l_mol_self = _zero_loss(
        mol_emb,
        pocket_emb,
        route_emb,
        protac_anchor_emb,
        protac_component_emb,
        protac_component_library_anchor_emb,
        protac_component_library_positive_emb,
        mol_self_anchor_emb,
        mol_self_positive_emb,
        activity_anchor_emb,
        activity_positive_emb,
        route_template_mol_emb,
        route_template_route_emb,
        protac_ternary_anchor_emb,
        protac_ternary_positive_emb,
        protein_interface_anchor_emb,
        protein_interface_positive_emb,
        interface_mutation_anchor_emb,
        interface_mutation_positive_emb,
        pdc_anchor_emb,
        pdc_component_emb,
    )
    l_activity_supervised = _zero_loss(
        mol_emb,
        pocket_emb,
        route_emb,
        protac_anchor_emb,
        protac_component_emb,
        mol_self_anchor_emb,
        mol_self_positive_emb,
        activity_anchor_emb,
        activity_positive_emb,
        route_template_mol_emb,
        route_template_route_emb,
        protac_ternary_anchor_emb,
        protac_ternary_positive_emb,
        protein_interface_anchor_emb,
        protein_interface_positive_emb,
        interface_mutation_anchor_emb,
        interface_mutation_positive_emb,
        pdc_anchor_emb,
        pdc_component_emb,
    )
    l_route_template = _zero_loss(
        mol_emb,
        pocket_emb,
        route_emb,
        protac_anchor_emb,
        protac_component_emb,
        mol_self_anchor_emb,
        mol_self_positive_emb,
        activity_anchor_emb,
        activity_positive_emb,
        route_template_mol_emb,
        route_template_route_emb,
        protac_ternary_anchor_emb,
        protac_ternary_positive_emb,
        protein_interface_anchor_emb,
        protein_interface_positive_emb,
        interface_mutation_anchor_emb,
        interface_mutation_positive_emb,
        pdc_anchor_emb,
        pdc_component_emb,
    )
    l_protac_ternary = _zero_loss(
        mol_emb,
        pocket_emb,
        route_emb,
        protac_ternary_anchor_emb,
        protac_ternary_positive_emb,
        protein_interface_anchor_emb,
        protein_interface_positive_emb,
        interface_mutation_anchor_emb,
        interface_mutation_positive_emb,
        pdc_anchor_emb,
        pdc_component_emb,
    )
    l_protein_interface = _zero_loss(
        mol_emb,
        pocket_emb,
        route_emb,
        protac_ternary_anchor_emb,
        protac_ternary_positive_emb,
        protein_interface_anchor_emb,
        protein_interface_positive_emb,
        interface_mutation_anchor_emb,
        interface_mutation_positive_emb,
        pdc_anchor_emb,
        pdc_component_emb,
    )
    l_interface_mutation = _zero_loss(
        mol_emb,
        pocket_emb,
        route_emb,
        protac_ternary_anchor_emb,
        protac_ternary_positive_emb,
        protein_interface_anchor_emb,
        protein_interface_positive_emb,
        interface_mutation_anchor_emb,
        interface_mutation_positive_emb,
        pdc_anchor_emb,
        pdc_component_emb,
    )
    l_pdc_component = _zero_loss(
        mol_emb,
        pocket_emb,
        route_emb,
        protac_ternary_anchor_emb,
        protac_ternary_positive_emb,
        protein_interface_anchor_emb,
        protein_interface_positive_emb,
        interface_mutation_anchor_emb,
        interface_mutation_positive_emb,
        pdc_anchor_emb,
        pdc_component_emb,
    )
    l_hard_negative = _zero_loss(
        mol_emb,
        pocket_emb,
        route_emb,
        protac_anchor_emb,
        protac_component_emb,
        mol_self_anchor_emb,
        mol_self_positive_emb,
        activity_anchor_emb,
        activity_positive_emb,
        route_template_mol_emb,
        route_template_route_emb,
        protac_ternary_anchor_emb,
        protac_ternary_positive_emb,
        protein_interface_anchor_emb,
        protein_interface_positive_emb,
        interface_mutation_anchor_emb,
        interface_mutation_positive_emb,
        pdc_anchor_emb,
        pdc_component_emb,
    )
    if mol_emb is not None and pocket_emb is not None:
        l_mol_pocket = _in_batch_contrastive_loss(mol_emb, pocket_emb, manifold, temperature)
    route_anchor = route_mol_emb if route_mol_emb is not None else mol_emb
    if route_anchor is not None and route_emb is not None:
        l_mol_route = _in_batch_contrastive_loss(route_anchor, route_emb, manifold, temperature)
    if pocket_route_pocket_emb is not None and pocket_route_route_emb is not None:
        l_pocket_route = _in_batch_contrastive_loss(
            pocket_route_pocket_emb,
            pocket_route_route_emb,
            manifold,
            temperature,
        )
    if protac_anchor_emb is not None and protac_component_emb is not None:
        l_protac_component = _in_batch_contrastive_loss(
            protac_anchor_emb,
            protac_component_emb,
            manifold,
            temperature,
        )
    if (
        protac_component_library_anchor_emb is not None
        and protac_component_library_positive_emb is not None
    ):
        l_protac_component_library = _in_batch_contrastive_loss(
            protac_component_library_anchor_emb,
            protac_component_library_positive_emb,
            manifold,
            temperature,
        )
    if mol_self_anchor_emb is not None and mol_self_positive_emb is not None:
        l_mol_self = _in_batch_contrastive_loss(
            mol_self_anchor_emb,
            mol_self_positive_emb,
            manifold,
            temperature,
        )
    if activity_anchor_emb is not None and activity_positive_emb is not None:
        l_activity_supervised = _activity_supervised_loss(
            activity_anchor_emb,
            activity_positive_emb,
            activity_delta,
            manifold,
        )
    if route_template_mol_emb is not None and route_template_route_emb is not None:
        l_route_template = _in_batch_contrastive_loss(
            route_template_mol_emb,
            route_template_route_emb,
            manifold,
            temperature,
        )
    if protac_ternary_anchor_emb is not None and protac_ternary_positive_emb is not None:
        l_protac_ternary = _in_batch_contrastive_loss(
            protac_ternary_anchor_emb,
            protac_ternary_positive_emb,
            manifold,
            temperature,
        )
    if protein_interface_anchor_emb is not None and protein_interface_positive_emb is not None:
        l_protein_interface = _in_batch_contrastive_loss(
            protein_interface_anchor_emb,
            protein_interface_positive_emb,
            manifold,
            temperature,
        )
    if interface_mutation_anchor_emb is not None and interface_mutation_positive_emb is not None:
        l_interface_mutation = _activity_supervised_loss(
            interface_mutation_anchor_emb,
            interface_mutation_positive_emb,
            interface_affinity_delta,
            manifold,
        )
    if pdc_anchor_emb is not None and pdc_component_emb is not None:
        l_pdc_component = _in_batch_contrastive_loss(
            pdc_anchor_emb,
            pdc_component_emb,
            manifold,
            temperature,
        )
    if negative_sampling == "hard_negative":
        l_hard_negative = _hard_negative_objective(
            (
                (mol_emb, pocket_emb),
                (route_anchor, route_emb),
                (pocket_route_pocket_emb, pocket_route_route_emb),
                (protac_anchor_emb, protac_component_emb),
                (
                    protac_component_library_anchor_emb,
                    protac_component_library_positive_emb,
                ),
                (mol_self_anchor_emb, mol_self_positive_emb),
                (route_template_mol_emb, route_template_route_emb),
                (protac_ternary_anchor_emb, protac_ternary_positive_emb),
                (protein_interface_anchor_emb, protein_interface_positive_emb),
                (interface_mutation_anchor_emb, interface_mutation_positive_emb),
                (pdc_anchor_emb, pdc_component_emb),
            ),
            manifold,
            hard_negative_margin,
        )

    l_mol_pocket = l_mol_pocket * w_mol_pocket
    l_mol_route = l_mol_route * w_mol_route
    l_pocket_route = l_pocket_route * w_pocket_route
    l_protac_component = l_protac_component * w_protac_component
    l_protac_component_library = (
        l_protac_component_library * w_protac_component_library
    )
    l_mol_self = l_mol_self * w_mol_self
    l_activity_supervised = l_activity_supervised * w_activity
    l_route_template = l_route_template * w_route_template
    l_protac_ternary = l_protac_ternary * w_protac_ternary
    l_protein_interface = l_protein_interface * w_protein_interface
    l_interface_mutation = l_interface_mutation * w_interface_mutation
    l_pdc_component = l_pdc_component * w_pdc_component
    l_hard_negative = l_hard_negative * w_hard_negative
    l_curvature_reg = _curvature_regularization(
        mol_emb,
        pocket_emb,
        route_emb,
        pocket_route_pocket_emb,
        pocket_route_route_emb,
        protac_anchor_emb,
        protac_component_emb,
        protac_component_library_anchor_emb,
        protac_component_library_positive_emb,
        mol_self_anchor_emb,
        mol_self_positive_emb,
        activity_anchor_emb,
        activity_positive_emb,
        route_template_mol_emb,
        route_template_route_emb,
        protac_ternary_anchor_emb,
        protac_ternary_positive_emb,
        protein_interface_anchor_emb,
        protein_interface_positive_emb,
        interface_mutation_anchor_emb,
        interface_mutation_positive_emb,
        pdc_anchor_emb,
        pdc_component_emb,
        manifold=manifold,
    ) * w_curvature
    total = (
        l_mol_pocket
        + l_mol_route
        + l_pocket_route
        + l_protac_component
        + l_protac_component_library
        + l_mol_self
        + l_activity_supervised
        + l_route_template
        + l_protac_ternary
        + l_protein_interface
        + l_interface_mutation
        + l_pdc_component
        + l_hard_negative
        + l_curvature_reg
    )
    stats = _contrastive_stats(
        mol_emb,
        pocket_emb,
        route_anchor,
        route_emb,
        manifold,
        pocket_route_pocket_emb=pocket_route_pocket_emb,
        pocket_route_route_emb=pocket_route_route_emb,
        protac_anchor_emb=protac_anchor_emb,
        protac_component_emb=protac_component_emb,
        protac_component_library_anchor_emb=protac_component_library_anchor_emb,
        protac_component_library_positive_emb=protac_component_library_positive_emb,
        mol_self_anchor_emb=mol_self_anchor_emb,
        mol_self_positive_emb=mol_self_positive_emb,
        activity_anchor_emb=activity_anchor_emb,
        activity_positive_emb=activity_positive_emb,
        route_template_mol_emb=route_template_mol_emb,
        route_template_route_emb=route_template_route_emb,
        protac_ternary_anchor_emb=protac_ternary_anchor_emb,
        protac_ternary_positive_emb=protac_ternary_positive_emb,
        protein_interface_anchor_emb=protein_interface_anchor_emb,
        protein_interface_positive_emb=protein_interface_positive_emb,
        interface_mutation_anchor_emb=interface_mutation_anchor_emb,
        interface_mutation_positive_emb=interface_mutation_positive_emb,
        pdc_anchor_emb=pdc_anchor_emb,
        pdc_component_emb=pdc_component_emb,
    )
    return {
        "total": total,
        "l_mol_pocket": l_mol_pocket,
        "l_mol_route": l_mol_route,
        "l_pocket_route": l_pocket_route,
        "l_protac_component": l_protac_component,
        "l_protac_component_library": l_protac_component_library,
        "l_mol_self": l_mol_self,
        "l_activity_supervised": l_activity_supervised,
        "l_route_template": l_route_template,
        "l_protac_ternary": l_protac_ternary,
        "l_protein_interface": l_protein_interface,
        "l_interface_mutation": l_interface_mutation,
        "l_pdc_component": l_pdc_component,
        "l_hard_negative": l_hard_negative,
        "l_curvature_reg": l_curvature_reg,
        **stats,
    }


def _zero_loss(*embeddings) -> torch.Tensor:
    for emb in embeddings:
        if emb is not None:
            return emb.sum() * 0.0
    return torch.tensor(0.0)


def _curvature_regularization(*embeddings, manifold) -> torch.Tensor:
    tensors = [emb for emb in embeddings if emb is not None]
    if not tensors:
        return torch.tensor(0.0)
    stacked = torch.cat(tensors, dim=0)
    lorentz_norm = manifold.inner(stacked, stacked, keepdim=False)
    target = torch.full_like(lorentz_norm, -1.0 / manifold.k)
    return (lorentz_norm - target).abs().mean()


def _contrastive_stats(
    mol_emb,
    pocket_emb,
    route_mol_emb,
    route_emb,
    manifold,
    pocket_route_pocket_emb=None,
    pocket_route_route_emb=None,
    protac_anchor_emb=None,
    protac_component_emb=None,
    protac_component_library_anchor_emb=None,
    protac_component_library_positive_emb=None,
    mol_self_anchor_emb=None,
    mol_self_positive_emb=None,
    activity_anchor_emb=None,
    activity_positive_emb=None,
    route_template_mol_emb=None,
    route_template_route_emb=None,
    protac_ternary_anchor_emb=None,
    protac_ternary_positive_emb=None,
    protein_interface_anchor_emb=None,
    protein_interface_positive_emb=None,
    interface_mutation_anchor_emb=None,
    interface_mutation_positive_emb=None,
    pdc_anchor_emb=None,
    pdc_component_emb=None,
) -> dict:
    positive = []
    negative = []
    all_embeddings = []
    for emb in (
        mol_emb,
        pocket_emb,
        route_mol_emb,
        route_emb,
        pocket_route_pocket_emb,
        pocket_route_route_emb,
        protac_anchor_emb,
        protac_component_emb,
        protac_component_library_anchor_emb,
        protac_component_library_positive_emb,
        mol_self_anchor_emb,
        mol_self_positive_emb,
        activity_anchor_emb,
        activity_positive_emb,
        route_template_mol_emb,
        route_template_route_emb,
        protac_ternary_anchor_emb,
        protac_ternary_positive_emb,
        protein_interface_anchor_emb,
        protein_interface_positive_emb,
        interface_mutation_anchor_emb,
        interface_mutation_positive_emb,
        pdc_anchor_emb,
        pdc_component_emb,
    ):
        if emb is not None:
            all_embeddings.append(emb.detach())
    for anchor, positive_emb in (
        (mol_emb, pocket_emb),
        (route_mol_emb, route_emb),
        (pocket_route_pocket_emb, pocket_route_route_emb),
        (protac_anchor_emb, protac_component_emb),
        (
            protac_component_library_anchor_emb,
            protac_component_library_positive_emb,
        ),
        (mol_self_anchor_emb, mol_self_positive_emb),
        (activity_anchor_emb, activity_positive_emb),
        (route_template_mol_emb, route_template_route_emb),
        (protac_ternary_anchor_emb, protac_ternary_positive_emb),
        (protein_interface_anchor_emb, protein_interface_positive_emb),
        (interface_mutation_anchor_emb, interface_mutation_positive_emb),
        (pdc_anchor_emb, pdc_component_emb),
    ):
        if anchor is None or positive_emb is None:
            continue
        anchor, positive_emb = _align_batches(anchor.detach(), positive_emb.detach())
        distances = _pairwise_lorentz_distance(anchor, positive_emb, manifold)
        positive.append(torch.diagonal(distances))
        if distances.shape[0] > 1:
            mask = ~torch.eye(distances.shape[0], dtype=torch.bool, device=distances.device)
            negative.append(distances[mask])
    if positive:
        positive_distance = float(torch.cat(positive).mean().cpu().item())
    else:
        positive_distance = 0.0
    if negative:
        negative_distance = float(torch.cat(negative).mean().cpu().item())
    else:
        negative_distance = 0.0
    retrieval_scores = []
    retrieval_pairs = {
        "mol_pocket_retrieval_top1": (mol_emb, pocket_emb),
        "mol_route_retrieval_top1": (route_mol_emb, route_emb),
        "pocket_route_retrieval_top1": (
            pocket_route_pocket_emb,
            pocket_route_route_emb,
        ),
        "protac_component_retrieval_top1": (
            protac_anchor_emb,
            protac_component_emb,
        ),
        "protac_component_library_retrieval_top1": (
            protac_component_library_anchor_emb,
            protac_component_library_positive_emb,
        ),
        "mol_self_retrieval_top1": (
            mol_self_anchor_emb,
            mol_self_positive_emb,
        ),
        "route_template_retrieval_top1": (
            route_template_mol_emb,
            route_template_route_emb,
        ),
        "protac_ternary_retrieval_top1": (
            protac_ternary_anchor_emb,
            protac_ternary_positive_emb,
        ),
        "protein_interface_retrieval_top1": (
            protein_interface_anchor_emb,
            protein_interface_positive_emb,
        ),
        "interface_mutation_retrieval_top1": (
            interface_mutation_anchor_emb,
            interface_mutation_positive_emb,
        ),
        "pdc_component_retrieval_top1": (
            pdc_anchor_emb,
            pdc_component_emb,
        ),
    }
    retrieval_metrics = {}
    for name, (anchor, positive_emb) in retrieval_pairs.items():
        if anchor is not None and positive_emb is not None:
            score = _retrieval_top1_accuracy(
                anchor.detach(),
                positive_emb.detach(),
                manifold,
            )
            retrieval_metrics[name] = score
            if name not in _DEGENERATE_RETRIEVAL_KEYS:
                retrieval_scores.append(score)
        else:
            retrieval_metrics[name] = 0.0
    retrieval_top1 = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0.0
    activity_pair_margin = _pair_distance_margin(
        activity_anchor_emb,
        activity_positive_emb,
        manifold,
    )
    interface_mutation_margin = _pair_distance_margin(
        interface_mutation_anchor_emb,
        interface_mutation_positive_emb,
        manifold,
    )
    if all_embeddings:
        embeddings = torch.cat(all_embeddings, dim=0)
        embedding_variance = float(embeddings[:, 1:].var(dim=0, unbiased=False).mean().cpu().item())
        lorentz_norm = -embeddings[:, 0].pow(2) + embeddings[:, 1:].pow(2).sum(dim=-1)
        lorentz_norm_deviation = float((lorentz_norm + 1.0).abs().mean().cpu().item())
    else:
        embedding_variance = 0.0
        lorentz_norm_deviation = 0.0
    collapse_ratio = 1.0 if embedding_variance < 1e-8 else 0.0
    return {
        "positive_distance": positive_distance,
        "negative_distance": negative_distance,
        "retrieval_top1": retrieval_top1,
        "embedding_variance": embedding_variance,
        "lorentz_norm_deviation": lorentz_norm_deviation,
        "collapse_ratio": collapse_ratio,
        "activity_pair_margin": activity_pair_margin,
        "interface_mutation_margin": interface_mutation_margin,
        **retrieval_metrics,
    }


def _pair_distance_margin(anchor_emb, positive_emb, manifold) -> float:
    if anchor_emb is None or positive_emb is None:
        return 0.0
    anchor_emb, positive_emb = _align_batches(anchor_emb.detach(), positive_emb.detach())
    distances = _pairwise_lorentz_distance(anchor_emb, positive_emb, manifold)
    if distances.shape[0] < 2:
        return 0.0
    positives = torch.diagonal(distances)
    mask = ~torch.eye(distances.shape[0], dtype=torch.bool, device=distances.device)
    negatives = distances[mask]
    return float((negatives.mean() - positives.mean()).cpu().item())


def _save_checkpoint(
    encoders: dict,
    optimizer,
    scheduler,
    epoch: int,
    loss: float,
    path: Path,
    *,
    checkpoint_type: str | None = None,
    global_step: int | None = None,
    epoch_step: int | None = None,
    n_batches: int | None = None,
    best_loss: float | None = None,
    epoch_loss_sum: float | None = None,
    epoch_loss_count: int | None = None,
) -> None:
    """Save a training checkpoint."""
    state = {
        "epoch": epoch,
        "loss": loss,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    optional_state = {
        "checkpoint_type": checkpoint_type,
        "global_step": global_step,
        "epoch_step": epoch_step,
        "n_batches": n_batches,
        "best_loss": best_loss,
        "epoch_loss_sum": epoch_loss_sum,
        "epoch_loss_count": epoch_loss_count,
    }
    state.update({
        key: value
        for key, value in optional_state.items()
        if value is not None
    })
    for name, model in encoders.items():
        state[f"encoder_{name}"] = _checkpoint_state_dict(_module(model))
    torch.save(state, str(path))


def _checkpoint_state_dict(model: nn.Module) -> dict:
    return {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith("inner._esm2_model.")
    }


def _load_checkpoint(
    encoders: dict,
    path: str | Path,
    device: torch.device,
    optimizer=None,
    scheduler=None,
) -> tuple[int, float]:
    """Load a training checkpoint. Returns (next_start_epoch, best_loss)."""
    state = torch.load(path, map_location=device, weights_only=True)
    _restore_checkpoint_state(encoders, state, optimizer=optimizer, scheduler=scheduler)
    return int(state.get("epoch", -1)) + 1, state.get("loss", float("inf"))


def _load_training_checkpoint(
    encoders: dict,
    path: str | Path,
    device: torch.device,
    optimizer=None,
    scheduler=None,
) -> TrainingResumeState:
    state = torch.load(path, map_location=device, weights_only=True)
    _restore_checkpoint_state(encoders, state, optimizer=optimizer, scheduler=scheduler)
    checkpoint_type = state.get("checkpoint_type")
    is_legacy_step_checkpoint = (
        checkpoint_type is None
        and Path(path).name.startswith("checkpoint_step_")
    )
    raw_epoch = int(state.get("epoch", -1))
    if checkpoint_type == "step":
        return TrainingResumeState(
            start_epoch=max(raw_epoch, 0),
            epoch_step=int(state.get("epoch_step", 0) or 0),
            best_loss=float(state.get("best_loss", state.get("loss", float("inf")))),
            epoch_loss_sum=float(state.get("epoch_loss_sum", 0.0) or 0.0),
            epoch_loss_count=int(state.get("epoch_loss_count", 0) or 0),
        )
    if is_legacy_step_checkpoint:
        return TrainingResumeState(
            start_epoch=max(raw_epoch, 0),
            epoch_step=0,
            best_loss=float(state.get("best_loss", state.get("loss", float("inf")))),
        )
    return TrainingResumeState(
        start_epoch=raw_epoch + 1,
        epoch_step=0,
        best_loss=float(state.get("best_loss", state.get("loss", float("inf")))),
    )


def _restore_checkpoint_state(
    encoders: dict,
    state: dict,
    optimizer=None,
    scheduler=None,
) -> None:
    for name, model in encoders.items():
        key = f"encoder_{name}"
        if key in state:
            model_state = _migrate_legacy_encoder_state_dict(name, _module(model), state[key])
            _module(model).load_state_dict(model_state, strict=False)
    if optimizer is not None and "optimizer" in state:
        try:
            optimizer.load_state_dict(state["optimizer"])
        except ValueError as exc:
            raise ValueError(
                "checkpoint optimizer state is incompatible with current HUMU encoder "
                "parameters; resume model weights without optimizer or restart optimizer"
            ) from exc
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])


def _migrate_legacy_encoder_state_dict(
    name: str,
    model: nn.Module,
    state_dict: dict,
) -> dict:
    migrated = dict(state_dict)
    current = model.state_dict()

    def copy_if_shape_matches(old_key: str, new_key: str) -> None:
        if old_key not in migrated or new_key in migrated or new_key not in current:
            return
        value = migrated[old_key]
        if tuple(value.shape) == tuple(current[new_key].shape):
            migrated[new_key] = value

    if name == "mol":
        copy_if_shape_matches("inner._atom_projection.weight", "inner._atom_projection.3.weight")
        copy_if_shape_matches("inner._atom_projection.bias", "inner._atom_projection.3.bias")
    elif name == "pocket":
        copy_if_shape_matches("inner._point_projection.weight", "inner._point_projection.3.weight")
        copy_if_shape_matches("inner._point_projection.bias", "inner._point_projection.3.bias")
    elif name == "route":
        copy_if_shape_matches("inner._route_projection.0.weight", "inner._feature_projection.weight")
        copy_if_shape_matches("inner._route_projection.0.bias", "inner._feature_projection.bias")
        copy_if_shape_matches("inner._route_projection.2.weight", "inner._output_projection.weight")
        copy_if_shape_matches("inner._route_projection.2.bias", "inner._output_projection.bias")
    return migrated


def _rotate_checkpoints(output_dir: Path, keep_last_n: int | None) -> None:
    """Remove old checkpoints, keeping only the most recent N."""
    if keep_last_n is None:
        return
    checkpoints = sorted(output_dir.glob("checkpoint_epoch_*.pt"))
    if len(checkpoints) > keep_last_n:
        for cp in checkpoints[:-keep_last_n]:
            cp.unlink()


def _should_validate_epoch(epoch: int, cfg: dict, loaders: dict) -> bool:
    eval_cfg = cfg.get("eval", {}) or {}
    every_n_epochs = int(eval_cfg.get("every_n_epochs", 0) or 0)
    return every_n_epochs > 0 and (epoch + 1) % every_n_epochs == 0 and "validation" in loaders


def _validate_epoch(
    encoders: dict[str, nn.Module],
    validation_loader,
    cfg: dict,
    device: torch.device,
    distributed: DistributedContext,
) -> dict:
    was_training = {name: model.training for name, model in encoders.items()}
    for model in encoders.values():
        model.eval()

    metric_keys = (
        "total",
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
        "l_hard_negative",
        "l_curvature_reg",
        "positive_distance",
        "negative_distance",
        "retrieval_top1",
        "mol_pocket_retrieval_top1",
        "mol_route_retrieval_top1",
        "pocket_route_retrieval_top1",
        "protac_component_retrieval_top1",
        "protac_component_library_retrieval_top1",
        "mol_self_retrieval_top1",
        "route_template_retrieval_top1",
        "protac_ternary_retrieval_top1",
        "protein_interface_retrieval_top1",
        "interface_mutation_retrieval_top1",
        "pdc_component_retrieval_top1",
        "activity_pair_margin",
        "interface_mutation_margin",
        "embedding_variance",
        "lorentz_norm_deviation",
        "collapse_ratio",
    )
    sums = {key: 0.0 for key in metric_keys}
    batches = 0
    samples = 0
    tree_distortion_sum = 0.0
    tree_distortion_count = 0
    activity_items = []
    pair_type_counts = Counter()
    source_counts = Counter()
    unique_source_coverage_sum = 0.0
    source_repeat_rate_sum = 0.0
    eval_cfg = cfg.get("eval", {}) or {}
    requested_metrics = set(eval_cfg.get("metrics", []) or [])
    activity_sources = _activity_sources_from_config(cfg)
    activity_records = (
        _load_activity_records(activity_sources)
        if "cliff_separation_auroc" in requested_metrics and activity_sources
        else {}
    )
    start = time.time()
    use_amp = bool(cfg.get("use_amp", False)) and device.type == "cuda"
    amp_dtype = _amp_dtype_from_config(cfg)

    try:
        with torch.no_grad():
            for paired_batch in validation_loader:
                with (
                    torch.amp.autocast("cuda", dtype=amp_dtype)
                    if use_amp
                    else _null_context()
                ):
                    losses, context = _forward_paired_batch(
                        encoders,
                        paired_batch,
                        cfg,
                        return_context=True,
                    )
                batches += 1
                samples += len(paired_batch.get("ligand_smiles", []))
                for key in metric_keys:
                    if key in losses:
                        sums[key] += _loss_float(losses[key])
                pair_type_counts.update(losses.get("pair_type_counts", {}))
                source_counts.update(losses.get("source_counts", {}))
                unique_source_coverage_sum += float(losses.get("unique_source_coverage", 0.0) or 0.0)
                source_repeat_rate_sum += float(losses.get("source_repeat_rate", 0.0) or 0.0)
                tree_distortion = _route_tree_distortion(
                    context.get("route_emb"),
                    context.get("route_items", []),
                )
                if tree_distortion is not None:
                    tree_distortion_sum += tree_distortion
                    tree_distortion_count += 1
                if "cliff_separation_auroc" in requested_metrics:
                    activity_items.extend(
                        _activity_items_from_batch(
                            context.get("mol_emb"),
                            context.get("ligand_smiles", []),
                        )
                    )
    finally:
        for name, model in encoders.items():
            model.train(was_training[name])

    reduced = _reduce_validation_sums(
        device,
        distributed,
        batches,
        samples,
        sums,
        tree_distortion_sum,
        tree_distortion_count,
        metric_keys,
    )
    global_batches = max(1, int(reduced["batches"]))
    metrics = {
        "val_batches": int(reduced["batches"]),
        "val_samples": int(reduced["samples"]),
        "validation_time_sec": time.time() - start,
    }
    for key in metric_keys:
        output_key = "val_loss" if key == "total" else key
        metrics[output_key] = reduced[key] / global_batches
    metrics["distance_margin"] = metrics["negative_distance"] - metrics["positive_distance"]
    metrics["pair_type_counts"] = dict(pair_type_counts)
    metrics["source_counts"] = dict(source_counts)
    metrics["unique_source_coverage"] = unique_source_coverage_sum / global_batches
    metrics["source_repeat_rate"] = source_repeat_rate_sum / global_batches
    if reduced["tree_distortion_count"] > 0:
        metrics["tree_distortion"] = (
            reduced["tree_distortion_sum"] / reduced["tree_distortion_count"]
        )
    if "cliff_separation_auroc" in requested_metrics:
        activity_items = _gather_activity_items(activity_items, distributed)
        score, status = _activity_cliff_auroc(
            activity_items,
            activity_records,
            eval_cfg,
            activity_source_configured=bool(activity_sources),
        )
        metrics["cliff_separation_auroc"] = score
        if status is not None:
            metrics["cliff_separation_auroc_status"] = status
    _apply_requested_validation_metrics(metrics, cfg)
    return metrics


def _reduce_validation_sums(
    device: torch.device,
    distributed: DistributedContext,
    batches: int,
    samples: int,
    sums: dict,
    tree_distortion_sum: float,
    tree_distortion_count: int,
    metric_keys: tuple[str, ...],
) -> dict:
    values = [
        float(batches),
        float(samples),
        tree_distortion_sum,
        float(tree_distortion_count),
        *(sums[key] for key in metric_keys),
    ]
    tensor_device = device if device.type == "cuda" else torch.device("cpu")
    tensor = torch.tensor(values, dtype=torch.float64, device=tensor_device)
    if distributed.enabled and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    result = {
        "batches": float(tensor[0].item()),
        "samples": float(tensor[1].item()),
        "tree_distortion_sum": float(tensor[2].item()),
        "tree_distortion_count": float(tensor[3].item()),
    }
    for offset, key in enumerate(metric_keys, start=4):
        result[key] = float(tensor[offset].item())
    return result


def _route_tree_distortion(route_emb: torch.Tensor | None, route_items: list[dict]) -> float | None:
    if route_emb is None or len(route_items) < 2:
        return None
    depths = torch.tensor(
        [
            float(route.get("tree_depth", route.get("steps", route.get("n_steps", 1))))
            for route in route_items
        ],
        dtype=torch.float32,
        device=route_emb.device,
    )
    tree_distances = (depths[:, None] - depths[None, :]).abs()
    mask = torch.triu(
        torch.ones_like(tree_distances, dtype=torch.bool),
        diagonal=1,
    )
    tree_values = tree_distances[mask]
    if tree_values.numel() == 0 or float(tree_values.max().item()) == 0.0:
        return None
    from mf_humu.manifold.lorentz import LorentzManifold
    manifold = LorentzManifold(curvature=1.0)
    embedding_values = _pairwise_lorentz_distance(route_emb, route_emb, manifold)[mask]
    tree_norm = tree_values / tree_values.mean().clamp_min(1e-8)
    embedding_norm = embedding_values / embedding_values.mean().clamp_min(1e-8)
    return float((embedding_norm - tree_norm).abs().mean().cpu().item())


def _activity_items_from_batch(
    mol_emb: torch.Tensor | None,
    ligand_smiles: list[str],
) -> list[dict]:
    if mol_emb is None or not ligand_smiles:
        return []
    n = min(mol_emb.shape[0], len(ligand_smiles))
    embeddings = mol_emb[:n].detach().cpu()
    return [
        {
            "ligand_smiles": str(ligand_smiles[index]),
            "embedding": embeddings[index].tolist(),
        }
        for index in range(n)
    ]


def _gather_activity_items(items: list[dict], distributed: DistributedContext) -> list[dict]:
    if not distributed.enabled or not dist.is_initialized():
        return items
    gathered: list[list[dict] | None] = [None for _ in range(distributed.world_size)]
    dist.all_gather_object(gathered, items)
    merged: list[dict] = []
    for rank_items in gathered:
        if rank_items:
            merged.extend(rank_items)
    return merged


def _load_activity_records(
    activity_source: str | os.PathLike | list[str] | tuple[str, ...] | None,
) -> dict[str, list[ActivityRecord]]:
    if not activity_source:
        return {}
    source_paths = (
        [activity_source]
        if isinstance(activity_source, str | os.PathLike)
        else list(activity_source)
    )
    records: dict[str, list[ActivityRecord]] = {}
    canonical_cache: dict[str, str] = {}
    for source in source_paths:
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"data.activity_source does not exist: {source_path}")
        if source_path.is_dir():
            files = sorted(source_path.glob("*.jsonl"))
        else:
            files = [source_path]
        for path in files:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = _activity_record_from_json(json.loads(line))
                    if record is None:
                        continue
                    key = canonical_cache.get(record.ligand_smiles)
                    if key is None:
                        key = _canonical_activity_smiles(record.ligand_smiles)
                        canonical_cache[record.ligand_smiles] = key
                    records.setdefault(key, []).append(record)
    return records


def _activity_sources_from_config(cfg: dict) -> list[str]:
    data_cfg = cfg.get("data", {}) or {}
    sources: list[str] = []

    def add(value) -> None:
        if not value:
            return
        if isinstance(value, dict):
            value = value.get("path")
        if not value:
            return
        text = str(value)
        if text not in sources:
            sources.append(text)

    add(data_cfg.get("activity_source"))
    configured = data_cfg.get("activity_sources") or []
    if isinstance(configured, str | os.PathLike) or isinstance(configured, dict):
        add(configured)
    else:
        for value in configured:
            add(value)
    return sources


def _activity_record_from_json(record: dict) -> ActivityRecord | None:
    smiles = (
        record.get("ligand_smiles")
        or record.get("canonical_smiles")
        or record.get("smiles")
    )
    target_id = record.get("target_id") or record.get("target_chembl_id")
    activity_value = (
        record.get("activity_value")
        if record.get("activity_value") is not None
        else record.get("pchembl_value")
    )
    if not smiles or not target_id or activity_value is None:
        return None
    return ActivityRecord(
        ligand_smiles=str(smiles),
        target_id=str(target_id),
        activity_value=float(activity_value),
        activity_type=str(record.get("activity_type", record.get("standard_type", ""))),
        assay_id=str(record.get("assay_id", "")),
    )


def _canonical_activity_smiles(smiles: str) -> str:
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise RuntimeError("RDKit is required for HUMU activity cliff validation") from exc
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Activity source contains invalid SMILES: {smiles}")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True)


def _activity_cliff_auroc(
    activity_items: list[dict],
    activity_records: dict[str, list[ActivityRecord]],
    eval_cfg: dict,
    *,
    activity_source_configured: bool,
) -> tuple[float | None, str | None]:
    if not activity_source_configured:
        return None, "missing_activity_cliff_labels"
    if (
        "activity_cliff_similarity_threshold" not in eval_cfg
        or "activity_cliff_delta_threshold" not in eval_cfg
    ):
        return None, "missing_activity_cliff_thresholds"
    if not activity_items or not activity_records:
        return None, "missing_activity_cliff_labels"

    similarity_threshold = float(eval_cfg["activity_cliff_similarity_threshold"])
    activity_delta_threshold = float(eval_cfg["activity_cliff_delta_threshold"])
    items_by_smiles: dict[str, list[list[float]]] = {}
    canonical_cache: dict[str, str] = {}
    for item in activity_items:
        ligand_smiles = str(item["ligand_smiles"])
        key = canonical_cache.get(ligand_smiles)
        if key is None:
            key = _canonical_activity_smiles(ligand_smiles)
            canonical_cache[ligand_smiles] = key
        if key in activity_records:
            items_by_smiles.setdefault(key, []).append(item["embedding"])

    by_target: dict[str, dict[str, list[float]]] = {}
    for smiles_key in items_by_smiles:
        for record in activity_records.get(smiles_key, []):
            by_target.setdefault(record.target_id, {}).setdefault(smiles_key, []).append(
                record.activity_value
            )

    embeddings = []
    labels = []
    fingerprint_cache = {}
    from mf_eval.cliff_analysis import cliff_separation_auroc
    for target_values in by_target.values():
        if len(target_values) < 2:
            continue
        smiles = list(target_values)
        activities = [float(median(values)) for values in target_values.values()]
        cliff_indices = _activity_cliff_indices(
            smiles,
            activities,
            similarity_threshold,
            activity_delta_threshold,
            fingerprint_cache,
        )
        cliff_smiles = {smiles[index] for index in cliff_indices}
        for smiles_key in smiles:
            for embedding in items_by_smiles[smiles_key]:
                embeddings.append(embedding)
                labels.append(smiles_key in cliff_smiles)

    if not embeddings:
        return None, "missing_activity_cliff_labels"
    if not any(labels) or all(labels):
        return None, "insufficient_activity_cliff_classes"
    score = cliff_separation_auroc(embeddings, labels)
    if score is None:
        return None, "insufficient_activity_cliff_classes"
    return score, None


def _activity_cliff_indices(
    smiles: list[str],
    activities: list[float],
    similarity_threshold: float,
    activity_delta_threshold: float,
    fingerprint_cache: dict,
) -> set[int]:
    if len(smiles) != len(activities):
        raise ValueError("smiles and activities must have the same length")
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdMolDescriptors
    except ImportError as exc:
        raise RuntimeError("RDKit is required for activity cliff analysis") from exc

    fingerprints = []
    for item in smiles:
        fingerprint = fingerprint_cache.get(item)
        if fingerprint is None:
            mol = Chem.MolFromSmiles(item)
            if mol is None:
                raise ValueError("activity cliff analysis requires valid SMILES")
            fingerprint = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            fingerprint_cache[item] = fingerprint
        fingerprints.append(fingerprint)

    cliff_indices: set[int] = set()
    for i in range(len(fingerprints)):
        for j in range(i + 1, len(fingerprints)):
            if abs(float(activities[i]) - float(activities[j])) < activity_delta_threshold:
                continue
            similarity = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
            if similarity >= similarity_threshold:
                cliff_indices.add(i)
                cliff_indices.add(j)
    return cliff_indices


def _apply_requested_validation_metrics(metrics: dict, cfg: dict) -> None:
    requested = cfg.get("eval", {}).get("metrics", []) or []
    for metric in requested:
        if metric == "mol_pocket_retrieval":
            metrics[metric] = metrics.get("mol_pocket_retrieval_top1", 0.0)
        elif metric == "mol_route_retrieval":
            metrics[metric] = metrics.get("mol_route_retrieval_top1", 0.0)
        elif metric == "pocket_route_retrieval":
            metrics[metric] = metrics.get("pocket_route_retrieval_top1", 0.0)
        elif metric == "protac_component_retrieval":
            metrics[metric] = metrics.get("protac_component_retrieval_top1", 0.0)
        elif metric == "protac_component_library_retrieval":
            metrics[metric] = metrics.get(
                "protac_component_library_retrieval_top1",
                0.0,
            )
        elif metric == "route_template_retrieval":
            metrics[metric] = metrics.get("route_template_retrieval_top1", 0.0)
        elif metric == "protac_ternary_retrieval":
            metrics[metric] = metrics.get("protac_ternary_retrieval_top1", 0.0)
        elif metric == "protein_interface_retrieval":
            metrics[metric] = metrics.get("protein_interface_retrieval_top1", 0.0)
        elif metric == "interface_mutation_retrieval":
            metrics[metric] = metrics.get("interface_mutation_retrieval_top1", 0.0)
        elif metric == "pdc_component_retrieval":
            metrics[metric] = metrics.get("pdc_component_retrieval_top1", 0.0)
        elif metric == "tree_distortion":
            if metric not in metrics:
                metrics[metric] = None
                metrics[f"{metric}_status"] = "missing_route_tree_depth_variance"
        elif metric == "cliff_separation_auroc":
            if metric not in metrics:
                metrics[metric] = None
                metrics[f"{metric}_status"] = "missing_activity_cliff_labels"


def _early_stopping_state(cfg: dict) -> dict:
    early_cfg = cfg.get("early_stopping", {}) or {}
    mode = str(early_cfg.get("mode", "min"))
    if mode not in {"min", "max"}:
        raise ValueError("early_stopping.mode must be 'min' or 'max'")
    return {
        "enabled": bool(early_cfg.get("enabled", False)),
        "monitor": str(early_cfg.get("monitor", "val_loss")),
        "mode": mode,
        "patience": max(1, int(early_cfg.get("patience", 5) or 1)),
        "min_delta": max(0.0, float(early_cfg.get("min_delta", 0.0) or 0.0)),
        "best_value": None,
        "best_epoch": None,
        "bad_checks": 0,
        "stop_epoch": None,
    }


def _update_early_stopping(state: dict, metrics: dict, epoch: int) -> tuple[bool, dict]:
    if not state.get("enabled", False):
        return False, state
    monitor = state["monitor"]
    if monitor not in metrics or metrics[monitor] is None:
        return False, state
    value = float(metrics[monitor])
    best_value = state.get("best_value")
    if best_value is None or _early_stopping_improved(
        value,
        float(best_value),
        state["mode"],
        float(state["min_delta"]),
    ):
        state = dict(state)
        state["best_value"] = value
        state["best_epoch"] = epoch + 1
        state["bad_checks"] = 0
        return False, state

    state = dict(state)
    state["bad_checks"] = int(state.get("bad_checks", 0)) + 1
    if state["bad_checks"] >= int(state["patience"]):
        state["stop_epoch"] = epoch + 1
        return True, state
    return False, state


def _early_stopping_improved(value: float, best_value: float, mode: str, min_delta: float) -> bool:
    if mode == "min":
        return value < best_value - min_delta
    return value > best_value + min_delta


def _format_early_stop_reason(state: dict) -> str:
    return (
        "Early stopping triggered: "
        f"monitor={state['monitor']} "
        f"mode={state['mode']} "
        f"best={float(state['best_value']):.6f} "
        f"best_epoch={state['best_epoch']} "
        f"stop_epoch={state['stop_epoch']} "
        f"patience={state['patience']} "
        f"min_delta={float(state['min_delta']):.6f}"
    )


def _write_validation_metrics(output_dir: Path, epoch: int, metrics: dict) -> Path:
    path = output_dir / "validation_metrics.jsonl"
    record = {"epoch": epoch + 1, **metrics}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return path


def _format_validation_summary(
    epoch: int,
    epochs: int,
    metrics: dict,
    metrics_path: Path,
) -> str:
    parts = [
        f"Validation {epoch + 1}/{epochs}:",
        f"val_loss={metrics['val_loss']:.4f}",
        f"retrieval_top1={metrics['retrieval_top1']:.4f}",
        f"distance_margin={metrics['distance_margin']:.4f}",
        f"collapse_ratio={metrics['collapse_ratio']:.4f}",
        f"samples={metrics['val_samples']}",
        f"metrics={metrics_path}",
    ]
    return " ".join(parts)


def _iter_pocket_batch(batch: dict):
    """Yield individual pocket data dicts from a batch."""
    keys = batch.keys()
    n = len(list(batch.values())[0]) if batch else 0
    for i in range(n):
        yield {k: batch[k][i] for k in keys}


def _iter_route_batch(batch: dict):
    """Yield individual route data dicts from a batch."""
    return _iter_pocket_batch(batch)


def _log_step(
    epoch: int,
    step: int,
    n_batches: int,
    losses: dict,
    lr: float,
    rank: int = 0,
    step_time: float | None = None,
    dataloader_time: float | None = None,
    forward_time: float | None = None,
    backward_time: float | None = None,
    gpu_stats: dict | None = None,
    preserve: bool = False,
) -> None:
    """Render one-line training progress."""
    progress = step / max(n_batches, 1)
    bar_length = 30
    filled = int(bar_length * progress)
    bar = "█" * filled + "░" * (bar_length - filled)
    timing_text = _format_timing(
        step_time,
        dataloader_time,
        forward_time,
        backward_time,
    )
    gpu_text = _format_gpu_stats(gpu_stats)
    component_text = "".join(
        f" | {name}: {_loss_float(losses[name]):.4f}"
        for name in (
            "l_mol_pocket",
            "l_mol_route",
            "l_pocket_route",
            "l_protac_component",
            "l_mol_self",
            "l_activity_supervised",
            "l_route_template",
            "l_protac_ternary",
            "l_protein_interface",
            "l_interface_mutation",
            "l_pdc_component",
            "l_hard_negative",
            "l_curvature_reg",
        )
        if name in losses
    )
    metric_text = "".join(
        f" | {name}: {_loss_float(losses[name]):.4f}"
        for name in (
            "positive_distance",
            "negative_distance",
            "embedding_variance",
            "collapse_ratio",
        )
        if name in losses
    )
    text = (
        f"[{bar}] {progress * 100:.1f}% | "
        f"Rank {rank} | Epoch {epoch + 1} | Batch {step}/{n_batches} | "
        f"Loss: {_loss_float(losses['total']):.4f} | LR: {lr:.6f}"
        f"{component_text}"
        f"{metric_text}"
        f"{timing_text}"
        f"{gpu_text}"
    )
    if preserve:
        sys.stdout.write(text + "\n")
    else:
        sys.stdout.write(f"\r{' ' * 240}\r{text[:240]}")
    sys.stdout.flush()


def _format_timing(
    step_time: float | None,
    dataloader_time: float | None,
    forward_time: float | None,
    backward_time: float | None,
) -> str:
    parts = []
    if step_time is not None:
        parts.append(f"Time: {step_time:.1f}s")
    if dataloader_time is not None:
        parts.append(f"Load: {dataloader_time:.1f}s")
    if forward_time is not None:
        parts.append(f"Fwd: {forward_time:.1f}s")
    if backward_time is not None:
        parts.append(f"Bwd: {backward_time:.1f}s")
    return "".join(f" | {part}" for part in parts)


def _format_gpu_stats(gpu_stats: dict | None) -> str:
    if not gpu_stats:
        return ""
    text = (
        f" | GPU: {gpu_stats['gpu_id']}"
        f" | Mem: {gpu_stats['memory_allocated_mb']:.0f}MB"
    )
    util = gpu_stats.get("utilization_pct")
    if util is not None:
        text += f" | Util: {util:.0f}%"
    return text


def _gpu_stats(device: torch.device) -> dict | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    gpu_id = device.index if device.index is not None else torch.cuda.current_device()
    stats = {
        "gpu_id": gpu_id,
        "memory_allocated_mb": torch.cuda.memory_allocated(gpu_id) / 1024 / 1024,
    }
    try:
        stats["utilization_pct"] = float(torch.cuda.utilization(gpu_id))
    except Exception:  # noqa: BLE001
        stats["utilization_pct"] = None
    return stats


def _clear_progress_line() -> None:
    sys.stdout.write("\r" + " " * 240 + "\r")
    sys.stdout.flush()


def _loss_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


class _null_context:
    """Context manager that does nothing (fallback for non-AMP)."""
    def __enter__(self):
        return None
    def __exit__(self, *args):
        return False


def _amp_dtype_from_config(cfg: dict) -> torch.dtype:
    """Resolve the configured CUDA autocast dtype."""
    value = str(cfg.get("amp_dtype", "float16")).strip().lower()
    if value in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if value in {"float16", "fp16", "half"}:
        return torch.float16
    raise ValueError(
        "amp_dtype must be one of: bfloat16, bf16, float16, fp16, half"
    )


def _cuda_sdp_backend_config(cfg: dict) -> dict[str, bool]:
    """Resolve CUDA scaled-dot-product attention backend safety switches."""
    backend_cfg = cfg.get("cuda_backends", {}) or {}
    return {
        "enable_cudnn_sdp": bool(backend_cfg.get("enable_cudnn_sdp", False)),
    }


def _configure_cuda_backends(cfg: dict, device: torch.device) -> None:
    if device.type != "cuda":
        return
    backend_cfg = _cuda_sdp_backend_config(cfg)
    enable_cudnn_sdp = backend_cfg["enable_cudnn_sdp"]
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(enable_cudnn_sdp)


def _validate_config(config: dict) -> dict:
    """Validate and set defaults for the pre-training configuration."""
    defaults = {
        "embed_dim": 129,
        "batch_size": 64,
        "epochs": 100,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "lambda_curvature": 0.01,
        "num_workers": 4,
        "use_amp": False,
        "gradient_clip_norm": 1.0,
        "output_dir": "checkpoints/humu/",
        "save_every_n_epochs": 5,
        "keep_last_n": 3,
        "resume_from": None,
        "cuda_backends": {"enable_cudnn_sdp": False},
        "ddp_find_unused_parameters": False,
        "early_stopping": {
            "enabled": False,
            "monitor": "val_loss",
            "mode": "min",
            "patience": 5,
            "min_delta": 0.0,
        },
        "loss_weights": {
            "mol_pocket": 1.0,
            "mol_route": 0.5,
            "pocket_route": 0.0,
            "protac_component": 0.0,
            "curvature_reg": 0.0,
        },
    }
    if "intent" in (config.get("loss_weights") or {}):
        raise ValueError("loss_weights.intent is not supported by HUMU pretraining")
    data_cfg = config.get("data") or {}
    if "intent_source" in data_cfg:
        raise ValueError("data.intent_source is not supported by HUMU pretraining")
    if "joint_oversample_factor" in data_cfg:
        raise ValueError(
            "data.joint_oversample_factor is not supported by HUMU pretraining; "
            "use data.objective_sampling.objectives instead"
        )
    cfg = {**defaults, **config}
    if "learning_rate" not in config and "lr" in config:
        cfg["learning_rate"] = config["lr"]
    if "embed_dim" not in config and "manifold_dim" in config:
        cfg["embed_dim"] = int(config["manifold_dim"]) + 1
    if float(cfg.get("curvature", 1.0)) != 1.0:
        raise ValueError(
            "curvature must be 1.0: the Lorentz manifold distance is only "
            "mathematically correct for unit curvature in this implementation"
        )
    return cfg
