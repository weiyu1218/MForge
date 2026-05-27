# HUMU Mid-Epoch Resume Plan

## Goal

Support exact mid-epoch resume for future HUMU step checkpoints while treating legacy step checkpoints without resume metadata as epoch-start resumes.

## Current Call Chain

`train.py` loads YAML and CLI overrides, then calls `humu_pretrain.pipeline.run`.

`run` builds encoders, wraps DDP, creates dataloaders, builds optimizer and scheduler, then loads `resume_from`.

Current checkpoint save writes only `epoch`, `loss`, optimizer, scheduler, and encoder state. Step checkpoint filenames include global step, but checkpoint payload does not include epoch-local step, batch count, or accumulated epoch loss.

Current checkpoint load always returns `epoch + 1`, so a step checkpoint saved during epoch 5 is treated as if epoch 5 completed.

## Decision

Do not infer batch position from legacy checkpoint filenames. Existing `checkpoint_step_00010500.pt` lacks reliable resume metadata, so this run should restart epoch 5 from `best_model.pt`.

For future step checkpoints, save explicit metadata:

- `checkpoint_type`
- `global_step`
- `epoch_step`
- `n_batches`
- `best_loss`
- `epoch_loss_sum`
- `epoch_loss_count`

Training resume will use the metadata only when present. Legacy step checkpoints start from their stored epoch with `epoch_step=0`.

## Risk

Future mid-epoch resume depends on deterministic sampler order for the same epoch. Existing `DistributedSampler.set_epoch(epoch)` preserves that assumption.

Legacy checkpoints cannot reconstruct RNG state or already accumulated epoch loss, so they are not used for exact mid-epoch resume.
