# HUMU pretrain intent cleanup plan

## Goal

Remove the intent encoder placeholder from HUMU pretraining and clear training-specific residual configuration, data-contract, loss, checkpoint, and test references.

## Call Chain

`pipelines/humu_pretrain/train.py` loads local package paths and invokes `humu_pretrain.pipeline.run()`.

`run()` validates config, builds encoders through `_build_encoders()`, wraps them for DDP, creates dataloaders through `humu_pretrain.data_loader.create_dataloaders()`, then trains each paired batch through `_forward_paired_batch()`.

`_forward_paired_batch()` currently encodes molecule, pocket, route, and intent payloads, then forwards all optional embeddings to `_compute_losses()`.

`_compute_losses()` currently computes molecule-pocket, molecule-route, pocket-route, intent, and curvature losses. Validation and progress logging consume the same loss keys.

`_save_checkpoint()` saves every encoder from the `encoders` dict as `encoder_<name>`. Removing the intent tower removes new `encoder_intent` checkpoint writes while `_restore_checkpoint_state()` continues to ignore extra legacy keys.

## Evidence

- `configs/models/humu_pretrain.yaml` describes three encoders but still carries `loss_weights.intent: 0.0`.
- `pipelines/humu_pretrain/src/humu_pretrain/pipeline.py` imports and builds `HUMUIntentEncoder`, defines `_DDP_DUMMY_INTENT`, and computes `l_intent`.
- `pipelines/humu_pretrain/src/humu_pretrain/data_loader.py` defines `IntentDataset` and `data.intent_source` handling.
- `pipelines/humu_pretrain/train.py`, `tests/conftest.py`, root `pyproject.toml`, and `uv.lock` include `humu_intent_encoder`.
- `tests/unit/test_humu_training.py` asserts the intent tower and intent data contract.

## Design

Keep HUMU pretraining as a three-tower training pipeline: molecule, pocket, route. Remove intent from the pretraining pipeline only. Do not alter CIG, IntentCone, orchestrator, generator conditioning, or downstream intent feedback code.

Configuration validation will reject `loss_weights.intent` and `data.intent_source` so stale pretraining configs fail explicitly instead of silently implying an intent tower exists.

Existing checkpoints with `encoder_intent` remain loadable because restore iterates over current encoder names and ignores extra checkpoint keys.

## Files

- Modify `pipelines/humu_pretrain/src/humu_pretrain/pipeline.py`
- Modify `pipelines/humu_pretrain/src/humu_pretrain/data_loader.py`
- Modify `pipelines/humu_pretrain/train.py`
- Modify `configs/models/humu_pretrain.yaml`
- Modify `tests/unit/test_humu_training.py`
- Modify `tests/conftest.py`
- Modify `pyproject.toml`
- Regenerate or edit `uv.lock` through `uv lock`
- Delete `models/mf-encoders/humu_intent_encoder`

## Validation

- Run focused HUMU tests: `pytest tests/unit/test_humu_training.py`
- Run artifact requirement tests: `pytest tests/unit/test_artifact_requirements.py`
- Search for remaining training-specific references: `rg -n "humu_intent|mf-encoders-humu-intent|encoder_intent|l_intent|intent_source|loss_weights.*intent|pretrain_intent" ...`
