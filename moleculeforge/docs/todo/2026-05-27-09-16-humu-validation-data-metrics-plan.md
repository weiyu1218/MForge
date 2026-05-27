# HUMU Validation Data Metrics Plan

## Goal

Add real validation support for HUMU `cliff_separation_auroc` without changing training loss or fabricating route depth.

## Current Call Chain

Validation runs through `humu_pretrain.pipeline._validate_epoch`, which calls `_forward_paired_batch`, computes losses, then aggregates requested metrics before `_write_validation_metrics` appends `validation_metrics.jsonl`.

`tree_distortion` is already computed by `_route_tree_distortion(route_emb, route_items)`. It returns `None` when route embeddings are missing, fewer than two route samples are present, or route depth values have no variance. Current route and joint samples checked from `data/processing/humu_pretrain` contain `tree_depth=1` in the sampled records, so a null result is expected and must not be replaced with synthetic depth.

`cliff_separation_auroc` is currently hardcoded to `None` in `_apply_requested_validation_metrics`. The repository already provides `mf_eval.cliff_analysis.find_activity_cliffs` and `cliff_separation_auroc`, but validation does not collect activity labels or molecule embeddings for this metric.

## Implementation Scope

- Add an optional `data.activity_source` JSONL directory contract.
- Load activity records keyed by ligand SMILES and target ID.
- During validation, collect molecule embeddings and ligand SMILES from batches.
- Match validation ligands to `data.activity_source` records outside the training dataset.
- Aggregate activity records by molecule-target before computing activity cliff labels with RDKit similarity and activity delta thresholds, then compute AUROC.
- Report explicit status strings when data is absent, insufficient, or single-class.
- Keep `tree_distortion` null when route depth has no variance.

## Non-Goals

- Do not change training loss.
- Do not fine-tune or restart training.
- Do not fabricate multi-step route depth from single-step USPTO-MIT reactions.
- Do not mutate existing processed dataset files in place.

## KISS Check

1. Real problem: yes, current validation metrics are null because required validation data and calculation are missing.
2. Simpler method: yes, add optional validation data and metric computation instead of changing training.
3. Breakage risk: existing configs without `activity_source` must continue to run and report a null status.
4. Needed now: yes, user requested these validation gaps be filled.

## Verification

- Add unit tests for activity matching, cliff AUROC computation, missing threshold status, and activity preflight.
- Run targeted HUMU training tests.
