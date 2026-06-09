# Owner A Production Execution Roadmap

Date: 2026-06-06
Scope: Owner A generation-upstream production-resource gates W6, W9, W10, W11,
and W13

## Purpose

This roadmap gives the execution order for the remaining Owner A production
work after the engineering-completion gates. It consolidates the gate-specific
run plans and records what can continue without additional user decisions,
what is resource-gated, and what requires an explicit stop-and-ask decision.

This document does not authorize long training, production deployment changes,
protected artifact writes, benchmark threshold changes, or Owner B
implementation edits.

## Current Position

Owner A has moved from engineering enablement to production evidence.

Completed locally:

- W2 and W8 feedback/JMCG engineering wiring.
- W6 TAR runner command target and service command contract.
- W9 neural HFM geometry decoder engineering path and runner contract.
- W10 supervised HCIV training/export path and production checkpoint loader.
- W11 HUMU-labeled FragFM data, strict-local 5k candidate, deployment-default
  hardening, runtime smoke, sample export, and MOSES validity wiring smoke.
- W13 canonical teacher embedding artifact utility and generator KD consumer
  wiring.

Not completed:

- Production-quality source data for W6, W9, W10, and W13.
- A production-trained FragFM artifact beyond the strict-local 5k candidate.
- Official benchmark evidence without threshold relaxation.
- Cluster readiness evidence for promoted artifacts and service configs.
- Release naming and promotion records for permanent production artifacts.

## Fixed Boundaries

These boundaries apply to every phase below:

- Keep HUMU pretraining frozen.
- Do not modify HUMU pretraining config, loss, encoder architecture, or
  checkpoint continuation.
- Do not overwrite `checkpoints/fragfm`, `checkpoints/humu`, or
  `checkpoints/hfm3d_4h200`.
- Treat `/workspace/SemMol` and `/workspace/Projects` as read/copy-only
  context. Do not write there and do not execute from there.
- Do not print `.env` secrets.
- Do not relax GuacaMol, PMO, or MOSES thresholds.
- Do not modify Owner B implementation files unless the user explicitly
  authorizes that handoff.
- Treat `checkpoints/fragfm_humu_5k/` as strict-local engineering evidence, not
  production acceptance.

## Recommended Execution Order

The recommended order is:

1. W11 production readiness remains the main near-term path because it already
   has local HUMU-labeled data, a strict candidate, deployment defaults, runtime
   smoke, and benchmark input wiring.
2. W6 TAR reward payload preparation is the lowest-compute production-resource
   gate once real reward data is available.
3. W10 HCIV production checkpoint preparation should follow when supervised
   CIG/HCIV data is available, because it improves intent conditioning without
   changing W2 steering rules.
4. W9 HFM decoder production preparation should proceed when real latent/SDF
   decoder source data is available, because the current decoder artifact is
   only smoke/full-flow evidence.
5. W13 KD should start with strict per-consumer teacher artifact export after a
   real teacher source is approved. Distillation runs come later because they
   span multiple consumers and benchmark comparisons.

Resource collection for W6, W9, W10, and W13 can happen in parallel with W11
planning, but long runs and deployment changes still require explicit
authorization.

## Short-Term Work That Can Continue Without New Decisions

These steps are safe because they do not change code behavior, protected
artifacts, deployment defaults, thresholds, or production resource choices.

### Evidence Indexing

Maintain one evidence trail for each gate:

- W6 reward payload source path, hash, preflight output, command smoke, and
  service smoke.
- W9 decoder source path, hash, loader preflight output, candidate artifact
  path, runner smoke, and HFM generator smoke.
- W10 HCIV supervised data path, hash, source preflight output, candidate
  checkpoint path, checkpoint load smoke, and compiler smoke.
- W11 FragFM candidate path, training manifest, strict quality report, runtime
  smoke, sample export report, benchmark wiring output, and cluster smoke.
- W13 teacher source path, hash, strict artifact export report, per-consumer
  dimension, distillation manifest, and benchmark comparison.

Do not invent production evidence. If a source or run does not exist, record the
missing resource as a blocker in the gate-specific gap document rather than
creating a placeholder artifact.

### W11 Evidence Hardening

Keep W11 as the default active gate. The safe near-term work is to keep the
current local evidence easy to verify:

- Re-read the FragFM artifact promotion policy before any default path change.
- Keep `checkpoints/fragfm_humu_5k/` as the current deployment default until a
  promoted replacement passes the recorded gates.
- Preserve the 8, 64, and 256 sample export reports as local benchmark-input
  wiring evidence only.
- Current user direction is to pause large-scale training and complete code
  engineering first.
- Keep improving W11 training/export/runtime observability and focused tests
  without launching stronger training.
- Use the W11 production training run plan only after stronger training is
  explicitly re-approved.
- Record any new W11 evidence in `progress.md` with the exact command and
  observed result.

### Source Data Preflight Preparation

For W6, W9, W10, and W13, the safe next action is to check whether approved
source files already exist under `data/processing/generator_artifacts/` and
then run only the documented preflight commands on real files.

If the expected source file does not exist, do not fabricate it. Record the
missing source file and required provenance in the relevant readiness-gap
document.

## Resource-Gated Production Steps

These steps require either real source data, compute resources, deployment
access, or an explicit user decision.

### W6 TAR

Next production action:

- Prepare a real reward-cost payload using
  `2026-06-06-W6-tar-production-run-plan.md`.
- Run the payload preflight.
- Run command smoke with `python -m generator_router_svc.tar_proxyless_runner`.
- Run service smoke by setting `TAR_PROXYLESS_SEARCH_COMMAND` only in the local
  command environment.

Stop before:

- choosing the production `TAR_PROXYLESS_SEARCH_COMMAND` value;
- changing Docker, Kubernetes, or Helm defaults;
- lowering reward payload gates;
- changing scheduler or router semantics.

Production acceptance requires:

- real reward provenance;
- downstream generator quality comparison;
- cluster config evidence and service request logs.

### W9 HFM Decoder

Next production action:

- Prepare a real decoder source artifact using
  `2026-06-06-W9-hfm-decoder-production-training-run-plan.md`.
- Validate at least 1000 source entries with 129-dimensional Lorentz-valid
  latents and RDKit-parseable SDF.
- Write the candidate artifact only to a new non-protected directory.

Stop before:

- launching decoder training;
- changing the candidate output directory for a long run;
- choosing `HFM_MOLECULAR_DECODER_COMMAND` as a deployment default;
- overwriting `checkpoints/hfm3d_4h200`;
- editing HFM Lorentz flow architecture.

Production acceptance requires:

- source provenance and hash;
- artifact load check;
- runner smoke;
- HFM generator smoke through the command contract;
- geometry benchmark and cluster evidence.

### W10 HCIV

Next production action:

- Prepare real supervised CIG plus target-HCIV JSONL using
  `2026-06-06-W10-hciv-production-training-run-plan.md`.
- Validate at least 1000 records with parseable CIG objects and 129-dimensional
  Lorentz-valid targets.
- Write the candidate checkpoint only to a new non-protected directory.

Stop before:

- launching HCIV training;
- changing `HCIV_CHECKPOINT_PATH` defaults;
- lowering source gates;
- changing W2 steering rules;
- using hash/random demo encoders as production targets.

Production acceptance requires:

- supervised target provenance;
- checkpoint load smoke;
- compiler learned-mode smoke;
- downstream intent-conditioned generation evidence;
- cluster service evidence.

### W11 FragFM

Next production action:

- First complete engineering code hardening and lightweight verification under
  the current no-large-training instruction.
- After stronger training is explicitly re-approved, use
  `2026-06-06-W11-fragfm-production-training-run-plan.md` for the next candidate.
- Keep any future training outputs under a new non-protected candidate directory.
- After training is explicitly approved and completed, run strict quality,
  runtime smoke, sample export, unchanged-threshold benchmark wiring, and
  cluster smoke.

Stop before:

- launching stronger FragFM training;
- changing epochs, hidden dimension, optimizer, device, or output path for a
  long run;
- choosing a permanent production artifact name;
- moving deployment defaults from `checkpoints/fragfm_humu_5k/`;
- editing benchmark thresholds.

Production acceptance requires:

- manifest evidence for data, HUMU coverage, model capacity, optimizer, and
  loss;
- strict quality report;
- production-scale generated samples;
- official benchmark evidence without threshold relaxation;
- real cluster cold-start and request/response evidence.

### W13 KD

Next production action:

- Start with an approved teacher-record source.
- Export strict canonical teacher embedding artifacts per consumer and expected
  dimension using `2026-06-06-W13-kd-production-run-plan.md`.
- Inspect export reports before any non-zero KD run.

Stop before:

- lowering minimum embedding count;
- changing expected dimensions;
- choosing permanent teacher artifact names;
- changing teacher deployment env values;
- launching distillation training;
- changing KD loss semantics.

Production acceptance requires:

- teacher source provenance;
- strict per-consumer artifact report;
- baseline-vs-KD quality comparison;
- official benchmark evidence without threshold relaxation;
- deployment evidence if a teacher service is involved.

## Long-Term Release Path

Production acceptance should be recorded in this order:

1. Source-data approval with hashes and provenance.
2. Candidate artifact creation in a non-protected path.
3. Focused artifact quality and runtime smoke.
4. Production-scale generated sample export or service output capture.
5. Official benchmark run without threshold relaxation.
6. Cluster cold-start, readiness, and service request/response evidence.
7. Promotion decision naming the permanent artifact path and deployment config.
8. Architecture and handoff document updates.

No gate should be marked production-complete until all relevant evidence exists
and the readiness-gap document no longer lists a blocking production resource.

## Back-Check Protocol

After each step, perform the smallest useful verification before continuing.

Documentation-only changes:

```bash
git diff --check -- docs/todo/owner-a-generation-upstream
.venv/bin/python - <<'PY'
from pathlib import Path

patterns = [
    "-" + " [ ]",
    "T" + "BD",
    "TO" + "DO",
    "implement " + "later",
    "fill " + "in",
]
for path in Path("docs/todo/owner-a-generation-upstream").glob("*.md"):
    text = path.read_text(encoding="utf-8")
    for pattern in patterns:
        if pattern in text:
            print(f"{path}:{pattern}")
PY
```

Artifact or data creation:

```bash
git status --short -- moleculeforge/checkpoints/fragfm \
  moleculeforge/checkpoints/humu \
  moleculeforge/checkpoints/hfm3d_4h200
```

Long-running process safety:

```bash
ps -eo pid,etime,cmd | rg -n "train|pytest|sample_export|tar_proxyless|kd_artifacts" || true
```

Focused tests or smokes:

- Run the exact gate-specific command from the relevant run plan.
- Read the exit code and output before recording success.
- Record warnings separately from failures.

## Stop-And-Ask Decisions

Ask the user before:

- launching long training or distillation;
- choosing permanent production artifact names;
- changing Docker, Kubernetes, Helm, or service env defaults;
- lowering source-data gates or benchmark thresholds;
- modifying Owner B implementation files;
- changing HUMU pretraining or HFM Lorentz flow architecture;
- overwriting or deleting protected or historical artifacts;
- killing processes that may belong to external work.

If none of these decisions is present, continue with the next focused
preflight, documentation update, or verification step and record the result.
