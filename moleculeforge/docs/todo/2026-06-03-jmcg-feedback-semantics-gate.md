# JMCG Feedback Semantics Gate

**Date:** 2026-06-03

**Parent plan:** `moleculeforge/docs/todo/2026-06-03-corearchitecture-v2-continuation-governance-plan.md`

**Previous gate:** `moleculeforge/docs/todo/2026-06-03-jmcg-feedback-contract-brief.md`

**Selected technical gate:** harden local HUMU feedback steering semantics for `jmcg_feedback`.

---

## 1. Scope

This gate improves the local feedback semantics after the `moleculeforge.jmcg.feedback.v1` envelope was introduced.

In scope:

- HFM-3D parser behavior for contract-shaped feedback records.
- Per-record `weight * confidence` handling.
- `polarity="attract"` and `polarity="repel"` steering behavior.
- Invalid embedding dimension accounting.
- Per-kind aggregation so one feedback kind cannot dominate only by record count.
- Metadata that makes steering decisions auditable.

Out of scope:

- Default HFM reading shared CRG directly.
- Property / pocket / intent producer implementation.
- Production DKI, Sigstore, benchmark, or KRAS full-pilot gates.
- Claims that JMCG joint sampling is complete.

## 2. Engineering Boundary

Confirmed decision:

- Default HFM must not directly read shared CRG.
- GeneratorCoord remains the explicit route HUMU feedback injection point.
- HFM-3D consumes feedback only through generator parameters.

## 3. Target Behavior

HFM-3D should:

- Accept legacy `route_humu_feedback` and `generation_feedback`.
- Accept contract-shaped `jmcg_feedback`.
- Drop records whose embedding dimension does not match active latent dimension.
- Drop records with invalid weight, invalid confidence, or unknown polarity.
- Clamp confidence to `[0, 1]`.
- Use effective record weight `weight * confidence`.
- Aggregate records within each `kind` first, then aggregate kinds.
- Treat `polarity="attract"` as movement toward the target.
- Treat `polarity="repel"` as movement away from the target, without using negative Euclidean averaging.
- Preserve global `feedback_steering_weight` and `feedback_steering_max_step` clamps.

## 4. Expected Metadata

Generated molecule metadata should expose:

- `feedback_steering_count`
- `feedback_steering_dropped_count`
- `feedback_steering_sources`
- `feedback_steering_kinds`
- `feedback_steering_effective_weight`
- `feedback_steering_weight`
- `feedback_steering_max_step`

## 5. Test Specification

Focused test specs should be added before code changes:

- `jmcg_feedback` with `weight=0` or `confidence=0` does not steer.
- Invalid embedding dimensions are dropped and counted.
- `polarity="repel"` increases distance from the feedback embedding when a safe step is possible.
- Multiple route records and one molecule record aggregate by kind before global target aggregation.

Per `.claude/CLAUDE.md`, pytest must not be run unless explicitly instructed by the user.

## 6. Back-Check Checklist

- [x] Did the implementation keep default HFM free of direct shared CRG reads?
- [x] Did legacy route feedback remain compatible?
- [x] Did metadata distinguish accepted and dropped feedback?
- [x] Did documentation state that this is local HUMU steering, not completed JMCG?
- [x] Were only static checks run locally?

## 7. Implementation Result

Modified:

- `moleculeforge/models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/generator.py`
- `moleculeforge/tests/unit/test_generators.py`

Completed:

- HFM-3D now filters feedback records by valid embedding dimension, non-negative weight, valid confidence, and known polarity.
- Confidence is clamped to `[0, 1]`.
- Effective record weight is computed as `weight * confidence`.
- Records are aggregated within each feedback `kind` before global aggregation.
- Initial kind weights are applied for `molecule`, `route`, `property`, `pocket`, `intent`, and `unspecified`.
- `polarity="repel"` is represented by reversing the spatial HUMU coordinates before projection, avoiding negative Euclidean averaging across the full Lorentz vector.
- Steering metadata now includes accepted count, dropped count, kind count, kinds, sources, and effective weight.

Focused test specs added:

- zero effective weight does not steer
- invalid embedding dimensions are dropped and counted
- repel feedback moves away from the feedback embedding
- per-kind aggregation records kind count and accepted count

Verification:

- `python -m py_compile` passed for HFM generator and generator tests.
- Pytest was not run because `.claude/CLAUDE.md` forbids local test execution without explicit instruction.

Remaining boundary:

- This remains local contract-shaped HUMU steering.
- Property / pocket / intent producers are not implemented in this gate.
- JMCG joint sampling remains incomplete.
