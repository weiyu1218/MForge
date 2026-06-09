# W8-E JMCG Engineering Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local generation-side JMCG engineering skeleton that can build serializable joint `(molecule, route, property, pocket, intent)` samples from HFM candidates and HUMU/JMCG feedback context.

**Architecture:** The skeleton lives under HFM-3D inference as `jmcg_sampler.py`. It is a contract and scoring layer, not a trained joint model. It parses existing `moleculeforge.jmcg.feedback.v1` payloads, validates 129-dimensional HUMU embeddings, and emits deterministic `engineering_skeleton` joint samples without changing HFM production generation defaults.

**Tech Stack:** Python dataclasses, PyTorch tensor distance on `LorentzManifold`, existing `mf_core.types.molecule.Molecule`, `moleculeforge.jmcg.feedback.v1`, pytest specs when authorized.

---

## Files

- Create: `moleculeforge/models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/inference/jmcg_sampler.py`
- Modify: `moleculeforge/models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/inference/__init__.py`
- Modify: `moleculeforge/tests/unit/test_generators.py`
- Modify: `moleculeforge/docs/architecture/current-implementation-vs-corearchitecture-v2.md`
- Modify: `moleculeforge/docs/architecture/corearchitecture-v2-completion-tasksplit.md`
- Modify: `moleculeforge/docs/architecture/corearchitecture-v2-completion-interface-acceptance.md`
- Modify: `moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`

## Task 1: Add JMCG Sampler Module

- [x] **Step 1: Define dataclasses**

Create `JMCGContextRecord` and `JMCGJointSample` dataclasses.

`JMCGContextRecord` fields:

- `kind: str`
- `subject: dict[str, object]`
- `humu_embedding: list[float] | None`
- `weight: float`
- `confidence: float`
- `polarity: str`
- `source: str`
- `metadata: dict[str, object]`

`JMCGJointSample` fields:

- `molecule: dict[str, object]`
- `route: dict[str, object] | None`
- `property_profile: dict[str, object]`
- `pocket: dict[str, object] | None`
- `intent: dict[str, object] | None`
- `joint_score: float`
- `metadata: dict[str, object]`

- [x] **Step 2: Parse feedback records**

Implement `parse_jmcg_context(payload: object) -> list[JMCGContextRecord]`.

Required behavior:

- accept JSON string, bytes, envelope dict with `records`, direct list, or single record dict;
- preserve non-embedding records as context;
- convert numeric embeddings to `list[float]`;
- reject unsupported payload types with `TypeError`.

- [x] **Step 3: Normalize candidates**

Implement helpers to normalize:

- `Molecule` objects;
- mapping candidates with `smiles`, `humu_embedding`, `metadata`;
- route mappings with `route_id`, `reactions`, `humu_embedding`;
- property mappings.

- [x] **Step 4: Score alignment**

Implement `JMCGEngineeringSampler` with:

```python
class JMCGEngineeringSampler:
    def __init__(self, embedding_dim: int = 129, curvature: float = 1.0) -> None: ...

    def sample(
        self,
        molecules: Sequence[object],
        *,
        routes: Sequence[Mapping[str, object]] | None = None,
        property_profile: Mapping[str, object] | None = None,
        jmcg_feedback: object = None,
        max_samples: int | None = None,
    ) -> list[JMCGJointSample]: ...
```

Scoring rules:

- only 129-dimensional embeddings participate in scoring;
- molecule embedding comes from candidate `humu_embedding` first, then candidate metadata `latent`;
- route / pocket / intent embeddings come from normalized route candidates and parsed feedback records;
- lower Lorentz distance means better alignment;
- missing embeddings produce score `0.0` and metadata explaining non-steering context.

## Task 2: Export And Test Specs

- [x] **Step 1: Export names**

Update `inference/__init__.py` to export:

- `JMCGContextRecord`
- `JMCGEngineeringSampler`
- `JMCGJointSample`
- `parse_jmcg_context`

- [x] **Step 2: Add test for legal joint sample output**

Add a focused test in `tests/unit/test_generators.py` that:

- creates a 129-dimensional projected molecule embedding;
- creates route / pocket / intent feedback records with the same embedding;
- calls `JMCGEngineeringSampler.sample()`;
- asserts one JSON-serializable output;
- asserts `metadata["mode"] == "engineering_skeleton"`;
- asserts route / pocket / intent fields are present.

- [x] **Step 3: Add test for invalid dimension handling**

Add a focused test that:

- passes a 128-dimensional molecule embedding and 128-dimensional feedback embedding;
- asserts no embedding participates in scoring;
- asserts metadata records dropped / ignored embedding counts;
- asserts the sample is still emitted as context-only skeleton output.

- [x] **Step 4: Add test for parser compatibility**

Add a focused test that:

- passes an envelope JSON string;
- asserts `parse_jmcg_context()` preserves non-steering property records and steering-capable route records.

## Task 3: Documentation Update

- [x] **Step 1: Update current implementation comparison**

Document that W8-E now has a local generation-side joint sample skeleton, but W8-R joint training quality remains open.

- [x] **Step 2: Update task split and interface acceptance**

Mark W8-E engineering skeleton as locally implemented and keep W8-R as a research gate.

- [x] **Step 3: Update Owner A progress and governance log**

Record files, verification, skipped pytest status, and remaining blockers.

## Task 4: Verification

- [x] **Step 1: Static syntax check**

Run:

```bash
python -m py_compile moleculeforge/models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/inference/jmcg_sampler.py moleculeforge/models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/inference/__init__.py moleculeforge/tests/unit/test_generators.py
```

Expected: exit code `0`.

- [x] **Step 2: Diff hygiene**

Run:

```bash
git diff --check
```

Expected: exit code `0`.

- [x] **Step 3: Command-level smoke**

Run a small `uv run python - <<'PY'` script that imports `JMCGEngineeringSampler`, creates one sample, and prints JSON.

Expected: exit code `0`.

- [ ] **Step 4: Focused pytest only if authorized**

Run only with explicit test authorization:

```bash
uv run pytest tests/unit/test_generators.py -q
```

Expected: exit code `0`.

## Back-Check Criteria

- [x] The skeleton emits explicit joint sample records.
- [x] It does not change default HFM generation behavior.
- [x] It does not fabricate HUMU embeddings from metadata-only context.
- [x] It preserves the 129-dimensional steering-capable contract.
- [x] W8-R production/research quality remains marked incomplete.
- [x] HUMU pretraining and checkpoints are untouched.
