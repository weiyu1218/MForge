# W2 Pocket / Intent Embedding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade eligible pocket / intent `jmcg_feedback` records from context-only to steering-capable records without inventing unsupported HUMU embeddings.

**Architecture:** Keep HUMU pretraining and HFM architecture frozen. Add optional orchestrator-side feedback enrichment that can call the existing HUMU encoder service for structured pocket geometry and can accept only already-valid 129-dimensional Lorentz full-coordinate intent sources. If enrichment is unavailable or invalid, preserve the current non-steering record behavior.

**Tech Stack:** Python, orchestrator-svc, gRPC `HUMUEncoderService`, `moleculeforge.jmcg.feedback.v1`, HFM-3D Lorentz feedback validation.

---

## File Map

- Modify: `moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`
  - Register the shared-file occupation before touching `orchestrator-svc/main.py`.
- Modify: `moleculeforge/services/orchestrator-svc/src/orchestrator_svc/main.py`
  - Add optional HUMU pocket encoding helpers.
  - Add safe 129-dimensional intent embedding extraction.
  - Keep fallback non-steering records unchanged.
- Modify: `moleculeforge/tests/unit/test_service_artifact_status.py`
  - Add focused specs for pocket enrichment, metadata-only pocket fallback, valid intent axis, and invalid 128-dimensional intent fallback.
- Modify after implementation: `moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`
  - Add execution log and back-check.
- Modify after implementation: `moleculeforge/docs/architecture/current-implementation-vs-corearchitecture-v2.md`
  - Update JMCG row only to claim W2 engineering enrichment, not completed JMCG.

## Task 1: Register Shared File Occupation

**Files:**

- Modify: `moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`

- [ ] Add a W2 implementation entry before touching business code:

```markdown
## 2026-06-03 W2 Implementation Gate Started

Shared file occupation:

- `moleculeforge/services/orchestrator-svc/src/orchestrator_svc/main.py`
  - Functions / region: `_jmcg_context_feedback_from_state`, `_intent_jmcg_feedback_record`, `_pocket_jmcg_feedback_record`, new local HUMU feedback helper functions.
  - Purpose: optional pocket / intent steering-capable JMCG feedback enrichment.

Scope guard:

- Do not change HUMU pretraining.
- Do not change HUMU encoder architecture.
- Do not change HFM model architecture.
- Do not modify `/workspace/SemMol` or `/workspace/Projects`.
```

- [ ] Back-check the entry:
  - It names the exact shared file.
  - It names the exact functions / region.
  - It keeps the frozen boundaries.

## Task 2: Add Failing Specs For W2 Enrichment

**Files:**

- Modify: `moleculeforge/tests/unit/test_service_artifact_status.py`

- [ ] Add a spec showing structured pocket geometry can produce a steering-capable pocket record through an injected helper.

Target shape:

```python
@pytest.mark.asyncio
async def test_full_workflow_generator_receives_pocket_feedback_embedding_when_encoder_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "orchestrator_full_workflow_pocket_embedding_test",
        ROOT / "services/orchestrator-svc/src/orchestrator_svc/main.py",
    )

    async def encode_pocket(payload):
        assert payload["coords"] == [[0.0, 0.0, 0.0]]
        return {
            "humu_embedding": [1.0] + [0.0] * 128,
            "curvature": 1.0,
            "source": "humu_encoder_svc",
            "evidence_ids": ["pocket-geometry"],
        }

    calls: list[object] = []

    class HFMGeneratorServicer:
        async def Generate(self, request, context):
            calls.append(request)
            return SimpleNamespace(
                molecules=[
                    json.dumps({"smiles": "CCO", "canonical_smiles": "CCO"}).encode()
                ]
            )

    fake_hfm_module = ModuleType("hfm_generator_svc.main")
    fake_hfm_module.HFMGeneratorServicer = HFMGeneratorServicer
    monkeypatch.setitem(sys.modules, "hfm_generator_svc.main", fake_hfm_module)
    monkeypatch.setattr(module, "_encode_pocket_humu_feedback", encode_pocket)

    await module.FullWorkflowClients().generate_candidates(
        {
            "run_id": "run-pocket-embedding",
            "cig": {
                "target_context": {
                    "pocket_id": "switch-ii",
                    "coords": [[0.0, 0.0, 0.0]],
                    "elements": ["C"],
                    "residue_types": ["GLY"],
                },
            },
            "request": {"n_samples": 1},
        }
    )

    jmcg_feedback = json.loads(calls[0].generator_params["jmcg_feedback"])
    pocket = next(record for record in jmcg_feedback["records"] if record["kind"] == "pocket")
    assert len(pocket["humu_embedding"]) == 129
    assert pocket["curvature"] == 1.0
    assert pocket["source"] == "humu_encoder_svc"
    assert pocket["evidence_ids"] == ["pocket-geometry"]
```

- [ ] Add a spec showing metadata-only pocket context stays non-steering.

Expected assertion:

```python
assert "humu_embedding" not in pocket_record
assert pocket_record["metadata"] == {"pocket_id": "switch-ii", "pdb_id": "6OIM"}
```

- [ ] Add a spec showing a valid 129-dimensional intent axis can become steering-capable.

Expected assertion:

```python
assert len(intent_record["humu_embedding"]) == 129
assert intent_record["metadata"]["embedding_source"] == "intent_cone.axis"
```

- [ ] Add a spec showing a plain 128-dimensional HCIV vector is not injected as `humu_embedding`.

Expected assertion:

```python
assert "humu_embedding" not in intent_record
assert intent_record["metadata"]["has_hciv"] is True
```

- [ ] Do not run pytest unless explicitly authorized by the user.

## Task 3: Add Safe W2 Helpers In Orchestrator

**Files:**

- Modify: `moleculeforge/services/orchestrator-svc/src/orchestrator_svc/main.py`

- [ ] Add import:

```python
import struct
```

- [ ] Add helper to validate current HFM steering embedding shape:

```python
_CURRENT_HFM_LORENTZ_DIM = 129


def _valid_hfm_feedback_embedding(value: object) -> list[float] | None:
    if not isinstance(value, list) or len(value) != _CURRENT_HFM_LORENTZ_DIM:
        return None
    try:
        embedding = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return embedding
```

- [ ] Add helper for intent embedding extraction:

```python
def _intent_feedback_embedding(state: dict) -> tuple[list[float] | None, dict]:
    intent_cone = state.get("intent_cone")
    if not isinstance(intent_cone, dict):
        return None, {}
    axis = _valid_hfm_feedback_embedding(intent_cone.get("axis"))
    if axis is None:
        return None, {}
    return axis, {"embedding_source": "intent_cone.axis"}
```

- [ ] Add helper to detect structured pocket geometry:

```python
def _pocket_encoder_payload(target_context: dict) -> dict | None:
    coords = target_context.get("coords") or target_context.get("coordinates")
    elements = target_context.get("elements")
    residues = target_context.get("residue_types") or target_context.get("residues")
    if not isinstance(coords, list) or not isinstance(elements, list) or not isinstance(residues, list):
        return None
    if len(coords) != len(elements) or len(coords) != len(residues) or not coords:
        return None
    return {
        "coords": coords,
        "elements": elements,
        "residue_types": residues,
    }
```

- [ ] Add gRPC helper that reuses `HUMU_ENCODER_TARGET` and returns `None` on unavailable target:

```python
async def _encode_pocket_humu_feedback(payload: dict) -> dict | None:
    target = os.environ.get("HUMU_ENCODER_TARGET", "").strip()
    if not target:
        return None
    from mf_core.proto_gen.moleculeforge.v1.humu import encoder_pb2, encoder_pb2_grpc

    channel = grpc.aio.insecure_channel(target)
    stub = encoder_pb2_grpc.HUMUEncoderServiceStub(channel)
    response = await stub.Encode(
        encoder_pb2.EncodeRequest(
            entity_type="pocket",
            input_data=json.dumps(payload, sort_keys=True).encode("utf-8"),
        )
    )
    embedding = _float32_embedding_from_bytes(response.humu_embedding)
    embedding = _valid_hfm_feedback_embedding(embedding)
    if embedding is None:
        return None
    return {
        "humu_embedding": embedding,
        "curvature": float(response.curvature),
        "source": "humu_encoder_svc",
        "evidence_ids": ["humu_encoder:pocket"],
    }
```

- [ ] Add float32 bytes helper:

```python
def _float32_embedding_from_bytes(payload: bytes) -> list[float]:
    if len(payload) % 4 != 0:
        raise ValueError("HUMU embedding bytes must contain float32 values")
    return [float(item[0]) for item in struct.iter_unpack("<f", payload)]
```

## Task 4: Wire Intent And Pocket Records Conservatively

**Files:**

- Modify: `moleculeforge/services/orchestrator-svc/src/orchestrator_svc/main.py`

- [ ] Convert `_attach_generation_feedback` and `_jmcg_context_feedback_from_state` to async. Do not add a second enrichment path in `FullWorkflowClients.generate_candidates`; keep all feedback assembly inside `_attach_generation_feedback`.

Preferred shape:

```python
async def _attach_generation_feedback(generator_params: dict, state: dict) -> None:
    jmcg_feedback = await _jmcg_context_feedback_from_state(state)
    ...
```

- [ ] Update the only business-code call site in `FullWorkflowClients.generate_candidates`:

```python
await _attach_generation_feedback(generator_params, state)
```

- [ ] Add intent embedding only when `_intent_feedback_embedding()` returns a 129-dimensional vector:

```python
embedding, embedding_metadata = _intent_feedback_embedding(state)
if embedding is not None:
    record["humu_embedding"] = embedding
    record["metadata"].update(embedding_metadata)
```

- [ ] Add pocket embedding only when `_pocket_encoder_payload()` and `_encode_pocket_humu_feedback()` both succeed:

```python
payload = _pocket_encoder_payload(target_context)
if payload is not None:
    feedback = await _encode_pocket_humu_feedback(payload)
    if feedback is not None:
        record["humu_embedding"] = feedback["humu_embedding"]
        record["curvature"] = feedback["curvature"]
        record["source"] = feedback["source"]
        record["evidence_ids"] = feedback["evidence_ids"]
```

- [ ] Preserve current non-steering behavior if any step fails closed.

## Task 5: Documentation And Back-Check

**Files:**

- Modify: `moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`
- Modify: `moleculeforge/docs/architecture/current-implementation-vs-corearchitecture-v2.md`

- [ ] Add W2 completion entry to `progress.md` with:
  - Files modified.
  - Whether pytest was authorized and run.
  - Static verification commands.
  - Explicit statement that HUMU pretraining and HFM architecture were not changed.

- [ ] Update architecture comparison JMCG row only to say:
  - Pocket records can become steering-capable when structured pocket geometry is encoded by HUMU encoder.
  - Intent records remain conservative unless a 129-dimensional Lorentz full-coordinate source is available.
  - This is still local feedback steering, not JMCG joint sampling.

## Task 6: Verification

**Files:**

- No code changes in this task.

- [ ] Always run:

```bash
python -m py_compile moleculeforge/services/orchestrator-svc/src/orchestrator_svc/main.py moleculeforge/tests/unit/test_service_artifact_status.py
git diff --check
rg -n "[ \t]+$" moleculeforge/services/orchestrator-svc/src/orchestrator_svc/main.py moleculeforge/tests/unit/test_service_artifact_status.py moleculeforge/docs/todo/owner-a-generation-upstream moleculeforge/docs/architecture/current-implementation-vs-corearchitecture-v2.md
```

- [ ] Only with explicit user authorization, run focused pytest:

```bash
uv run pytest tests/unit/test_service_artifact_status.py tests/unit/test_generators.py -q
```

Expected result after authorization: focused W2 specs pass, existing route feedback behavior does not regress.

## Self-Review

- Spec coverage: W2 rules from the preflight are represented in Tasks 2-4.
- Placeholder scan: No implementation step depends on a vague placeholder.
- Type consistency: Current HFM feedback embedding length is fixed at 129 for this gate.
- Scope check: This plan does not modify HUMU pretraining, HUMU encoder architecture, or HFM architecture.
