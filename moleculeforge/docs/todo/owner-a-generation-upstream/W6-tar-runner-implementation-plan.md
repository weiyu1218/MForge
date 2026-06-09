# W6 TAR ProxylessNAS Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide a concrete local `TAR_PROXYLESS_SEARCH_COMMAND` runner that executes the existing TAR ProxylessNAS reward-cost scheduler from stdin JSON and returns service-compatible stdout JSON.

**Architecture:** The runner lives inside `generator_router_svc` so it can be called as `python -m generator_router_svc.tar_proxyless_runner`. It reuses `TaskAwareRouter` and `ProxylessSearchScheduler`; it does not create a separate TAR algorithm or training data pipeline.

**Tech Stack:** Python 3.11+, PyTorch, `mf_core.routing.task_router`, generator-router gRPC command contract, pytest for focused validation when authorized.

---

## Files

- Create: `moleculeforge/services/generator-router-svc/src/generator_router_svc/tar_proxyless_runner.py`
- Modify: `moleculeforge/tests/unit/test_task_router.py`
- Modify: `moleculeforge/docs/architecture/current-implementation-vs-corearchitecture-v2.md`
- Modify: `moleculeforge/docs/architecture/corearchitecture-v2-completion-tasksplit.md`
- Modify: `moleculeforge/docs/architecture/corearchitecture-v2-completion-interface-acceptance.md`
- Modify: `moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`

## Task 1: Add Runner Module

- [x] **Step 1: Create module skeleton**

Create `tar_proxyless_runner.py` with:

```python
from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from typing import Any

from mf_core.routing.task_router import (
    GENERATOR_NAMES,
    ProxylessSearchScheduler,
    TaskAwareRouter,
)
```

- [x] **Step 2: Add payload normalization**

Implement `_payload_mapping(payload, name)` to require dict values and `_payload_float(payload, name, default)` to parse finite floats where needed.

- [x] **Step 3: Run scheduler through existing TAR code**

Implement `run_proxyless_search(payload: dict[str, Any]) -> dict[str, Any]`:

```python
router = TaskAwareRouter(n_generators=len(GENERATOR_NAMES))
scheduler = ProxylessSearchScheduler(
    router=router,
    generator_costs=generator_costs,
    cost_weight=cost_weight,
    learning_rate=learning_rate,
    temperature=temperature,
)
result = scheduler.run(reward_batches_by_dataset)
```

Return a dict containing:

- `rounds`
- `architecture_probabilities`
- `architecture_logits`
- `generator_names`
- `cost_weight`
- `learning_rate`
- `temperature`

- [x] **Step 4: Add CLI entry**

Implement `main(argv: Sequence[str] | None = None) -> int` that reads stdin, parses JSON, prints sorted JSON to stdout, prints errors to stderr, and returns `1` for invalid input.

## Task 2: Add Focused Tests

- [x] **Step 1: Import runner module in test file**

Use local imports inside tests to avoid broad import-time side effects.

- [x] **Step 2: Add direct runner behavior test**

Add a test that calls `run_proxyless_search()` with two reward batches and asserts:

- `rounds` has length 2;
- `architecture_probabilities["fragfm"] > architecture_probabilities["hfm_3d"]`;
- `architecture_logits` contains active generator names;
- `generator_names == GENERATOR_NAMES`.

- [x] **Step 3: Add CLI subprocess test**

Add a test that runs:

```bash
python -m generator_router_svc.tar_proxyless_runner
```

with stdin JSON and asserts exit code `0`, valid stdout JSON, and empty stderr.

- [x] **Step 4: Add service command integration test**

Add or update a test that configures:

```python
TAR_PROXYLESS_SEARCH_COMMAND = f"{sys.executable} -m generator_router_svc.tar_proxyless_runner"
```

and calls `GeneratorRouterServicer().RunProxylessSearch()` to prove the service can invoke the real runner command.

## Task 3: Documentation Update

- [x] **Step 1: Update current implementation comparison**

Change W6 wording from "runner command missing" to "local runner command target exists; production dataset/cluster validation still missing".

- [x] **Step 2: Update task split**

Mark W6 AI-local code as implemented, while keeping production training data/artifact as a remaining external gate.

- [x] **Step 3: Update interface acceptance**

Record the concrete command:

```bash
python -m generator_router_svc.tar_proxyless_runner
```

and the payload/result contract.

- [x] **Step 4: Update Owner A progress**

Record modified files, verification commands, skipped pytest status if not authorized, and back-check.

## Task 4: Verification

- [x] **Step 1: Static syntax check**

Run:

```bash
python -m py_compile moleculeforge/services/generator-router-svc/src/generator_router_svc/tar_proxyless_runner.py moleculeforge/tests/unit/test_task_router.py
```

Expected: exit code `0`.

- [x] **Step 2: Diff hygiene**

Run:

```bash
git diff --check
```

Expected: exit code `0`.

- [ ] **Step 3: Focused pytest only if authorized**

Run only with explicit test authorization:

```bash
uv run pytest tests/unit/test_task_router.py -q
```

Expected: exit code `0`.

## Back-Check Criteria

- [x] The runner reuses `ProxylessSearchScheduler`.
- [x] The runner accepts the same payload used by `_proxyless_search_from_command`.
- [x] The service can use the runner as a real `TAR_PROXYLESS_SEARCH_COMMAND`.
- [x] Production training data and cluster validation remain marked incomplete.
- [x] HUMU pretraining, HUMU encoder, HFM architecture, and checkpoints are untouched.
- [x] `/workspace/SemMol` and `/workspace/Projects` are untouched.
