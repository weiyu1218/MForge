/* MoleculeForge full-workflow client. */

const API = "/v1";
let bearerToken = "";

function $(s) { return document.querySelector(s); }
function $$(s) { return Array.from(document.querySelectorAll(s)); }

function fmt(n, d = 2) {
  if (n === null || n === undefined) return "—";
  if (typeof n === "boolean") return n ? "✓" : "✗";
  if (typeof n !== "number") return String(n);
  if (Number.isInteger(n)) return String(n);
  return Number(n).toFixed(d);
}

function badge(text, kind = "muted") {
  return `<span class="badge ${kind}">${text}</span>`;
}

function authHeaders() {
  if (!bearerToken) {
    throw new Error("Bearer Token is required.");
  }
  return { Authorization: `Bearer ${bearerToken}` };
}

async function api(path, opts = {}) {
  const { headers = {}, ...requestOptions } = opts;
  const res = await fetch(API + path, {
    ...requestOptions,
    headers: {
      "Content-Type": "application/json",
      ...authHeaders(),
      ...headers,
    },
  });
  const text = await res.text();
  let json = null;
  try { json = text ? JSON.parse(text) : null; } catch { json = { raw: text }; }
  if (!res.ok) {
    const detail = json?.detail ? JSON.stringify(json.detail) : text;
    throw new Error(`${res.status} ${res.statusText}: ${detail}`);
  }
  return json;
}

/* SmilesDrawer factory — single shared instance so we don't rebuild themes. */
const drawer = new SmilesDrawer.Drawer({
  width: 220, height: 160,
  bondThickness: 0.9,
  fontSizeLarge: 7,
  fontSizeSmall: 4,
  themes: {
    light: {
      C: '#1f2632', O: '#cc1f1f', N: '#3a51b5', S: '#c79b3a',
      F: '#1aa069', Cl: '#1aa069', Br: '#a06b1f', I: '#7c2dbf',
      P: '#a14fa1', BACKGROUND: '#ffffff',
    }
  }
});

function renderMolecule(canvas, smiles) {
  if (!canvas) return;
  SmilesDrawer.parse(smiles, (tree) => {
    drawer.draw(tree, canvas, "light", false);
  }, () => {
    // parse error — fall back to text label
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#888";
    ctx.font = "11px monospace";
    ctx.fillText(smiles.slice(0, 24), 6, 16);
  });
}

/* ---------------- device status ---------------- */

async function pollHealth() {
  const el = $("#device-status");
  try {
    const h = await fetch("/health").then((r) => r.json());
    const n = h.gpu?.device_count ?? 0;
    if (h.status === "healthy" && n > 0) {
      el.classList.remove("warn", "bad");
      el.classList.add("ok");
      el.innerHTML = `● ${n}× CUDA <span class="muted small">${(h.devices||[]).join(", ")}</span>`;
    } else {
      el.classList.remove("ok", "bad");
      el.classList.add("warn");
      el.textContent = "CPU only";
    }
  } catch {
    el.classList.remove("ok", "warn");
    el.classList.add("bad");
    el.textContent = "backend unreachable";
  }
}
pollHealth();
setInterval(pollHealth, 15000);

/* ---------------- composer ---------------- */

const intentEl = $("#intent");
const meta = $("#intent-meta");
intentEl.addEventListener("input", () => {
  const len = intentEl.value.trim().length;
  meta.textContent = len ? `${len} chars` : "";
});

$$(".chip").forEach((c) => {
  c.addEventListener("click", () => {
    intentEl.value = c.dataset.intent;
    intentEl.dispatchEvent(new Event("input"));
    intentEl.focus();
  });
});

$("#new-run").addEventListener("click", () => {
  invalidateActiveRun();
  $("#workbench").hidden = true;
  intentEl.value = "";
  intentEl.dispatchEvent(new Event("input"));
  intentEl.focus();
  window.scrollTo({ top: 0, behavior: "smooth" });
});

$("#bearer-token").addEventListener("change", () => {
  bearerToken = $("#bearer-token").value.trim();
  refreshHistory();
});

$("#run").addEventListener("click", async () => {
  const intent = intentEl.value.trim();
  if (!intent) {
    alert("Tell me what to design first.");
    return;
  }
  bearerToken = $("#bearer-token").value.trim();
  const requiredInputs = [
    "#bearer-token",
    "#project-id",
    "#max-refinements",
    "#n-samples",
    "#generation-strategy",
    "#retrosyn-engine",
    "#validation-policy",
    "#teacher-version",
    "#allow-synthetic",
    "#kd-weight",
    "#selection-policy",
  ];
  const invalidInput = requiredInputs
    .map((selector) => $(selector))
    .find((element) => !element.checkValidity());
  if (invalidInput) {
    invalidInput.reportValidity();
    return;
  }
  let validationPolicy;
  let selectionPolicy;
  try {
    validationPolicy = JSON.parse($("#validation-policy").value);
    selectionPolicy = JSON.parse($("#selection-policy").value);
  } catch (error) {
    alert(`Policy JSON is invalid: ${error.message}`);
    return;
  }
  $("#run").disabled = true;
  try {
    const r = await api("/orchestrator/design", {
      method: "POST",
      body: JSON.stringify({
        workflow_scope: "full",
        project_id: $("#project-id").value.trim(),
        nl_input: intent,
        max_refinements: Number($("#max-refinements").value),
        n_samples: Number($("#n-samples").value),
        generation_strategy: $("#generation-strategy").value.trim(),
        retrosyn_engine: $("#retrosyn-engine").value.trim(),
        validation_policy: validationPolicy,
        teacher_policy: {
          teacher_source: "hypseek",
          teacher_version: $("#teacher-version").value.trim(),
          allow_synthetic: $("#allow-synthetic").value === "true",
          kd_weight: Number($("#kd-weight").value),
        },
        selection_policy: selectionPolicy,
      }),
    });
    await openRun(r.run_id || r.design_id, { live: true, intent });
    refreshHistory();
  } catch (e) {
    alert(e.message);
  } finally {
    $("#run").disabled = false;
  }
});

/* ---------------- run rendering ---------------- */

let activeRunId = null;
let activeRunGeneration = 0;
let activeRunRequestRevision = 0;
let activeRunAppliedRevision = 0;
let activeRunTerminal = false;
let pools = { novel: [], known: [], all: [] };
let activePool = "novel";

function setRunStatus(status) {
  const el = $("#run-status");
  el.className = "badge " + (status || "muted");
  el.textContent = status || "idle";
  updateRunControls(String(status || "").toLowerCase());
}

function updateRunControls(status) {
  const terminal = isTerminalRun(status);
  $("#pause-run").hidden = status !== "running";
  $("#resume-run").hidden = status !== "paused";
  $("#cancel-run").hidden = terminal || !activeRunId;
  $("#evidence-resume").hidden = status !== "awaiting_evidence";
}

function showWorkbench() {
  $("#workbench").hidden = false;
}

function formatRunMetadata(value) {
  if (Array.isArray(value)) {
    return value.map((item) => {
      if (typeof item === "string") return item;
      if (item && typeof item === "object") {
        return String(item.name || item.property || JSON.stringify(item));
      }
      return String(item ?? "");
    }).filter(Boolean).join(", ");
  }
  if (value && typeof value === "object") {
    return Object.keys(value).length ? JSON.stringify(value) : "";
  }
  return typeof value === "string" ? value.trim() : "";
}

function renderObjectives(obj) {
  const o = $("#objectives");
  if (!obj) {
    o.innerHTML = `<div class="muted">No active run</div>`;
    return;
  }
  const c = obj.constraints || {};
  const consRows = [];
  if (Array.isArray(c.molecular_weight)) consRows.push(["MW range", c.molecular_weight.map(v => v ?? "∞").join(" – ")]);
  if (Array.isArray(c.logp)) consRows.push(["logP range", c.logp.map(v => v ?? "∞").join(" – ")]);
  for (const k of ["hbd_max","hba_max","tpsa_max","qed_min","sa_max","rotatable_bonds_max"]) {
    if (c[k] !== undefined && c[k] !== null) consRows.push([k, c[k]]);
  }

  const targets = (obj.targets || []).map(t => `<span class="tag target">${t}</span>`).join("");
  const inds = (obj.indications || []).map(t => `<span class="tag indication">${t}</span>`).join("");
  const task = obj.task ? `<span class="tag task">${obj.task}</span>` : "";
  const prio = (obj.objectives_priority || []).map(t => `<span class="tag priority">${t}</span>`).join("");
  const inc = (c.must_include_smarts || []).map(s => `<span class="tag include" title="SMARTS">+ ${s}</span>`).join("");
  const exc = (c.must_exclude_smarts || []).map(s => `<span class="tag exclude" title="SMARTS">− ${s}</span>`).join("");
  const seeds = (obj.scaffold_hints || []).map(s => `<div class="constraint-row"><span class="k">seed</span><span>${s}</span></div>`).join("");
  const runObjectives = formatRunMetadata(obj.objectives);
  const runSummary = formatRunMetadata(obj.summary);
  const executionDevices = formatRunMetadata(obj.devices_used);

  o.innerHTML = `
    <div class="obj-section">
      <h4>Intent summary</h4>
      <div class="fg-dim small">${obj.intent_summary || "—"}</div>
    </div>
    ${runObjectives ? `<div class="obj-section"><h4>Run objectives</h4><div class="fg-dim small">${runObjectives}</div></div>` : ""}
    ${runSummary ? `<div class="obj-section"><h4>Run summary</h4><div class="fg-dim small">${runSummary}</div></div>` : ""}
    ${executionDevices ? `<div class="obj-section"><h4>Execution devices</h4><div class="fg-dim small">${executionDevices}</div></div>` : ""}
    ${task ? `<div class="obj-section"><h4>Task</h4>${task}</div>` : ""}
    ${targets ? `<div class="obj-section"><h4>Targets</h4>${targets}</div>` : ""}
    ${inds ? `<div class="obj-section"><h4>Therapeutic areas</h4>${inds}</div>` : ""}
    ${prio ? `<div class="obj-section"><h4>Priorities</h4>${prio}</div>` : ""}
    ${inc || exc ? `<div class="obj-section"><h4>Sub-structure rules</h4>${inc}${exc}</div>` : ""}
    ${consRows.length ? `<div class="obj-section"><h4>Numeric constraints</h4>${consRows.map(([k,v]) => `<div class="constraint-row"><span class="k">${k}</span><span>${v}</span></div>`).join("")}</div>` : ""}
    ${seeds ? `<div class="obj-section"><h4>Scaffold seeds</h4>${seeds}</div>` : ""}
    <div class="obj-section"><h4>Sample size</h4><span class="tag">${obj.n_samples ?? "—"}</span></div>
  `;
}

function clearReasoning() {
  $("#reasoning").innerHTML = "";
}

function appendStep(step, { final = false } = {}) {
  const wrap = $("#reasoning");
  // Mark all previous as done first
  $$(".step.active").forEach((el) => {
    el.classList.remove("active");
    el.classList.add("done");
  });
  const div = document.createElement("div");
  div.className = "step active";
  div.dataset.idx = step.step_index;
  div.dataset.stage = step.stage;

  let payloadHTML = "";
  if (step.payload && Object.keys(step.payload).length) {
    const summary = formatPayload(step);
    if (summary) {
      payloadHTML = `<div class="payload">${summary}</div>`;
    }
  }
  div.innerHTML = `
    <div class="stage">${step.stage}</div>
    <div class="title">${step.title}</div>
    ${step.detail ? `<div class="detail">${step.detail}</div>` : ""}
    ${payloadHTML}
  `;
  wrap.appendChild(div);
  // auto scroll
  wrap.scrollTop = wrap.scrollHeight;
  if (final) {
    div.classList.remove("active");
    div.classList.add("done");
  }
}

function formatPayload(step) {
  const p = step.payload || {};
  const lines = [];
  switch (step.stage) {
    case "nl_parse":
      if (p.tokens?.length) {
        lines.push(p.tokens.map(t => `<span class="tag">${t}</span>`).join(" "));
      }
      break;
    case "objectives":
      if (p.constraints_human?.length) {
        lines.push(`<pre>${p.constraints_human.join("\n")}</pre>`);
      }
      break;
    case "generation":
      if (p.examples?.length) {
        lines.push(`<pre>${p.examples.slice(0, 5).join("\n")}</pre>`);
      }
      break;
    case "scoring":
      if (p.devices?.length) {
        lines.push(p.devices.map(d => `<span class="tag">${d}</span>`).join(" ") +
                   ` <span class="muted">${p.elapsed_ms} ms</span>`);
      }
      break;
    case "constraint_filter":
      if (p.rejection_examples?.length) {
        const sample = p.rejection_examples.slice(0, 3).map(r =>
          `${r.smiles}\n   ↳ ${r.reasons.join("; ")}`).join("\n");
        lines.push(`<pre>${sample}</pre>`);
      }
      break;
    case "novelty":
      if (p.known_hits?.length) {
        const lines2 = p.known_hits.map(h => `${h.smiles}  →  ${h.name}`);
        lines.push(`<pre>${lines2.join("\n")}</pre>`);
      } else {
        lines.push(`<span class="tag">${p.n_novel ?? 0} novel</span> <span class="tag">${p.n_known ?? 0} known</span>`);
      }
      break;
    case "ranking":
      if (p.top?.length) {
        lines.push(`<pre>${p.top.slice(0, 5).join("\n")}</pre>`);
      }
      break;
    case "summary":
      lines.push(`<span class="tag good">${p.n_novel ?? 0} novel</span> ` +
                 `<span class="tag info">${p.n_known ?? 0} known</span>`);
      break;
  }
  return lines.join("");
}

function classifyAdmet(p) {
  const tags = [];
  if (p.lipinski_pass !== undefined) tags.push(p.lipinski_pass ? badge("Lipinski", "good") : badge("Lipinski", "bad"));
  return tags.join(" ");
}

function buildPropsBlock(props) {
  const fields = [
    ["MW", fmt(props.molecular_weight, 1)],
    ["logP", fmt(props.logp, 2)],
    ["QED", fmt(props.qed, 3)],
    ["SA",  fmt(props.sa_score, 2)],
    ["TPSA", fmt(props.tpsa, 1)],
    ["HBD/HBA", `${fmt(props.hbd)}/${fmt(props.hba)}`],
  ];
  return fields.map(([k, v]) => `<div><span class="k">${k}</span>${v}</div>`).join("");
}

function renderResult(r) {
  const props = r.properties || r;
  const cs = r.canonical_smiles || r.smiles;
  const isNovel = r.is_novel ?? !r.known_match;
  const card = document.createElement("div");
  card.className = "mol-card " + (isNovel ? "novel" : "known") + (r.pareto_optimal ? " pareto" : "");
  const safeSmi = cs.replace(/"/g, "&quot;");
  const knownLabel = r.known_match
    ? `<span class="badge info" title="DrugBank ${r.known_match.drugbank_id || ""}">${r.known_match.name}</span>`
    : `<span class="badge good">novel</span>`;
  const paretoBadge = r.pareto_optimal ? `<span class="badge gold">Pareto</span>` : "";
  const lipinski = (props.drug_likeness?.lipinski_pass)
    ? `<span class="badge good">Lipinski</span>`
    : `<span class="badge warn">Lipinski✗</span>`;
  const rankNum = (typeof r.rank === "number") ? `#${r.rank}` : "";

  card.innerHTML = `
    <div class="row1">
      <span>${rankNum} <span class="muted">${props.formula || ""}</span></span>
      <span class="badges">${paretoBadge}${knownLabel}</span>
    </div>
    <div class="canvas"><canvas></canvas></div>
    <div class="smiles" title="${safeSmi}">${cs}</div>
    <div class="props">${buildPropsBlock(props)}</div>
    <div class="row1"><span>${lipinski}</span><span class="muted small">${props.device || ""}</span></div>
  `;
  card.addEventListener("click", () => showDetail(r));
  // draw after attach so the canvas has dimensions
  requestAnimationFrame(() => {
    const c = card.querySelector("canvas");
    c.width = c.clientWidth || 220;
    c.height = c.clientHeight || 160;
    renderMolecule(c, cs);
  });
  return card;
}

function showResults(pool) {
  activePool = pool;
  $$(".result-tab").forEach((t) => t.classList.toggle("active", t.dataset.pool === pool));
  const wrap = $("#results");
  wrap.innerHTML = "";
  const items = pools[pool] || [];
  if (!items.length) {
    wrap.innerHTML = `<div class="muted">No ${pool} candidates yet.</div>`;
    return;
  }
  for (const r of items) {
    wrap.appendChild(renderResult(r));
  }
}

$$(".result-tab").forEach((t) =>
  t.addEventListener("click", () => showResults(t.dataset.pool))
);

function ingestResults(rows) {
  pools = { novel: [], known: [], all: [] };
  for (const r of rows) {
    pools.all.push(r);
    (r.is_novel ? pools.novel : pools.known).push(r);
  }
  $("#cnt-novel").textContent = pools.novel.length;
  $("#cnt-known").textContent = pools.known.length;
  $("#cnt-all").textContent = pools.all.length;
  $("#result-counts").textContent =
    `${pools.all.length} total · ${pools.novel.length} novel · ${pools.known.length} known`;
  showResults(activePool);
}

function objectValue(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function candidateSmiles(row) {
  if (!row || typeof row !== "object") return "";
  return String(row.canonical_smiles || row.smiles || "");
}

function titleCaseStage(stage) {
  const text = String(stage || "orchestrator").toLowerCase().replace(/_/g, " ");
  return text.replace(/\b\w/g, (c) => c.toUpperCase());
}

function appendQueue(map, key, index) {
  if (!key) return;
  const queue = map.get(key) || [];
  queue.push(index);
  map.set(key, queue);
}

function takeAvailable(queue, excluded) {
  while (queue?.length && excluded.has(queue[0])) queue.shift();
  return queue?.length ? queue.shift() : undefined;
}

function mergeCandidateValidation(candidates, validationRows) {
  const merged = candidates.map((candidate) => ({ ...objectValue(candidate) }));
  const byId = new Map();
  const byIdAndSmiles = new Map();
  const bySmiles = new Map();
  merged.forEach((candidate, index) => {
    const candidateId = String(candidate.candidate_id || "");
    const smiles = candidateSmiles(candidate);
    appendQueue(byId, candidateId, index);
    appendQueue(bySmiles, smiles, index);
    appendQueue(byIdAndSmiles, candidateId && smiles ? `${candidateId}\0${smiles}` : "", index);
  });

  const explicitMatches = new Map();
  const explicitlyMatched = new Set();
  validationRows.forEach((validationRow, validationIndex) => {
    const candidateId = String(validationRow.candidate_id || "");
    const smiles = candidateSmiles(validationRow);
    if (!candidateId || !smiles) return;
    const index = takeAvailable(
      byIdAndSmiles.get(`${candidateId}\0${smiles}`),
      explicitlyMatched,
    );
    if (index !== undefined) {
      explicitMatches.set(validationIndex, index);
      explicitlyMatched.add(index);
    }
  });
  validationRows.forEach((validationRow, validationIndex) => {
    if (explicitMatches.has(validationIndex)) return;
    const candidateId = String(validationRow.candidate_id || "");
    if (!candidateId) return;
    const index = takeAvailable(byId.get(candidateId), explicitlyMatched);
    if (index !== undefined) {
      explicitMatches.set(validationIndex, index);
      explicitlyMatched.add(index);
    }
  });

  const reserved = new Set(explicitMatches.values());
  const claimed = new Set();
  validationRows.forEach((validationRow, validationIndex) => {
    const candidateId = String(validationRow.candidate_id || "");
    const smiles = candidateSmiles(validationRow);
    let index = explicitMatches.get(validationIndex);
    if (candidateId && index === undefined) return;
    if (index === undefined && smiles) {
      index = takeAvailable(
        bySmiles.get(smiles),
        new Set([...reserved, ...claimed]),
      );
    }
    if (index === undefined) return;
    claimed.add(index);
    const candidate = merged[index];
    const combined = {
      ...candidate,
      ...objectValue(validationRow),
      properties: {
        ...objectValue(candidate.properties),
        ...objectValue(validationRow.properties),
      },
    };
    if (!candidateId && candidate.candidate_id) {
      combined.candidate_id = candidate.candidate_id;
    }
    merged[index] = combined;
  });
  return merged;
}

function orchestratorCandidateRows(state) {
  const candidates = Array.isArray(state.candidates) ? state.candidates : [];
  const validation = objectValue(state.validation);
  const validationRows = Array.isArray(validation.results) ? validation.results : [];
  const sourceRows = candidates.length
    ? mergeCandidateValidation(candidates, validationRows)
    : validationRows;
  return sourceRows.map((candidate, idx) => {
    const merged = objectValue(candidate);
    const properties = {
      ...objectValue(candidate.properties),
      ...objectValue(candidate),
    };
    const canonical = candidateSmiles(merged);
    if (!canonical) return null;
    const knownMatch = merged.known_match || properties.known_match || null;
    return {
      ...merged,
      canonical_smiles: canonical,
      smiles: canonical,
      rank: typeof merged.rank === "number" ? merged.rank : idx + 1,
      is_novel: merged.is_novel ?? !knownMatch,
      known_match: knownMatch,
      pareto_optimal: Boolean(merged.pareto_optimal),
      composite_score: merged.composite_score ?? properties.composite_score,
      properties,
    };
  }).filter(Boolean);
}

function renderOrchestratorRun(result, intent) {
  const state = objectValue(result.state);
  const request = objectValue(state.request);
  const rows = orchestratorCandidateRows(state);
  const runId = result.design_id || result.run_id || state.run_id || "orchestrator-run";
  $("#run-id").textContent = runId;
  showWorkbench();
  setRunStatus(result.status || state.status || "queued");
  renderObjectives({
    intent_summary: state.nl_input || request.nl_input || intent,
    objectives: result.objectives,
    summary: result.summary,
    devices_used: result.devices_used,
    task: state.workflow_scope || request.workflow_scope,
    targets: Array.isArray(request.targets) ? request.targets : [],
    constraints: objectValue(request.constraints),
    n_samples: request.n_samples ?? rows.length,
  });

  const history = Array.isArray(result.history)
    ? result.history
    : (Array.isArray(state.history) ? state.history : []);
  if (history.length && !$("#reasoning .step")) {
    history.forEach((stage, idx) => {
      appendStep({
        step_index: idx,
        stage: String(stage).toLowerCase(),
        title: titleCaseStage(stage),
        detail: result.status || "",
      }, { final: true });
    });
  }

  ingestResults(rows);
}

/* ---------------- run lifecycle ---------------- */

function claimActiveRun(runId) {
  activeRunGeneration += 1;
  activeRunId = runId;
  activeRunRequestRevision = 0;
  activeRunAppliedRevision = 0;
  activeRunTerminal = false;
  return activeRunGeneration;
}

function invalidateActiveRun() {
  activeRunGeneration += 1;
  activeRunId = null;
  activeRunRequestRevision = 0;
  activeRunAppliedRevision = 0;
  activeRunTerminal = false;
}

function ownsActiveRun(runId, generation) {
  return activeRunId === runId && activeRunGeneration === generation;
}

function isTerminalRun(status) {
  return ["completed", "rejected", "failed", "interrupted"].includes(status);
}

function beginActiveRunRequest(runId, generation) {
  if (!ownsActiveRun(runId, generation)) return null;
  activeRunRequestRevision += 1;
  return activeRunRequestRevision;
}

function applyActiveRunSnapshot(snapshot, intent, runId, generation, revision) {
  if (!ownsActiveRun(runId, generation) || activeRunTerminal) return false;
  const state = objectValue(snapshot.state);
  const status = String(snapshot.status || state.status || "").toLowerCase();
  const terminal = isTerminalRun(status);
  if (!terminal && revision < activeRunAppliedRevision) return false;
  activeRunAppliedRevision = Math.max(activeRunAppliedRevision, revision);
  renderOrchestratorRun(snapshot, intent);
  activeRunTerminal = terminal;
  return true;
}

async function pollOrchestratorRun(runId, intent, generation, initialAfterStep) {
  let afterStep = initialAfterStep;
  while (ownsActiveRun(runId, generation)) {
    await new Promise((resolve) => setTimeout(resolve, 1500));
    if (!ownsActiveRun(runId, generation)) return;
    const eventPage = await api(
      `/orchestrator/runs/${runId}/events?after_step=${afterStep}`,
    );
    for (const event of (eventPage.events || [])) {
      afterStep = Math.max(afterStep, Number(event.step_index));
      appendStep({
        step_index: event.step_index,
        stage: event.stage,
        title: titleCaseStage(event.stage),
        detail: "",
        payload: objectValue(event.payload),
      });
    }
    const revision = beginActiveRunRequest(runId, generation);
    if (revision === null) return;
    const snapshot = await api(`/orchestrator/runs/${runId}`);
    if (!applyActiveRunSnapshot(snapshot, intent, runId, generation, revision)) return;
    if (activeRunTerminal) {
      refreshHistory();
      return;
    }
  }
}

async function openRun(runId, { live = false, intent = "" } = {}) {
  const generation = claimActiveRun(runId);
  $("#run-id").textContent = runId;
  showWorkbench();
  setRunStatus("queued");
  clearReasoning();
  ingestResults([]);
  renderObjectives(null);

  const revision = beginActiveRunRequest(runId, generation);
  const snap = await api(`/orchestrator/runs/${runId}`);
  if (!applyActiveRunSnapshot(snap, intent, runId, generation, revision)) return;

  if (activeRunTerminal) {
    return;
  }
  if (!live) return;
  const state = objectValue(snap.state);
  const stateEvents = Array.isArray(state.events) ? state.events : [];
  pollOrchestratorRun(runId, intent, generation, stateEvents.length - 1).catch((error) => {
    if (ownsActiveRun(runId, generation)) console.error(error);
  });
}

async function runAction(action, payload = {}) {
  if (!activeRunId) return;
  await api(`/orchestrator/runs/${activeRunId}/${action}`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
  const generation = activeRunGeneration;
  const revision = beginActiveRunRequest(activeRunId, generation);
  const snapshot = await api(`/orchestrator/runs/${activeRunId}`);
  applyActiveRunSnapshot(snapshot, "", activeRunId, generation, revision);
}

$("#pause-run").addEventListener("click", () => {
  runAction("pause").catch((error) => alert(error.message));
});

$("#resume-run").addEventListener("click", () => {
  runAction("resume").catch((error) => alert(error.message));
});

$("#cancel-run").addEventListener("click", () => {
  runAction("cancel").catch((error) => alert(error.message));
});

$("#submit-evidence").addEventListener("click", async () => {
  let externalEvidence;
  try {
    externalEvidence = JSON.parse($("#external-evidence").value);
  } catch (error) {
    alert(`External evidence JSON is invalid: ${error.message}`);
    return;
  }
  if (!Array.isArray(externalEvidence)) {
    alert("External evidence must be a JSON array.");
    return;
  }
  try {
    await runAction("evidence/resume", { external_evidence: externalEvidence });
    $("#external-evidence").value = "";
  } catch (error) {
    alert(error.message);
  }
});

/* ---------------- detail drawer ---------------- */

const drawerEl = $("#drawer");
$("#drawer-close").addEventListener("click", () => (drawerEl.hidden = true));
$(".drawer-backdrop").addEventListener("click", () => (drawerEl.hidden = true));

function showDetail(r) {
  const props = r.properties || r;
  const admet = props.admet || {};
  const dl = props.drug_likeness || {};
  const known = r.known_match;
  const cs = r.canonical_smiles || r.smiles;

  const physRows = [
    ["Canonical SMILES", `<code>${cs}</code>`],
    ["Formula", props.formula || "—"],
    ["MW", fmt(props.molecular_weight, 3)],
    ["Exact mass", fmt(props.exact_mass, 4)],
    ["Heavy atoms", fmt(props.heavy_atoms)],
    ["logP", fmt(props.logp, 3)],
    ["TPSA", fmt(props.tpsa, 1)],
    ["HBD / HBA", `${fmt(props.hbd)} / ${fmt(props.hba)}`],
    ["Rotatable bonds", fmt(props.rotatable_bonds)],
    ["Aromatic rings / rings", `${fmt(props.aromatic_rings)} / ${fmt(props.rings)}`],
    ["Fraction sp³", fmt(props.fraction_csp3, 3)],
    ["QED", fmt(props.qed, 3)],
    ["SA score", fmt(props.sa_score, 2)],
    ["Lipinski violations", fmt(props.lipinski_violations)],
    ["HUMU ‖embedding‖", fmt(props.humu_embedding_norm, 4)],
    ["Device", props.device || "—"],
  ];

  const admetRows = Object.entries(admet).map(([k, v]) => {
    let cls = "";
    if (k === "herg_risk") cls = v === "high" ? "bad" : v === "medium" ? "warn" : "good";
    return `<tr><th>${k}</th><td class="${cls}">${typeof v === "number" ? fmt(v, 3) : (typeof v === "boolean" ? fmt(v) : String(v))}</td></tr>`;
  }).join("");

  const flags = [
    dl.lipinski_pass ? badge("Lipinski", "good") : badge("Lipinski✗", "bad"),
    dl.veber_pass ? badge("Veber", "good") : badge("Veber✗", "warn"),
    dl.egan_pass ? badge("Egan", "good") : badge("Egan✗", "warn"),
    r.pareto_optimal ? badge("Pareto-optimal", "gold") : "",
    r.is_novel ? badge("novel", "good") : badge(`known: ${known?.name || ""}`, "info"),
  ].filter(Boolean).join(" ");

  $("#drawer-content").innerHTML = `
    <h2>${known ? known.name : "Novel candidate"}</h2>
    <div class="muted small">rank #${r.rank ?? "—"} · composite=${fmt(r.composite_score, 3)}</div>
    <div class="canvas"><canvas></canvas></div>
    <div style="margin-bottom:10px;">${flags}</div>
    ${known ? `
      <div class="obj-section" style="margin-bottom:14px;">
        <h4 style="margin-bottom:4px;">Reference match</h4>
        <div class="constraint-row"><span class="k">DrugBank</span><span>${known.drugbank_id || "—"}</span></div>
        <div class="constraint-row"><span class="k">Indications</span><span>${known.indications || "—"}</span></div>
        <div class="constraint-row"><span class="k">Target</span><span>${known.target || "—"}</span></div>
      </div>
    ` : ""}
    <h3>Physicochemical descriptors</h3>
    <table>${physRows.map(([k, v]) => `<tr><th>${k}</th><td>${v}</td></tr>`).join("")}</table>
    <h3 style="margin-top:18px;">ADMET (HUMU + property head)</h3>
    <table>${admetRows}</table>
  `;
  const c = $("#drawer-content .canvas canvas");
  c.width = c.clientWidth || 480;
  c.height = c.clientHeight || 240;
  renderMolecule(c, cs);
  drawerEl.hidden = false;
}

/* ---------------- history ---------------- */

async function refreshHistory() {
  const list = $("#history");
  if (!bearerToken) {
    list.innerHTML = `<div class="muted small">Enter a Bearer Token to load runs.</div>`;
    return;
  }
  try {
    const r = await api("/orchestrator/runs?page_size=30");
    if (!r.runs.length) {
      list.innerHTML = `<div class="muted small">No runs yet.</div>`;
      return;
    }
    list.innerHTML = r.runs.map((run) => `
      <div class="history-item ${run.run_id === activeRunId ? "active" : ""}" data-run="${run.run_id}">
        <div class="h-title">${run.intent}</div>
        <div class="h-meta">
          <span class="badge ${run.status}">${run.status}</span>
          <span>${run.n_candidates ?? 0} mols · ${run.n_novel ?? 0} novel</span>
        </div>
      </div>
    `).join("");
    $$(".history-item").forEach((el) =>
      el.addEventListener("click", () => {
        const run = r.runs.find((item) => item.run_id === el.dataset.run);
        openRun(el.dataset.run, {
          live: !isTerminalRun(run.status),
          intent: run.intent || "",
        });
      })
    );
  } catch (e) { console.error(e); }
}
$("#refresh-history").addEventListener("click", refreshHistory);
refreshHistory();
setInterval(refreshHistory, 30000);
