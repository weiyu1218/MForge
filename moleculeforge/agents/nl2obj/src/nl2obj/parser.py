"""Natural language → structured design objectives.

Pure-Python parser (regex + heuristics), no LLM call required.

Outputs an `Objectives` dict consumed by the design pipeline:

  {
    "intent_summary": str,
    "task": "lead_opt" | "scaffold_hop" | "hit_finding" | "de_novo",
    "targets": list[str],          # protein / disease names
    "scaffold_hints": list[str],   # SMILES seeds, chemotypes, named drugs
    "constraints": {
        "molecular_weight": [low, high] | None,
        "logp":             [low, high] | None,
        "hbd_max": int | None, "hba_max": int | None,
        "tpsa_max": float | None,
        "rotatable_bonds_max": int | None,
        "qed_min": float | None,
        "sa_max": float | None,
        "must_include_smarts": list[str],
        "must_exclude_smarts": list[str],
    },
    "objectives_priority": list[str],  # e.g. ["qed","sa","logp"]
    "n_samples": int,
    "tokens": list[str],          # tokens recognised — used for the trace
  }
"""
from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Token banks
# ---------------------------------------------------------------------------

# Known target keywords → standardised name
TARGETS = {
    r"\bkras\s*g12c\b": "KRAS G12C",
    r"\bkras\b": "KRAS",
    r"\begfr\b": "EGFR",
    r"\bbraf\b": "BRAF",
    r"\bpik3ca\b": "PIK3CA",
    r"\balk\b": "ALK",
    r"\bros1\b": "ROS1",
    r"\bbcr[\s-]?abl\b": "BCR-ABL",
    r"\bvegfr\b": "VEGFR",
    r"\bcox[\s-]?2\b": "COX-2",
    r"\bcox[\s-]?1\b": "COX-1",
    r"\bjak\s*[123]\b": "JAK kinase",
    r"\bbtk\b": "BTK",
    r"\bparp\b": "PARP",
    r"\bcdk\s*[0-9]+\b": "CDK kinase",
    r"\bp53\b": "p53",
    r"\bmtor\b": "mTOR",
    r"\bpd[-\s]?l1\b": "PD-L1",
    r"\bser\w*tonin\b|\b5[-\s]?ht\b": "Serotonin receptor",
    r"\bdopamine\b|\bd[12]\s*receptor\b": "Dopamine receptor",
    r"\bgpcr\b": "GPCR",
    r"\bsars[-\s]?cov[-\s]?2\b|\bcovid\b": "SARS-CoV-2",
    r"\bm[\w]*pro\b": "main protease",
    r"\bcox\b": "COX",
    r"\bgaba\b": "GABA receptor",
    r"\bhdac\b": "HDAC",
    r"\bdhfr\b": "DHFR",
    r"\bache\b|\bacetylcholinesterase\b": "AChE",
    r"\bnav\s*1?\.?[0-9]\b": "Sodium channel",
}

# Indication / disease keywords → label
INDICATIONS = {
    r"\bcancer\b|\boncolog\w+\b|\btumou?r\b|\bantineoplas\w+\b|抗肿瘤|肿瘤": "oncology",
    r"\bdiabet\w+\b|糖尿病": "diabetes",
    r"\bhypertens\w+\b|高血压": "hypertension",
    r"\bdepress\w+\b|抑郁": "depression",
    r"\binflammat\w+\b|\barthrit\w+\b|抗炎|关节炎": "inflammation",
    r"\bantibacter\w+\b|\binfection\b|感染|抗菌": "infection",
    r"\balzheimer\b|\bdementia\b|阿尔兹海默|痴呆": "Alzheimer's",
    r"\bparkinson\b|帕金森": "Parkinson's",
    r"\bantiviral\b|\bvir\w+\b|抗病毒|病毒": "antiviral",
    r"\bpain\b|\banalgesi\w+\b|镇痛|止痛": "pain",
    r"\bcardiovascular\b|\bheart\b|心血管": "cardiovascular",
}

# Task class keywords
TASKS = {
    r"\blead\s*opt\w*\b|\boptimi[sz]\w+\b": "lead_opt",
    r"\bscaffold\s*hop\b|\bbioisoster\w+\b": "scaffold_hop",
    r"\bhit\s*find\w+\b|\bscreen\w+\b": "hit_finding",
    r"\bde\s*novo\b|\bgenerate\s*new\b|\bnovel\b": "de_novo",
}

# Named seed scaffolds → SMILES
SEED_SMILES_BY_NAME = {
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "ibuprofen": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "acetaminophen": "CC(=O)Nc1ccc(O)cc1",
    "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "imatinib": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
    "gefitinib": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    "erlotinib": "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",
    "celecoxib": "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
    "metformin": "CN(C)C(=N)NC(N)=N",
    "atorvastatin": (
        "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)"
        "n1CC[C@@H](O)C[C@@H](O)CC(=O)O"
    ),
    "fluoxetine": "CNCCC(c1ccc(C(F)(F)F)cc1)Oc1ccccc1",
    "sildenafil": "CCCc1nn(C)c2c1NC(=NC2=O)c1cc(S(=O)(=O)N2CCN(C)CC2)ccc1OCC",
}

# Functional-group SMARTS hints
SMARTS_HINTS = {
    "michael acceptor": "C=CC(=O)",
    "warhead": "C=CC(=O)",
    "amide": "C(=O)N",
    "酰胺": "C(=O)N",
    "碳酰胺": "C(=O)N",
    "sulfonamide": "S(=O)(=O)N",
    "磺酰胺": "S(=O)(=O)N",
    "carboxylic acid": "C(=O)O",
    "carboxyl": "C(=O)O",
    "羧基": "C(=O)O",
    "羧酸": "C(=O)O",
    "primary amine": "[NX3;H2]",
    "secondary amine": "[NX3;H1]",
    "halogen": "[F,Cl,Br,I]",
    "卤素": "[F,Cl,Br,I]",
    "fluorine": "F",
    "氟": "F",
    "trifluoromethyl": "C(F)(F)F",
    "三氟甲基": "C(F)(F)F",
    "pyridine": "c1ccncc1",
    "吡啶": "c1ccncc1",
    "benzene": "c1ccccc1",
    "苯环": "c1ccccc1",
    "ester": "C(=O)O[C,c]",
    "酯": "C(=O)O[C,c]",
    "ether": "[OD2]([#6])[#6]",
    "醚": "[OD2]([#6])[#6]",
    "nitrile": "C#N",
    "氰基": "C#N",
    "hydroxyl": "[OH]",
    "羟基": "[OH]",
}

# Numeric ranges — patterns are applied in order
NUMERIC_PATTERNS = [
    # Range "MW between 200 and 500" or "MW 200-500"
    (
        re.compile(
            r"\b(?:mw|molecular\s*weight)\b[^\d]{0,30}"
            r"(\d{2,4})\s*(?:-|to|和|至)\s*(\d{2,4})",
            re.I,
        ),
        "molecular_weight", "range",
    ),
    (
        re.compile(r"\b(?:mw|molecular\s*weight)\b[^\d]{0,30}<\s*(\d{2,4})", re.I),
        "molecular_weight", "max",
    ),
    (
        re.compile(r"\b(?:mw|molecular\s*weight)\b[^\d]{0,30}>\s*(\d{2,4})", re.I),
        "molecular_weight", "min",
    ),
    (
        re.compile(
            r"\b(?:logp|clogp)\b[^\d-]{0,20}"
            r"(-?\d+(?:\.\d+)?)\s*(?:-|to|至)\s*(-?\d+(?:\.\d+)?)",
            re.I,
        ),
        "logp", "range",
    ),
    (
        re.compile(r"\b(?:logp|clogp)\b[^\d-]{0,20}<\s*(-?\d+(?:\.\d+)?)", re.I),
        "logp", "max",
    ),
    (
        re.compile(r"\b(?:logp|clogp)\b[^\d-]{0,20}>\s*(-?\d+(?:\.\d+)?)", re.I),
        "logp", "min",
    ),
    (
        re.compile(r"\bhbd\b[^\d]{0,15}(?:<=|≤|<)\s*(\d+)", re.I),
        "hbd_max", "scalar",
    ),
    (
        re.compile(r"\bhba\b[^\d]{0,15}(?:<=|≤|<)\s*(\d+)", re.I),
        "hba_max", "scalar",
    ),
    (
        re.compile(r"\b(?:tpsa)\b[^\d]{0,15}(?:<=|≤|<)\s*(\d+(?:\.\d+)?)", re.I),
        "tpsa_max", "scalar",
    ),
    (
        re.compile(r"\bqed\b[^\d]{0,15}(?:>=|≥|>)\s*(\d+(?:\.\d+)?)", re.I),
        "qed_min", "scalar",
    ),
    (
        re.compile(r"\bsa\s*(?:score)?\b[^\d]{0,15}(?:<=|≤|<)\s*(\d+(?:\.\d+)?)", re.I),
        "sa_max", "scalar",
    ),
    (
        re.compile(r"\brotatable\s*bonds?\b[^\d]{0,15}(?:<=|≤|<)\s*(\d+)", re.I),
        "rotatable_bonds_max", "scalar",
    ),
    (
        re.compile(
            r"(?:^|[\s,，;；])(\d+)\s+(?:[\w\-]+\s+){0,5}"
            r"(?:samples?|candidates?|molecules?|inhibitors?|compounds?|drugs?|"
            r"agonists?|antagonists?|analogu?es?|个分子|个候选|个|分子|候选)",
            re.I,
        ),
        "n_samples", "scalar",
    ),
    (
        re.compile(r"分子量[^\d]{0,10}(\d{2,4})\s*(?:-|到|至)\s*(\d{2,4})", re.I),
        "molecular_weight", "range",
    ),
    (
        re.compile(r"分子量[^\d]{0,10}<\s*(\d{2,4})", re.I),
        "molecular_weight", "max",
    ),
]

# Objective-priority phrases
PRIORITY_KEYWORDS = [
    (r"\bdrug[-\s]?like\w*\b|\bqed\b", "qed"),
    (r"\bsynthe(?:s|t)\w*\b|\bsa\s*score\b|\beasy\s*to\s*make\b", "sa"),
    (r"\bbioavailab\w+\b|\boral\b", "bioavailability"),
    (r"\blogp\b|\blipophil\w+\b", "logp"),
    (r"\bsolubl?\w*\b", "solubility"),
    (r"\bpotenc\w+\b|\baffinity\b|\bic50\b|\bki\b|\bpkd\b", "potency"),
    (r"\bselectiv\w+\b", "selectivity"),
    (r"\bsafet\w+\b|\bherg\b", "safety"),
]

# ---------------------------------------------------------------------------


def _apply_numeric(text: str, constraints: dict, tokens: list[str]) -> None:
    for pattern, key, kind in NUMERIC_PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        if kind == "range":
            lo, hi = float(m.group(1)), float(m.group(2))
            if lo > hi:
                lo, hi = hi, lo
            constraints[key] = [lo, hi]
            tokens.append(f"{key}: {lo}–{hi}")
        elif kind == "max":
            v = float(m.group(1))
            existing = constraints.get(key)
            if existing and isinstance(existing, list):
                existing[1] = v
            else:
                constraints[key] = [None, v]
            tokens.append(f"{key} ≤ {v}")
        elif kind == "min":
            v = float(m.group(1))
            existing = constraints.get(key)
            if existing and isinstance(existing, list):
                existing[0] = v
            else:
                constraints[key] = [v, None]
            tokens.append(f"{key} ≥ {v}")
        elif kind == "scalar":
            constraints[key] = (
                int(m.group(1)) if "max" in key or "samples" in key else float(m.group(1))
            )
            tokens.append(f"{key}={constraints[key]}")


def _detect_targets(text: str) -> tuple[list[str], list[str]]:
    found: list[str] = []
    matches: list[str] = []
    for pat, label in TARGETS.items():
        if re.search(pat, text, flags=re.I):
            if label not in found:
                found.append(label)
                matches.append(f"target:{label}")
    return found, matches


def _detect_indications(text: str) -> list[str]:
    out: list[str] = []
    for pat, label in INDICATIONS.items():
        if re.search(pat, text, flags=re.I):
            if label not in out:
                out.append(label)
    return out


def _detect_task(text: str) -> str:
    for pat, label in TASKS.items():
        if re.search(pat, text, flags=re.I):
            return label
    if re.search(r"\bsimilar\s+to\b|\blike\s+\w+\b", text, flags=re.I):
        return "lead_opt"
    return "de_novo"


def _detect_priority(text: str) -> list[str]:
    out: list[str] = []
    for pat, label in PRIORITY_KEYWORDS:
        if re.search(pat, text, flags=re.I) and label not in out:
            out.append(label)
    return out or ["qed", "sa"]


def _detect_seed_smiles(text: str) -> tuple[list[str], list[str]]:
    """Return (seed_smiles, hint_strings)."""
    smiles: list[str] = []
    hints: list[str] = []
    for name, smi in SEED_SMILES_BY_NAME.items():
        if re.search(rf"\b{re.escape(name)}\b", text, flags=re.I):
            smiles.append(smi)
            hints.append(f"seed:{name}")
    try:
        from rdkit import Chem  # local import keeps module light
    except ImportError:
        return smiles, hints
    # Inline SMILES: a token that looks like a SMILES with at least one ring or [X]
    for tok in re.findall(r"[A-Za-z0-9@/\\\[\]\(\)=#+\-\.]{6,}", text):
        if any(ch in tok for ch in "()[]=#") and any(ch.isalpha() for ch in tok):
            try:
                mol = Chem.MolFromSmiles(tok)
            except (RuntimeError, ValueError):
                continue
            if mol is not None and tok not in smiles:
                smiles.append(tok)
                hints.append(f"inline_smiles:{tok}")
    return smiles, hints


def _detect_smarts(text: str) -> tuple[list[str], list[str]]:
    must_include: list[str] = []
    must_exclude: list[str] = []
    text_l = text.lower()
    EXCLUDE_WORDS = ("avoid", "exclude", "without", "no ", "不含", "禁止", "排除")
    for kw, smarts in SMARTS_HINTS.items():
        kw_l = kw.lower()
        # Pure-ascii keywords use word boundaries; anything else uses substring.
        if kw_l.isascii():
            pat = re.compile(rf"\b{re.escape(kw_l)}\b")
            present = pat.search(text_l) is not None
        else:
            present = kw_l in text_l
        if not present:
            continue
        # Detect exclusion intent in nearby window
        idx = text_l.find(kw_l)
        window = text_l[max(0, idx - 30): idx]
        is_exclude = any(w in window for w in EXCLUDE_WORDS)
        if is_exclude and smarts not in must_exclude:
            must_exclude.append(smarts)
        elif not is_exclude and smarts not in must_include and smarts not in must_exclude:
            must_include.append(smarts)
    return must_include, must_exclude


def _detect_activity(text: str) -> dict[str, Any]:
    activity = {"type": "", "direction": "", "target_value": None}
    ic50_match = re.search(
        r"\bIC50\b\s*(?:below|under|<\s*)\s*(\d+(?:\.\d+)?)\s*(?:nM|nmol)",
        text,
        re.IGNORECASE,
    )
    if ic50_match:
        activity["type"] = "IC50"
        activity["direction"] = "minimize"
        activity["target_value"] = float(ic50_match.group(1))
    return activity


def _detect_admet(text: str) -> dict[str, float | None]:
    admet = {
        "oral_bioavailability_min": None,
        "cyp3a4_ic50_min": None,
    }
    if re.search(r"oral\s+bioavail", text, re.IGNORECASE):
        admet["oral_bioavailability_min"] = 0.3
    if re.search(r"CYP3A4", text, re.IGNORECASE):
        admet["cyp3a4_ic50_min"] = 10.0
    return admet


def _detect_synthetic_constraints(text: str) -> dict[str, int]:
    constraints = {"max_synthetic_steps": 10}
    steps_match = re.search(r"(\d+)\s*steps?", text, re.IGNORECASE)
    if steps_match:
        constraints["max_synthetic_steps"] = int(steps_match.group(1))
    return constraints


def _detect_binding_mode(text: str) -> str:
    if re.search(r"irreversible", text, re.IGNORECASE):
        return "covalent_irreversible"
    if re.search(r"covalent", text, re.IGNORECASE):
        return "covalent_reversible"
    return "competitive"


def _target_entity_name(label: str) -> str:
    if label.upper().startswith("KRAS"):
        return "KRAS"
    return label


def _target_details(targets: list[str], text: str) -> list[dict[str, str]]:
    binding_mode = _detect_binding_mode(text)
    return [
        {
            "name": _target_entity_name(target),
            "label": target,
            "binding_mode": binding_mode,
        }
        for target in targets
    ]


def parse(intent_text: str) -> dict[str, Any]:
    """Parse free-form NL into a structured objectives dict.

    Always returns a usable result; degrades gracefully if nothing was
    recognised (defaults: de novo, n=24, prioritise QED + SA).
    """
    text = (intent_text or "").strip()
    tokens: list[str] = []

    targets, target_tokens = _detect_targets(text)
    tokens.extend(target_tokens)
    indications = _detect_indications(text)
    for ind in indications:
        tokens.append(f"indication:{ind}")

    task = _detect_task(text)
    tokens.append(f"task:{task}")

    seed_smiles, seed_tokens = _detect_seed_smiles(text)
    tokens.extend(seed_tokens)

    must_include, must_exclude = _detect_smarts(text)
    for s in must_include:
        tokens.append(f"include:{s}")
    for s in must_exclude:
        tokens.append(f"exclude:{s}")

    priorities = _detect_priority(text)
    for p in priorities:
        tokens.append(f"priority:{p}")

    constraints: dict[str, Any] = {
        "must_include_smarts": must_include,
        "must_exclude_smarts": must_exclude,
    }
    _apply_numeric(text, constraints, tokens)
    if re.search(r"\bLipinski\b", text, re.IGNORECASE):
        constraints["lipinski_strict"] = True
        tokens.append("constraint:lipinski")

    n_samples = int(constraints.pop("n_samples", 24) or 24)
    n_samples = max(4, min(n_samples, 256))
    activity = _detect_activity(text)
    admet_constraints = _detect_admet(text)
    synthetic_constraints = _detect_synthetic_constraints(text)
    targets_structured = _target_details(targets, text)

    summary_bits: list[str] = []
    if targets:
        summary_bits.append("targets: " + ", ".join(targets))
    if indications:
        summary_bits.append("areas: " + ", ".join(indications))
    summary_bits.append(f"task: {task}")
    if seed_smiles:
        summary_bits.append(f"{len(seed_smiles)} seed scaffold(s)")
    if must_include:
        summary_bits.append(f"{len(must_include)} required group(s)")
    if must_exclude:
        summary_bits.append(f"{len(must_exclude)} forbidden group(s)")
    summary = "; ".join(summary_bits) or "open-ended de-novo design"

    return {
        "intent_summary": summary,
        "task": task,
        "targets": targets,
        "target_details": targets_structured,
        "indications": indications,
        "scaffold_hints": seed_smiles,
        "constraints": constraints,
        "activity": activity,
        "admet_constraints": admet_constraints,
        "ip_constraints": {},
        "synthetic_constraints": synthetic_constraints,
        "objectives_priority": priorities,
        "n_samples": n_samples,
        "tokens": tokens,
    }
