"""DeepSeek-backed CIG semantic parser command.

stdin: {"text": "..."}
stdout: extracted intent JSON object consumed by CIGCompiler.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any


_SYSTEM_PROMPT = """You are MoleculeForge's CIG semantic parser.
Return only a JSON object. Do not include markdown.
The object must describe a molecular design intent for downstream CIG building.
Allowed top-level keys:
- properties: list of objects with name, direction, priority
- targets: list of objects with name
- constraints: object such as max_mw, min_mw, lipinski_strict
- activity: object with type, direction, target_value
- admet_constraints: object such as oral_bioavailability_min, cyp3a4_ic50_min
- synthetic_constraints: object such as max_synthetic_steps
Use property names from: qed, sa_score, logp, solubility, binding_affinity, selectivity, safety.
"""


def parse_semantic_text(text: str) -> dict[str, Any]:
    if not text.strip():
        raise RuntimeError("semantic parser input text is required")
    payload = _deepseek_json(
        _SYSTEM_PROMPT,
        f"Parse this molecular design intent:\n{text.strip()}",
        timeout_env="CIG_SEMANTIC_PARSER_TIMEOUT_SECONDS",
    )
    _validate_extracted_intent(payload)
    return payload


def _deepseek_json(system_prompt: str, user_prompt: str, *, timeout_env: str) -> dict[str, Any]:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
    model = (
        os.environ.get("CIG_DEEPSEEK_MODEL")
        or os.environ.get("DEEPSEEK_MODEL")
        or "deepseek-v4-flash"
    ).strip()
    timeout = float(os.environ.get(timeout_env, "60"))
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": float(os.environ.get("CIG_DEEPSEEK_TEMPERATURE", "0")),
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"DeepSeek request failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("DeepSeek response was not valid JSON") from exc
    content = _message_content(response_payload)
    parsed = _json_object_from_content(content)
    return parsed


def _message_content(response_payload: dict[str, Any]) -> str:
    try:
        content = response_payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("DeepSeek response missing choices[0].message.content") from exc
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("DeepSeek response content is empty")
    return content


def _json_object_from_content(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
            stripped = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise RuntimeError("DeepSeek content was not a JSON object") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("DeepSeek content must be a JSON object")
    return payload


def _validate_extracted_intent(payload: dict[str, Any]) -> None:
    if "properties" in payload and not isinstance(payload["properties"], list):
        raise RuntimeError("semantic parser properties must be a list")
    if "targets" in payload and not isinstance(payload["targets"], list):
        raise RuntimeError("semantic parser targets must be a list")
    if "constraints" in payload and not isinstance(payload["constraints"], dict):
        raise RuntimeError("semantic parser constraints must be an object")
    if not payload.get("properties") and not payload.get("targets"):
        raise RuntimeError("semantic parser output requires properties or targets")


def main() -> int:
    try:
        request = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError("stdin must be a JSON object") from exc
    if not isinstance(request, dict):
        raise RuntimeError("stdin must be a JSON object")
    text = request.get("text")
    if not isinstance(text, str):
        raise RuntimeError("stdin requires string field: text")
    print(json.dumps(parse_semantic_text(text), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
