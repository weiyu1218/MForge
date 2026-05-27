#!/usr/bin/env python
"""Pre-commit secrets scanner.

Scans staged files for potential secrets: API keys, tokens, passwords,
private keys, and other sensitive patterns before they reach the repository.

Triggered by: pre-commit hook `secrets-scan`
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Patterns that indicate potential secrets (case-insensitive)
_SECRET_PATTERNS: list[tuple[str, str]] = [
    # (regex, description)
    (r"(?:api[_-]?key|apikey)\s*[:=]\s*[\"'`][A-Za-z0-9_\-]{20,}[\"'`]", "API Key"),
    (r"(?:secret|secret_key)\s*[:=]\s*[\"'`][A-Za-z0-9_\-+/]{20,}[\"'`]", "Secret Key"),
    (r"(?:password|passwd)\s*[:=]\s*[\"'`](?!.*(?:placeholder|changeme|example|test|dummy|password))[^\"'`\s]{8,}[\"'`]", "Hardcoded Password"),
    (r"(?:token|auth_token|access_token)\s*[:=]\s*[\"'`][A-Za-z0-9_\-.]{20,}[\"'`]", "Access Token"),
    (r"(?:private[_-]?key|privkey)\s*[:=]\s*[\"'`]", "Private Key"),
    (r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----", "PEM Private Key"),
    (r"sk-[A-Za-z0-9]{32,}", "OpenAI/Anthropic API Key"),
    (r"AKIA[0-9A-Z]{16}", "AWS Access Key ID"),
    (r"(?:mongodb|postgresql|mysql|redis)://[^/\s]+:[^/\s]+@", "Database Connection String with Credentials"),
    (r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}", "GitHub Personal Access Token"),
    (r"ya29\.[0-9A-Za-z\-_]+", "Google OAuth Token"),
]

# Files to skip (generated, vendored, test fixtures with dummy keys)
_SKIP_PATTERNS: list[str] = [
    "**/proto_gen/**",
    "**/.venv/**",
    "**/node_modules/**",
    "**/uv.lock",
    "**/poetry.lock",
    "**/*.ipynb",
    "**/test_fixtures/**",
]


def _should_skip(file_path: Path) -> bool:
    """Check if file should be skipped."""
    from fnmatch import fnmatch

    path_str = str(file_path).replace("\\", "/")
    for pattern in _SKIP_PATTERNS:
        if fnmatch(path_str, pattern):
            return True
    return False


def scan_file(file_path: Path) -> list[str]:
    """Scan a single file for secrets. Returns list of issue descriptions."""
    if _should_skip(file_path):
        return []

    issues: list[str] = []
    try:
        content = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, PermissionError):
        return []

    for pattern, description in _SECRET_PATTERNS:
        matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            line_num = content[: match.start()].count("\n") + 1
            issues.append(
                f"  {file_path}:{line_num} — Potential {description} detected"
            )

    return issues


def main() -> int:
    """Entry point. Scans files passed as arguments (from pre-commit)."""
    files = [Path(f) for f in sys.argv[1:]] if len(sys.argv) > 1 else []

    if not files:
        # No files passed: scan all tracked Python/YAML/TOML files
        files = list(Path(".").rglob("*.py")) + list(Path(".").rglob("*.yaml")) + list(Path(".").rglob("*.yml")) + list(Path(".").rglob("*.toml"))

    all_issues: list[str] = []
    for file_path in files:
        if file_path.is_file():
            issues = scan_file(file_path)
            all_issues.extend(issues)

    if all_issues:
        print("🚨 Secret scan found potential issues:")
        for issue in all_issues:
            print(issue)
        print(f"\nTotal: {len(all_issues)} potential secret(s) detected.")
        print("Remove these values or replace with environment variables before committing.")
        return 1

    print("✓ Secret scan passed — no secrets detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
