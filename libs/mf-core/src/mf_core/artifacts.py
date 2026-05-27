"""Artifact and runtime dependency requirement checks."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse


@dataclass(frozen=True)
class RequirementStatus:
    """Resolved status for one required artifact or runtime dependency."""

    name: str
    configured: bool
    available: bool
    required: bool
    path: str | None
    source: str
    message: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "configured": self.configured,
            "available": self.available,
            "required": self.required,
            "path": self.path,
            "source": self.source,
            "message": self.message,
        }


@dataclass(frozen=True)
class ArtifactRequirement:
    """File or directory artifact required by a service."""

    name: str
    env_var: str
    kind: str = "file"
    required: bool = True
    path: str | None = None


@dataclass(frozen=True)
class ToolRequirement:
    """Executable required by a service."""

    name: str
    executable: str
    env_var: str | None = None
    required: bool = True


@dataclass(frozen=True)
class PythonPackageRequirement:
    """Importable Python package required by a service."""

    name: str
    module: str
    required: bool = True


@dataclass(frozen=True)
class ArtifactManifest:
    """Unified manifest of artifacts and runtime dependencies."""

    schema_version: str
    artifacts: list[ArtifactRequirement]
    tools: list[ToolRequirement]
    python_packages: list[PythonPackageRequirement]


def check_artifact(
    requirement: ArtifactRequirement,
    env: Mapping[str, str] | None = None,
) -> RequirementStatus:
    env = env or os.environ
    raw_path = requirement.path or env.get(requirement.env_var)
    if not raw_path:
        return RequirementStatus(
            name=requirement.name,
            configured=False,
            available=False,
            required=requirement.required,
            path=None,
            source=requirement.env_var,
            message=f"{requirement.env_var} is required for {requirement.name}",
        )

    if requirement.kind == "uri":
        parsed = urlparse(raw_path)
        available = bool(parsed.scheme and (parsed.netloc or parsed.path))
        resolved_path = raw_path
    else:
        artifact_path = Path(raw_path).expanduser()
        resolved_path = str(artifact_path)
    if requirement.kind == "directory":
        available = artifact_path.is_dir()
    elif requirement.kind == "path":
        available = artifact_path.exists()
    elif requirement.kind == "file":
        available = artifact_path.is_file()
    elif requirement.kind == "uri":
        pass
    else:
        raise ValueError(f"Unsupported artifact kind for {requirement.name}: {requirement.kind}")

    return RequirementStatus(
        name=requirement.name,
        configured=True,
        available=available,
        required=requirement.required,
        path=resolved_path,
        source=requirement.env_var if requirement.path is None else "manifest",
        message=(
            f"{requirement.name} is available"
            if available
            else f"{requirement.name} is not available: {resolved_path}"
        ),
    )


def check_tool(
    requirement: ToolRequirement,
    env: Mapping[str, str] | None = None,
) -> RequirementStatus:
    env = env or os.environ
    if requirement.env_var and env.get(requirement.env_var):
        tool_path = Path(env[requirement.env_var]).expanduser()
        available = tool_path.is_file() and os.access(tool_path, os.X_OK)
        return RequirementStatus(
            name=requirement.name,
            configured=True,
            available=available,
            required=requirement.required,
            path=str(tool_path),
            source=requirement.env_var,
            message=(
                f"{requirement.name} executable is available"
                if available
                else f"{requirement.env_var} is not an executable file: {tool_path}"
            ),
        )

    resolved = shutil.which(requirement.executable, path=env.get("PATH"))
    return RequirementStatus(
        name=requirement.name,
        configured=resolved is not None,
        available=resolved is not None,
        required=requirement.required,
        path=resolved,
        source="PATH",
        message=(
            f"{requirement.name} executable is available"
            if resolved
            else f"{requirement.executable} was not found on PATH"
        ),
    )


def check_python_package(requirement: PythonPackageRequirement) -> RequirementStatus:
    spec = importlib.util.find_spec(requirement.module)
    return RequirementStatus(
        name=requirement.name,
        configured=spec is not None,
        available=spec is not None,
        required=requirement.required,
        path=requirement.module if spec is not None else None,
        source="python",
        message=(
            f"{requirement.module} is importable"
            if spec is not None
            else f"{requirement.module} is not importable"
        ),
    )


def load_artifact_manifest(path: str | Path) -> ArtifactManifest:
    manifest_path = Path(path)
    with manifest_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return ArtifactManifest(
        schema_version=str(data.get("schema_version", "")),
        artifacts=[
            ArtifactRequirement(
                name=str(item["name"]),
                env_var=str(item["env_var"]),
                kind=str(item.get("kind", "file")),
                required=bool(item.get("required", True)),
                path=item.get("path"),
            )
            for item in data.get("artifacts", [])
        ],
        tools=[
            ToolRequirement(
                name=str(item["name"]),
                executable=str(item["executable"]),
                env_var=item.get("env_var"),
                required=bool(item.get("required", True)),
            )
            for item in data.get("tools", [])
        ],
        python_packages=[
            PythonPackageRequirement(
                name=str(item["name"]),
                module=str(item["module"]),
                required=bool(item.get("required", True)),
            )
            for item in data.get("python_packages", [])
        ],
    )


def check_manifest(manifest: ArtifactManifest) -> list[RequirementStatus]:
    return [
        *(check_artifact(requirement) for requirement in manifest.artifacts),
        *(check_tool(requirement) for requirement in manifest.tools),
        *(check_python_package(requirement) for requirement in manifest.python_packages),
    ]


def require_available(statuses: list[RequirementStatus]) -> None:
    missing = [status for status in statuses if status.required and not status.available]
    if missing:
        details = "; ".join(f"{status.name}: {status.message}" for status in missing)
        raise RuntimeError(f"Required artifacts or tools are unavailable: {details}")
