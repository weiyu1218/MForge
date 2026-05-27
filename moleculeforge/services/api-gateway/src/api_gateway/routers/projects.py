"""Project management endpoints."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

_projects: dict[str, dict] = {}


class ProjectCreate(BaseModel):
    name: str
    description: str = ""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@router.post("/")
async def create_project(request: ProjectCreate) -> dict[str, Any]:
    project_id = request.name
    _projects[project_id] = {
        "project_id": project_id,
        "name": request.name,
        "description": request.description,
        "status": "active",
        "created_at": _now(),
        "designs": [],
    }
    return _projects[project_id]


@router.get("/")
async def list_projects() -> dict[str, Any]:
    return {"projects": list(_projects.values()), "n_projects": len(_projects)}


@router.get("/{project_id}")
async def get_project(project_id: str) -> dict[str, Any]:
    proj = _projects.get(project_id)
    if proj is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return proj


@router.delete("/{project_id}")
async def delete_project(project_id: str) -> dict[str, Any]:
    removed = _projects.pop(project_id, None)
    if removed is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"deleted": True, "project_id": project_id}
