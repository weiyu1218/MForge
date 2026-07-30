"""Project management endpoints."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from fastapi import APIRouter
from pydantic import BaseModel

from api_gateway.routers.design import (
    orchestrator_delete,
    orchestrator_get,
    orchestrator_post,
)

router = APIRouter()


class ProjectCreate(BaseModel):
    name: str
    description: str = ""


def _project_response(project: dict[str, Any]) -> dict[str, Any]:
    return {
        "project_id": project["project_id"],
        "name": project["name"],
        "description": project["description"],
        "status": "active",
        "created_at": project["created_at"],
        "designs": [],
    }


def _orchestrator_project_path(project_id: str) -> str:
    return f"/v1/orchestrator/projects/{quote(project_id, safe='')}"


@router.post("/")
async def create_project(request: ProjectCreate) -> dict[str, Any]:
    project, _ = await orchestrator_post(
        "/v1/orchestrator/projects",
        request.model_dump(),
    )
    return _project_response(project)


@router.get("/")
async def list_projects() -> dict[str, Any]:
    payload, _ = await orchestrator_get("/v1/orchestrator/projects")
    projects = [_project_response(project) for project in payload["projects"]]
    return {"projects": projects, "n_projects": len(projects)}


@router.get("/{project_id:path}")
async def get_project(project_id: str) -> dict[str, Any]:
    project, _ = await orchestrator_get(_orchestrator_project_path(project_id))
    return _project_response(project)


@router.delete("/{project_id:path}")
async def delete_project(project_id: str) -> dict[str, Any]:
    payload, _ = await orchestrator_delete(_orchestrator_project_path(project_id))
    return payload
