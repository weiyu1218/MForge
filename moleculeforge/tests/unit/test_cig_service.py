"""Unit tests for the CIG compiler service boundary."""

from __future__ import annotations

import asyncio

import pytest


def _run(coro):
    return asyncio.run(coro)


class _Request:
    nl_query = "Design a KRAS G12C inhibitor"
    project_id = "test-project"


@pytest.mark.asyncio
async def test_compile_service_fails_fast_without_production_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cig_compiler_svc.main import CIGCompilerServicer

    monkeypatch.delenv("CIG_SEMANTIC_PARSER_URI", raising=False)
    service = CIGCompilerServicer()

    with pytest.raises(RuntimeError, match="CIG_SEMANTIC_PARSER_URI"):
        await service.Compile(_Request(), None)


def test_compile_service_uses_injected_compiler(monkeypatch: pytest.MonkeyPatch) -> None:
    from cig_compiler_svc.main import CIGCompilerServicer

    async def reject_uniprot_query(*_args, **_kwargs) -> list[dict]:
        raise AssertionError("injected local demo compiler must stay offline")

    monkeypatch.setattr(
        "cig_compiler_svc.domain.tools.uniprot_tool.query_uniprot_entry",
        reject_uniprot_query,
    )

    class Compiler:
        async def compile(self, nl_query: str, seed=None):
            from cig_compiler_svc.domain.compiler import CIGCompiler

            assert nl_query == _Request.nl_query
            compiler = CIGCompiler(
                mode="local_demo",
                encoding_mode="hash",
                enable_grounding=False,
            )
            return await compiler.compile(nl_query, seed=seed)

    response = _run(CIGCompilerServicer(compiler=Compiler()).Compile(_Request(), None))

    assert response.cig["source_user_input"] == _Request.nl_query
    assert response.parse_confidence is None
