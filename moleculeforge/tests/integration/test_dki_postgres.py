"""Integration tests for DKI PostgreSQL layer."""

from __future__ import annotations

import os
from uuid import uuid4

import pytest

try:
    from mf_core.db.orm import (
        Base,
        MoleculeORM,
        OracleCallORM,
        ParetoFrontORM,
        RunORM,
    )
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    _HAS_SQLALCHEMY = True
except ImportError:
    _HAS_SQLALCHEMY = False

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest.fixture
async def session():
    if not _HAS_SQLALCHEMY:
        pytest.skip("SQLAlchemy is not installed")
    database_url = os.environ.get("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is required for PostgreSQL integration tests")
    engine = create_async_engine(database_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as db_session:
        yield db_session
        await db_session.rollback()
    await engine.dispose()


async def test_molecule_repo_crud(session) -> None:
    molecule = MoleculeORM(
        smiles="CCO",
        inchikey=f"IK{uuid4().hex[:25]}",
        mw=46.07,
        logp=-0.1,
    )
    session.add(molecule)
    await session.commit()

    result = await session.execute(
        select(MoleculeORM).where(MoleculeORM.inchikey == molecule.inchikey)
    )

    stored = result.scalar_one()
    assert stored.smiles == "CCO"
    assert stored.mw == 46.07


async def test_run_repo_create_and_update(session) -> None:
    run = RunORM(cig_id=f"cig-{uuid4()}", status="pending")
    session.add(run)
    await session.commit()

    run.status = "completed"
    run.hv_pareto = 0.42
    await session.commit()

    stored = await session.get(RunORM, run.id)
    assert stored.status == "completed"
    assert stored.hv_pareto == 0.42


async def test_oracle_call_repo_write(session) -> None:
    call = OracleCallORM(
        run_id=f"run-{uuid4()}",
        oracle_name="rdkit_l0",
        smiles="CCO",
        score=0.8,
        extra={"uncertainty": 0.05},
    )
    session.add(call)
    await session.commit()

    stored = await session.get(OracleCallORM, call.id)
    assert stored.oracle_name == "rdkit_l0"
    assert stored.extra["uncertainty"] == 0.05


async def test_pareto_front_repo(session) -> None:
    front = ParetoFrontORM(
        run_id=f"run-{uuid4()}",
        smiles="CCO",
        objectives={"qed": 0.7},
        hv_contribution=0.1,
    )
    session.add(front)
    await session.commit()

    stored = await session.get(ParetoFrontORM, front.id)
    assert stored.objectives["qed"] == 0.7
    assert stored.hv_contribution == 0.1
