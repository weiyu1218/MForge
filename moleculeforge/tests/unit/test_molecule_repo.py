"""Unit tests for MoleculeRepository (Mock SQLAlchemy session)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest


@pytest.fixture
def mock_session():
    session = AsyncMock()
    session.execute = AsyncMock()
    session.flush = AsyncMock()
    session.get = AsyncMock()
    session.add = MagicMock()
    return session


@pytest.mark.unit
@pytest.mark.asyncio
async def test_upsert_molecule(mock_session) -> None:
    from mf_core.db.repositories.molecule_repo import MoleculeRepository

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result

    repo = MoleculeRepository(mock_session)
    mol = await repo.upsert(
        smiles="CCO",
        inchikey="LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        mw=46.07,
        logp=-0.31,
    )

    assert mol is not None
    assert mol.smiles == "CCO"
    assert mol.inchikey == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
    mock_session.execute.assert_called_once()
    statement = mock_session.execute.call_args.args[0]
    assert statement is not None
    assert getattr(statement, "whereclause", None) is not None
    mock_session.add.assert_called_once_with(mol)
    mock_session.flush.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_upsert_updates_existing_molecule(mock_session) -> None:
    from mf_core.db.repositories.molecule_repo import MoleculeRepository
    from mf_core.db.orm import MoleculeORM

    existing = MoleculeORM(
        id=uuid4(),
        smiles="CC",
        inchikey="LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        mw=30.0,
        logp=0.0,
    )
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = existing
    mock_session.execute.return_value = mock_result

    repo = MoleculeRepository(mock_session)
    mol = await repo.upsert(
        smiles="CCO",
        inchikey="LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        mw=46.07,
        logp=-0.31,
    )

    assert mol is existing
    assert existing.smiles == "CCO"
    assert existing.mw == 46.07
    assert existing.logp == -0.31
    mock_session.add.assert_not_called()
    mock_session.flush.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_by_inchikey(mock_session) -> None:
    from mf_core.db.repositories.molecule_repo import MoleculeRepository
    from mf_core.db.orm import MoleculeORM

    mock_mol = MoleculeORM(
        id=uuid4(),
        smiles="CCO",
        inchikey="LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
    )
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_mol
    mock_session.execute.return_value = mock_result

    repo = MoleculeRepository(mock_session)
    mol = await repo.get_by_inchikey("LFQSCWFLJHTTHZ-UHFFFAOYSA-N")

    assert mol is not None
    assert mol.smiles == "CCO"
    statement = mock_session.execute.call_args.args[0]
    assert statement is not None
    assert getattr(statement, "whereclause", None) is not None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_batch_upsert_empty(mock_session) -> None:
    from mf_core.db.repositories.molecule_repo import MoleculeRepository

    repo = MoleculeRepository(mock_session)
    count = await repo.batch_upsert([])
    assert count == 0
    mock_session.execute.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_by_id(mock_session) -> None:
    from mf_core.db.repositories.molecule_repo import MoleculeRepository
    from mf_core.db.orm import MoleculeORM

    mol_id = uuid4()
    mock_mol = MoleculeORM(id=mol_id, smiles="CCO")
    mock_session.get.return_value = mock_mol

    repo = MoleculeRepository(mock_session)
    mol = await repo.get_by_id(mol_id)

    assert mol is not None
    assert mol.id == mol_id
    mock_session.get.assert_called_once_with(MoleculeORM, mol_id)
