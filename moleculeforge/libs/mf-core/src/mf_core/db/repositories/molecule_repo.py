"""MoleculeRepository — CRUD for molecules in PostgreSQL."""
from __future__ import annotations

import uuid

from sqlalchemy import select

from mf_core.db.orm import MoleculeORM


class MoleculeRepository:
    """Repository for molecule upsert and queries."""

    def __init__(self, session):
        self.session = session

    async def upsert(self, smiles: str, inchikey: str | None = None,
                     mw: float | None = None, logp: float | None = None) -> MoleculeORM:
        molecule = None
        if inchikey:
            result = await self.session.execute(
                select(MoleculeORM).where(MoleculeORM.inchikey == inchikey)
            )
            molecule = result.scalar_one_or_none()

        if molecule is None:
            molecule = MoleculeORM(smiles=smiles, inchikey=inchikey, mw=mw, logp=logp)
            self.session.add(molecule)
        else:
            molecule.smiles = smiles
            molecule.inchikey = inchikey
            molecule.mw = mw
            molecule.logp = logp

        await self.session.flush()
        return molecule

    async def get_by_inchikey(self, inchikey: str) -> MoleculeORM | None:
        result = await self.session.execute(
            select(MoleculeORM).where(MoleculeORM.inchikey == inchikey)
        )
        return result.scalar_one_or_none()

    async def get_by_id(self, mol_id: uuid.UUID) -> MoleculeORM | None:
        return await self.session.get(MoleculeORM, mol_id)

    async def batch_upsert(self, molecules: list[dict]) -> int:
        if not molecules:
            return 0
        for molecule in molecules:
            await self.upsert(**molecule)
        return len(molecules)
