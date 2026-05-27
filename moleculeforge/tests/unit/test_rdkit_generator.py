"""Unit tests for the RDKit random generator."""

from __future__ import annotations

import asyncio

import pytest


def _run(coro):
    return asyncio.run(coro)


class TestRDKitRandomGenerator:
    def test_import(self) -> None:
        from mf_generators.rdkit_random.generator import RDKitRandomGenerator

        gen = RDKitRandomGenerator()
        assert gen.name == "rdkit_random"
        assert gen.version == "0.1.0"

    def test_supported_modes(self) -> None:
        from mf_generators.rdkit_random.generator import RDKitRandomGenerator

        gen = RDKitRandomGenerator()
        assert "hit_finding" in gen.supported_modes

    def test_mutate_atom_type(self) -> None:
        from rdkit import Chem

        from mf_generators.rdkit_random.mutator import mutate_atom_type

        mol = Chem.MolFromSmiles("c1ccccc1")
        result = mutate_atom_type(mol)
        # Mutation may or may not succeed; just ensure no crash
        assert result is None or isinstance(result, Chem.Mol)

    def test_random_mutate_valid_smiles(self) -> None:
        from mf_generators.rdkit_random.mutator import random_mutate

        result = random_mutate("c1ccccc1", n_mutations=1)
        # May or may not succeed
        assert result is None or isinstance(result, str)

    def test_random_mutate_invalid_smiles(self) -> None:
        from mf_generators.rdkit_random.mutator import random_mutate

        result = random_mutate("not_a_smiles!!!")
        assert result is None

    def test_generate_yields_molecules(self) -> None:
        from mf_core.types.cig import ChemicalIntentGraph, ObjectiveNode, ObjectiveType
        from mf_core.types.hciv import HCIV, IntentCone
        from mf_generators.rdkit_random.generator import RDKitRandomGenerator

        gen = RDKitRandomGenerator(seed=42)

        hciv = HCIV(coordinates=[1.0, 0.0, 0.0, 0.0], dim=3)
        cone = IntentCone(apex=hciv, axis_direction=hciv, angle_radians=0.5, length=1.0)
        cig = ChemicalIntentGraph(
            intent_id="test",
            target_context={"pocket_embedding": None},
            objective_nodes=[
                ObjectiveNode(
                    id="obj-qed",
                    type=ObjectiveType.CONTINUOUS_MAXIMIZE,
                    oracle="rdkit_oracle_l0",
                    weight=1.0,
                )
            ],
            source_user_input="test",
        )

        async def _collect():
            mols = []
            async for mol in gen.generate(hciv, cone, cig, n_samples=5, seed=42):
                mols.append(mol)
                if len(mols) >= 5:
                    break
            return mols

        mols = _run(_collect())
        assert len(mols) > 0
        for mol in mols:
            assert mol.smiles
            assert mol.canonical_smiles
            assert mol.generator_name == "rdkit_random"
            assert mol.humu_embedding is None
