"""HUMU molecule encoder backed by RDKit graph features."""
from __future__ import annotations

import torch
import torch.nn as nn
from mf_humu.encoders.lorentz_attention import LorentzAttention
from mf_humu.manifold.learnable_lorentz import LearnableLorentzManifold
from mf_humu.manifold.lorentz import LorentzManifold

_ATOM_FEATURE_DIM = 16


class HUMUMoleculeEncoder(nn.Module):
    """Encode molecular graphs into Lorentz hyperboloid embeddings."""

    def __init__(
        self,
        dim: int = 128,
        curvature: float = 1.0,
        learnable_curvature: bool = False,
    ):
        super().__init__()
        manifold_cls = LearnableLorentzManifold if learnable_curvature else LorentzManifold
        self.manifold = manifold_cls(curvature=curvature)
        self.dim = dim
        self._atom_projection = nn.Linear(_ATOM_FEATURE_DIM, dim + 1)
        self._attention = LorentzAttention(
            dim=dim,
            heads=8,
            curvature=curvature,
            learnable_curvature=learnable_curvature,
        )

    def forward(self, molecule_smiles: str | list[str]) -> torch.Tensor:
        if isinstance(molecule_smiles, list):
            return self.encode_batch(molecule_smiles)
        return self.encode(molecule_smiles)

    def encode(self, molecule_smiles: str) -> torch.Tensor:
        """Encode a valid SMILES string through graph-derived atom features."""
        features, adjacency = self._graph_features(molecule_smiles)
        features = features.to(self._param_device())
        adjacency = adjacency.to(self._param_device())

        propagated = self._propagate(features, adjacency)
        x = self._atom_projection(propagated).unsqueeze(0)
        x = self.manifold._project(x)
        x = self._attention(x)
        embedding = x.mean(dim=1)
        return self.manifold._project(embedding)

    def encode_batch(self, smiles_list: list[str]) -> torch.Tensor:
        if not smiles_list:
            raise ValueError("molecule encoder requires at least one valid SMILES string")
        return torch.cat([self.encode(smiles) for smiles in smiles_list], dim=0)

    def _param_device(self) -> torch.device:
        return self._atom_projection.weight.device

    def _graph_features(self, molecule_smiles: str) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            from rdkit import Chem, rdBase
        except ImportError as exc:
            raise RuntimeError("RDKit is required for HUMU molecule graph encoding") from exc

        if not isinstance(molecule_smiles, str) or not molecule_smiles.strip():
            raise ValueError("molecule encoder requires a valid SMILES string")
        mol = _mol_from_smiles(Chem, rdBase, molecule_smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError("molecule encoder requires a valid SMILES string")

        features = torch.tensor(
            [self._atom_features(atom) for atom in mol.GetAtoms()],
            dtype=torch.float32,
        )
        adjacency = torch.eye(mol.GetNumAtoms(), dtype=torch.float32)
        for bond in mol.GetBonds():
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            order = float(bond.GetBondTypeAsDouble())
            adjacency[begin, end] = order
            adjacency[end, begin] = order
        return features, adjacency

    def _atom_features(self, atom) -> list[float]:
        try:
            from rdkit import Chem
        except ImportError as exc:
            raise RuntimeError("RDKit is required for HUMU molecule graph encoding") from exc

        hybridization = atom.GetHybridization()
        atomic_number = atom.GetAtomicNum()
        return [
            atomic_number / 100.0,
            atom.GetDegree() / 8.0,
            atom.GetTotalValence() / 8.0,
            atom.GetFormalCharge() / 5.0,
            atom.GetTotalNumHs() / 8.0,
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
            float(hybridization == Chem.HybridizationType.SP),
            float(hybridization == Chem.HybridizationType.SP2),
            float(hybridization == Chem.HybridizationType.SP3),
            float(atomic_number == 6),
            float(atomic_number == 7),
            float(atomic_number == 8),
            float(atomic_number == 16),
            float(atomic_number in {9, 17, 35, 53}),
            float(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED),
        ]

    def _propagate(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        normalized = adjacency / degree
        x = features
        for _ in range(2):
            x = normalized @ x
        return x


def _mol_from_smiles(Chem, rdBase, smiles: str):
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None
        sanitize_ops = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
        try:
            Chem.SanitizeMol(mol, sanitizeOps=sanitize_ops)
        except Exception:  # noqa: BLE001
            return None
    return mol
