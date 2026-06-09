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

    def forward(self, molecule_smiles: str | dict | list[str | dict]) -> torch.Tensor:
        if isinstance(molecule_smiles, list):
            return self.encode_batch(molecule_smiles)
        return self.encode(molecule_smiles)

    def encode(self, molecule_smiles: str | dict) -> torch.Tensor:
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

    def encode_batch(self, smiles_list: list[str | dict]) -> torch.Tensor:
        if not smiles_list:
            raise ValueError("molecule encoder requires at least one valid SMILES string")
        return torch.cat([self.encode(smiles) for smiles in smiles_list], dim=0)

    def _param_device(self) -> torch.device:
        return self._atom_projection.weight.device

    def _graph_features(self, molecule_smiles: str | dict) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            from rdkit import Chem, rdBase
        except ImportError as exc:
            raise RuntimeError("RDKit is required for HUMU molecule graph encoding") from exc

        coords = None
        if isinstance(molecule_smiles, dict):
            coords = molecule_smiles.get("coords", molecule_smiles.get("coordinates"))
            molecule_smiles = str(molecule_smiles.get("smiles") or "")
        if not isinstance(molecule_smiles, str) or not molecule_smiles.strip():
            raise ValueError("molecule encoder requires a valid SMILES string")
        mol = _mol_from_smiles(Chem, rdBase, molecule_smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError("molecule encoder requires a valid SMILES string")

        features = torch.tensor(
            [self._atom_features(atom) for atom in mol.GetAtoms()],
            dtype=torch.float32,
        )
        if coords is not None:
            features = features + self._geometry_features(coords, mol.GetNumAtoms())
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

    def _geometry_features(self, coords: object, atom_count: int) -> torch.Tensor:
        if not isinstance(coords, list) or len(coords) != atom_count:
            raise ValueError("molecule encoder coords must match atom count")
        coordinates = []
        for coord in coords:
            if not isinstance(coord, list | tuple) or len(coord) != 3:
                raise ValueError("molecule encoder coords must contain 3D points")
            coordinates.append([float(value) for value in coord])
        coordinate_tensor = torch.tensor(coordinates, dtype=torch.float32)
        centered = coordinate_tensor - coordinate_tensor.mean(dim=0, keepdim=True)
        pairwise = torch.cdist(centered, centered)
        non_self = ~torch.eye(atom_count, dtype=torch.bool)
        masked = pairwise.masked_fill(~non_self, 0.0)
        neighbor_count = max(atom_count - 1, 1)
        mean_distance = masked.sum(dim=-1) / float(neighbor_count)
        max_distance = pairwise.max(dim=-1).values
        radial_distance = torch.linalg.vector_norm(centered, dim=-1)
        distance_scale = pairwise.max().clamp_min(1.0)

        geometry = torch.zeros(atom_count, _ATOM_FEATURE_DIM, dtype=torch.float32)
        geometry[:, 1] = radial_distance / distance_scale
        geometry[:, 2] = mean_distance / distance_scale
        geometry[:, 3] = max_distance / distance_scale
        return geometry

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
