"""Neural geometry decoder artifact helpers for HFM-3D."""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from mf_core.geometry import normalize_lorentz_embedding

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover
    Chem = None


@dataclass(frozen=True)
class GeometryTrainingExample:
    entry_id: str
    smiles: str
    latent: torch.Tensor
    atom_types: list[str]
    coordinates: torch.Tensor


@dataclass(frozen=True)
class GeometryDecoderEntry:
    entry_id: str
    smiles: str
    latent: torch.Tensor
    atom_types: list[str]


class NeuralGeometryDecoder(nn.Module):
    def __init__(self, latent_dim: int = 129, max_atoms: int = 64) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.max_atoms = int(max_atoms)
        self.network = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.max_atoms * 3),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        coordinates = self.network(latent)
        return coordinates.reshape(latent.shape[0], self.max_atoms, 3)


@dataclass
class NeuralGeometryDecoderArtifact:
    model: NeuralGeometryDecoder
    entries: list[GeometryDecoderEntry]
    latent_dim: int = 129
    max_atoms: int = 64
    source_decoder_artifact: str = ""

    @classmethod
    def load(
        cls,
        path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> "NeuralGeometryDecoderArtifact":
        payload = torch.load(path, map_location=map_location, weights_only=True)
        if not isinstance(payload, Mapping):
            raise ValueError("HFM-3D neural geometry decoder artifact must be a dict")
        latent_dim = int(payload.get("latent_dim", 129))
        max_atoms = int(payload.get("max_atoms", 64))
        model = NeuralGeometryDecoder(latent_dim=latent_dim, max_atoms=max_atoms)
        model_state = payload.get("model_state")
        if not isinstance(model_state, Mapping):
            raise ValueError("HFM-3D neural geometry decoder artifact requires model_state")
        model.load_state_dict(model_state)
        model.eval()
        entries = [_artifact_entry_from_payload(entry) for entry in payload.get("entries", [])]
        if not entries:
            raise ValueError("HFM-3D neural geometry decoder artifact requires entries")
        return cls(
            model=model,
            entries=entries,
            latent_dim=latent_dim,
            max_atoms=max_atoms,
            source_decoder_artifact=str(payload.get("source_decoder_artifact", "")),
        )

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema": "moleculeforge.hfm_3d.neural_geometry_decoder.v1",
                "model_state": self.model.state_dict(),
                "latent_dim": int(self.latent_dim),
                "max_atoms": int(self.max_atoms),
                "entries": [
                    {
                        "id": entry.entry_id,
                        "smiles": entry.smiles,
                        "latent": entry.latent.detach().cpu().float().tolist(),
                        "atom_types": list(entry.atom_types),
                    }
                    for entry in self.entries
                ],
                "source_decoder_artifact": self.source_decoder_artifact,
            },
            output_path,
        )

    def predict_coordinates(
        self,
        latent: torch.Tensor,
        atom_count: int,
    ) -> torch.Tensor:
        atom_count = int(atom_count)
        if atom_count < 1 or atom_count > self.max_atoms:
            raise ValueError("HFM-3D neural geometry decoder atom_count is out of range")
        latent = latent.detach().cpu().float()
        if latent.numel() != self.latent_dim:
            raise ValueError("HFM-3D neural geometry decoder latent dimension mismatch")
        with torch.no_grad():
            coordinates = self.model(latent.unsqueeze(0))[0, :atom_count].cpu()
        return coordinates

    def nearest_entry(self, latent: torch.Tensor) -> GeometryDecoderEntry:
        latent = latent.detach().cpu().float()
        if latent.numel() != self.latent_dim:
            raise ValueError("HFM-3D neural geometry decoder latent dimension mismatch")
        distances = [
            torch.sum((latent - entry.latent.float()) ** 2)
            for entry in self.entries
        ]
        return self.entries[int(torch.argmin(torch.stack(distances)).item())]

    def decode(self, latent: torch.Tensor) -> dict[str, object]:
        entry = self.nearest_entry(latent)
        coordinates = self.predict_coordinates(
            latent,
            atom_count=len(entry.atom_types),
        )
        return {
            "smiles": entry.smiles,
            "atom_types": list(entry.atom_types),
            "coordinates": [
                [float(value) for value in coordinate]
                for coordinate in coordinates.tolist()
            ],
            "decoder_entry_id": entry.entry_id,
            "metadata": {
                "decoder_mode": "neural_geometry_decoder",
                "source_decoder_artifact": self.source_decoder_artifact,
            },
        }


def load_geometry_training_examples(
    decoder_artifact: str | Path,
) -> list[GeometryTrainingExample]:
    path = Path(decoder_artifact)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("HFM-3D geometry decoder source artifact must be a JSON object")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("HFM-3D geometry decoder source artifact requires entries")
    return [
        _geometry_example_from_decoder_entry(index, entry)
        for index, entry in enumerate(entries)
    ]


def _geometry_example_from_decoder_entry(
    index: int,
    entry: object,
) -> GeometryTrainingExample:
    if not isinstance(entry, Mapping):
        raise ValueError("HFM-3D geometry decoder entry must be a JSON object")
    smiles = entry.get("smiles")
    latent = entry.get("latent")
    sdf = entry.get("sdf_bytes", entry.get("sdf"))
    if not isinstance(smiles, str) or not smiles:
        raise ValueError("HFM-3D geometry decoder entry requires smiles")
    if not isinstance(latent, list) or len(latent) != 129:
        raise ValueError("HFM-3D geometry decoder entry requires 129-d latent")
    normalized_latent = normalize_lorentz_embedding(
        latent,
        expected_dim=129,
        curvature=1.0,
    )
    if normalized_latent is None:
        raise ValueError("HFM-3D geometry decoder entry requires valid Lorentz latent")
    if not isinstance(sdf, str) or not sdf:
        raise ValueError("HFM-3D geometry decoder entry requires sdf")

    atom_types, coordinates = _parse_sdf_geometry(sdf)
    return GeometryTrainingExample(
        entry_id=str(entry.get("id", index)),
        smiles=_canonical_smiles(smiles),
        latent=torch.tensor(normalized_latent, dtype=torch.float32),
        atom_types=atom_types,
        coordinates=coordinates,
    )


def _parse_sdf_geometry(sdf_text: str) -> tuple[list[str], torch.Tensor]:
    if Chem is None:
        raise ImportError("RDKit is required for HFM-3D geometry decoder artifacts")
    mol = Chem.MolFromMolBlock(sdf_text, sanitize=False, removeHs=False)
    if mol is None:
        raise ValueError("HFM-3D geometry decoder entry sdf must be parseable")
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:
        raise ValueError("HFM-3D geometry decoder entry sdf failed sanitization") from exc
    mol = Chem.RemoveHs(mol)
    if mol.GetNumConformers() < 1:
        raise ValueError("HFM-3D geometry decoder entry sdf requires coordinates")
    conformer = mol.GetConformer()
    atom_types = [atom.GetSymbol() for atom in mol.GetAtoms()]
    coordinates = []
    for atom_idx in range(mol.GetNumAtoms()):
        position = conformer.GetAtomPosition(atom_idx)
        coordinates.append([float(position.x), float(position.y), float(position.z)])
    return atom_types, torch.tensor(coordinates, dtype=torch.float32)


def _canonical_smiles(smiles: str) -> str:
    if Chem is None:
        raise ImportError("RDKit is required for HFM-3D geometry decoder artifacts")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"HFM-3D geometry decoder entry has invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol)


def train_geometry_decoder_artifact(
    decoder_artifact: str | Path,
    output_artifact: str | Path,
    epochs: int = 5,
    batch_size: int = 32,
    device: str | torch.device = "cpu",
    max_atoms: int | None = None,
    learning_rate: float = 1e-3,
) -> NeuralGeometryDecoderArtifact:
    examples = load_geometry_training_examples(decoder_artifact)
    latent_dim = int(examples[0].latent.numel())
    if any(int(example.latent.numel()) != latent_dim for example in examples):
        raise ValueError("HFM-3D geometry decoder examples must share latent dimension")
    max_observed_atoms = max(int(example.coordinates.shape[0]) for example in examples)
    max_atoms = max_observed_atoms if max_atoms is None else int(max_atoms)
    if max_atoms < max_observed_atoms:
        raise ValueError("HFM-3D geometry decoder max_atoms is smaller than training data")

    torch.manual_seed(0)
    device = torch.device(device)
    model = NeuralGeometryDecoder(latent_dim=latent_dim, max_atoms=max_atoms).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    latents = torch.stack([example.latent for example in examples]).to(device)
    targets = torch.zeros(len(examples), max_atoms, 3, dtype=torch.float32, device=device)
    mask = torch.zeros(len(examples), max_atoms, 1, dtype=torch.float32, device=device)
    for example_idx, example in enumerate(examples):
        atom_count = int(example.coordinates.shape[0])
        targets[example_idx, :atom_count] = example.coordinates.to(device)
        mask[example_idx, :atom_count] = 1.0

    batch_size = max(1, int(batch_size))
    for _epoch in range(max(1, int(epochs))):
        model.train()
        for start in range(0, len(examples), batch_size):
            end = min(start + batch_size, len(examples))
            prediction = model(latents[start:end])
            loss = ((prediction - targets[start:end]) ** 2 * mask[start:end]).sum()
            loss = loss / mask[start:end].sum().clamp_min(1.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    artifact = NeuralGeometryDecoderArtifact(
        model=model.cpu(),
        entries=[
            GeometryDecoderEntry(
                entry_id=example.entry_id,
                smiles=example.smiles,
                latent=example.latent.cpu(),
                atom_types=list(example.atom_types),
            )
            for example in examples
        ],
        latent_dim=latent_dim,
        max_atoms=max_atoms,
        source_decoder_artifact=str(Path(decoder_artifact)),
    )
    artifact.save(output_artifact)
    return artifact


def _artifact_entry_from_payload(entry: object) -> GeometryDecoderEntry:
    if not isinstance(entry, Mapping):
        raise ValueError("HFM-3D neural geometry decoder entry must be a JSON object")
    latent = entry.get("latent")
    atom_types = entry.get("atom_types")
    smiles = entry.get("smiles")
    if not isinstance(latent, list):
        raise ValueError("HFM-3D neural geometry decoder entry requires latent")
    if not isinstance(atom_types, list) or not atom_types:
        raise ValueError("HFM-3D neural geometry decoder entry requires atom_types")
    if not isinstance(smiles, str) or not smiles:
        raise ValueError("HFM-3D neural geometry decoder entry requires smiles")
    return GeometryDecoderEntry(
        entry_id=str(entry.get("id", "")),
        smiles=_canonical_smiles(smiles),
        latent=torch.tensor([float(value) for value in latent], dtype=torch.float32),
        atom_types=[str(atom_type) for atom_type in atom_types],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an HFM-3D neural geometry decoder")
    parser.add_argument("--artifact", required=True, help="Path to geometry decoder .pt")
    args = parser.parse_args(argv)
    try:
        request = json.load(sys.stdin)
        if not isinstance(request, Mapping):
            raise ValueError("request must be a JSON object")
        latent = request.get("latent")
        if not isinstance(latent, list):
            raise ValueError("request requires latent list")
        artifact = NeuralGeometryDecoderArtifact.load(args.artifact, map_location="cpu")
        payload = artifact.decode(torch.tensor([float(value) for value in latent]))
        json.dump(payload, sys.stdout, separators=(",", ":"), sort_keys=True)
        sys.stdout.write("\n")
        return 0
    except Exception as exc:
        print(f"HFM-3D neural geometry decoder failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
