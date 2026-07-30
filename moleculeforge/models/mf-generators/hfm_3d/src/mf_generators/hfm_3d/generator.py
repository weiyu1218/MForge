"""HFM-3D: Hyperbolic Flow Matching for 3D molecular generation."""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from mf_core.artifacts import CommandRequirement, check_command, require_available
from mf_core.geometry.lorentz import normalize_lorentz_embedding
from mf_core.plugins.generator import GeneratorPlugin
from mf_core.types.humu import IntentCone
from mf_core.types.molecule import Molecule
from mf_humu.manifold.lorentz import LorentzManifold
from mf_humu.operations.intent_cone import sample_within_cone

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Geometry import Point3D
except ImportError:  # pragma: no cover
    Chem = None
    AllChem = None
    Point3D = None

_MOLECULAR_DECODER_COMMAND = CommandRequirement(
    "hfm_molecular_decoder_command",
    "HFM_MOLECULAR_DECODER_COMMAND",
    required=False,
)
_VALIDATION_ARTIFACT_SCHEMA = "moleculeforge.validation_artifact.v1"
_VALIDATION_ARTIFACT_PURPOSE = "synthetic_pipeline_validation_only"
_VALIDATION_ARTIFACT_METADATA_FILE = "moleculeforge_validation_artifact.json"
_VALIDATION_ARTIFACT_MARKER_KEY = "moleculeforge_validation_artifact"
_VALIDATION_ARTIFACT_SEED = 7


class ExternalMolecularDecoder:
    def __init__(self, command: str):
        self.command = command
        self.timeout = float(os.getenv("HFM_MOLECULAR_DECODER_TIMEOUT_SECONDS", "300"))

    def decode(self, embedding: torch.Tensor) -> dict:
        _require_command_available(_MOLECULAR_DECODER_COMMAND, self.command)
        completed = subprocess.run(
            shlex.split(self.command),
            input=json.dumps(
                {"latent": embedding.detach().cpu().float().tolist()},
                sort_keys=True,
            ),
            capture_output=True,
            text=True,
            timeout=self.timeout,
            check=False,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            raise RuntimeError(f"HFM-3D molecular decoder command failed: {stderr}")
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "HFM-3D molecular decoder command returned invalid JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError("HFM-3D molecular decoder command must return a JSON object")
        return payload


class HFM3DGenerator(GeneratorPlugin):
    """Hyperbolic Flow Matching generator with learnable score decoder."""

    def __init__(
        self,
        checkpoint_path: str = "",
        device: str = "cpu",
        mode: str = "production_real",
        decoder_path: str = "",
        smiles_decoder: Callable[[torch.Tensor], str] | None = None,
        molecular_decoder: object | None = None,
    ) -> None:
        if mode not in {"production_real", "local_demo"}:
            raise ValueError(f"Unknown HFM3DGenerator mode: {mode}")
        self.manifold = LorentzManifold(curvature=1.0)
        self.device = device
        self.mode = mode
        self.checkpoint_path = checkpoint_path
        self.decoder_path = decoder_path
        self._checkpoint_loaded = False
        self._model = None
        self._decoder = None
        self._smiles_decoder = smiles_decoder
        self._molecular_decoder = molecular_decoder or _molecular_decoder_from_env()
        self._decoder_entries: list[dict[str, object]] = []
        self._build_model()
        if checkpoint_path:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"HFM-3D checkpoint not found: {checkpoint_path}")
            self.load_checkpoint(checkpoint_path)
        if decoder_path:
            self._decoder_entries = self._load_decoder_artifact(decoder_path)

    def _build_model(self) -> None:
        """Initialize the flow matching model and learned decoder."""
        from mf_generators.hfm_3d.model.lorentz_flow_matching import LorentzFlowMatching
        self._model = LorentzFlowMatching(dim=128, curvature=1.0, n_steps=20)
        self._decoder = nn.Sequential(
            nn.Linear(129, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 1024),
        )
        if self.device != "cpu":
            self._model.to(self.device)
            self._decoder.to(self.device)

    async def generate(
        self,
        batch_size: int,
        intent_cone: IntentCone | None = None,
        **kwargs: object,
    ) -> list[Molecule]:
        """Generate molecules via Lorentz flow matching with intent cone sampling."""
        if self.mode == "production_real" and (
            not self._checkpoint_loaded
            or (not self._decoder_entries and self._molecular_decoder is None)
        ):
            raise RuntimeError(
                "HFM-3D production generation requires a checkpoint and decoder artifact "
                "or molecular decoder"
            )
        sampling_seed = kwargs.get("sampling_seed")
        if self.mode == "production_real" and sampling_seed is None:
            raise RuntimeError("HFM-3D production generation requires sampling_seed")
        seed = int(sampling_seed) if sampling_seed is not None else None

        samples = []
        if intent_cone is not None:
            latent_points = sample_within_cone(
                intent_cone,
                n_samples=batch_size,
                manifold=self.manifold,
            )
        else:
            latent_points = self._sample_prior(batch_size, seed=seed)

        if isinstance(latent_points, np.ndarray):
            latent_points = torch.from_numpy(latent_points).float()
        if self.device != "cpu":
            latent_points = latent_points.to(self.device)

        with torch.no_grad():
            pre_flow_latent = latent_points.detach().cpu().clone()
            requested_flow_steps = kwargs.get("flow_steps")
            flow_steps = int(
                requested_flow_steps
                if requested_flow_steps is not None
                else getattr(self._model, "n_steps", 0) or 0
            )
            flow_max_step_norm = float(kwargs.get("flow_max_step_norm", 0.1))
            latent_points = self._run_lorentz_flow(
                latent_points,
                flow_steps,
                flow_max_step_norm=flow_max_step_norm,
            )
            if not torch.isfinite(latent_points).all():
                raise RuntimeError("HFM-3D flow produced non-finite latent")
            feedback_target, feedback_metadata = self._feedback_steering_target(
                kwargs,
                latent_points,
            )
            if feedback_target is not None:
                latent_points = self._apply_feedback_steering(
                    latent_points,
                    feedback_target,
                    feedback_metadata["feedback_steering_weight"],
                    feedback_metadata["feedback_steering_max_step"],
                )
                if not torch.isfinite(latent_points).all():
                    raise RuntimeError("HFM-3D feedback steering produced non-finite latent")
            for i in range(batch_size):
                conformer_seed = seed + i if seed is not None else 0
                smiles, sdf_bytes, decoder_entry_id, decoder_metadata = self._decode_molecule(
                    latent_points[i],
                    conformer_seed,
                )
                metadata = {
                    "generator_name": "hfm_3d",
                    "checkpoint": self.checkpoint_path,
                    "decode_artifact": self.decoder_path,
                    "decoder_entry_id": decoder_entry_id,
                    "sampling_seed": "" if seed is None else str(seed),
                    "input_cone": self._input_cone_provenance(intent_cone),
                    "flow_steps": str(flow_steps),
                    "flow_max_step_norm": str(flow_max_step_norm),
                    "pre_flow_latent": json.dumps(pre_flow_latent[i].tolist()),
                    "latent": json.dumps(
                        latent_points[i].detach().cpu().tolist()
                    ),
                }
                metadata.update(feedback_metadata)
                metadata.update(decoder_metadata)
                samples.append(
                    Molecule(
                        smiles=smiles,
                        sdf_bytes=sdf_bytes,
                        metadata=metadata,
                    )
                )
        return samples

    def _sample_prior(self, n: int, seed: int | None = None) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)
        z = torch.randn(n, 129, generator=generator)
        z[..., 0] = 0
        origin = torch.zeros(1, 129)
        origin[..., 0] = 1.0
        if self.device != "cpu":
            z = z.to(self.device)
            origin = origin.to(self.device)
        return self.manifold.expmap(origin.expand(n, -1), z)

    def _run_lorentz_flow(
        self,
        latent_points: torch.Tensor,
        flow_steps: int,
        flow_max_step_norm: float = 0.1,
    ) -> torch.Tensor:
        if flow_steps <= 0 or self._model is None:
            return latent_points
        if not hasattr(self._model, "compute_vector_field"):
            raise RuntimeError("HFM-3D flow model must expose compute_vector_field")
        max_step_norm = max(0.0, float(flow_max_step_norm))
        step_size = 1.0 / float(flow_steps)
        evolved = latent_points
        for step in range(flow_steps):
            t_value = (step + 0.5) * step_size
            t = torch.full(
                (evolved.shape[0], 1),
                t_value,
                device=evolved.device,
                dtype=evolved.dtype,
            )
            velocity = self._model.compute_vector_field(evolved, t)
            if velocity.shape != evolved.shape:
                raise RuntimeError("HFM-3D flow vector field shape must match latent points")
            tangent_step = velocity * step_size
            if max_step_norm > 0.0:
                step_norm = torch.linalg.vector_norm(
                    tangent_step,
                    dim=-1,
                    keepdim=True,
                ).clamp_min(self.manifold.eps)
                bounded_norm = torch.minimum(
                    step_norm,
                    torch.full_like(step_norm, max_step_norm),
                )
                tangent_step = tangent_step * (bounded_norm / step_norm)
            evolved = self.manifold.expmap(evolved, tangent_step)
        return evolved

    def _feedback_steering_target(
        self,
        kwargs: dict[str, object],
        latent_points: torch.Tensor,
    ) -> tuple[torch.Tensor | None, dict[str, str]]:
        records = []
        for key in ("jmcg_feedback", "route_humu_feedback", "generation_feedback"):
            records.extend(self._feedback_embedding_records(kwargs.get(key)))
        if not records:
            return None, {}

        embedding_dim = int(latent_points.shape[-1])
        accepted_records, dropped_count = _valid_feedback_records(records, embedding_dim)
        if not accepted_records:
            return None, {}

        weight = float(kwargs.get("feedback_steering_weight", 0.15))
        weight = max(0.0, min(weight, 1.0))
        max_step = float(kwargs.get("feedback_steering_max_step", 0.1))
        max_step = max(0.0, max_step)
        kind_targets = []
        kind_effective_weights = []
        for kind in sorted({str(record["kind"]) for record in accepted_records}):
            kind_records = [
                record
                for record in accepted_records
                if str(record["kind"]) == kind
            ]
            stacked = torch.stack(
                [
                    _weighted_feedback_embedding(
                        record,
                        dtype=latent_points.dtype,
                        device=latent_points.device,
                    )
                    for record in kind_records
                ]
            )
            kind_weight = sum(
                float(record["effective_weight"])
                for record in kind_records
            )
            if kind_weight <= 0.0:
                continue
            kind_mean = stacked.sum(dim=0) / kind_weight
            kind_target = kind_mean * _feedback_kind_weight(kind)
            kind_targets.append(kind_target)
            kind_effective_weights.append(
                (kind_weight / float(len(kind_records))) * _feedback_kind_weight(kind)
            )
        if not kind_targets:
            return None, {}
        stacked_kind_targets = torch.stack(kind_targets).to(latent_points.device)
        target = self.manifold._project(
            stacked_kind_targets.mean(dim=0, keepdim=True)
        ).squeeze(0)
        sources = [
            str(record["source"])
            for record in accepted_records
            if record["source"]
        ]
        kinds = [
            str(record["kind"])
            for record in accepted_records
            if record.get("kind")
        ]
        effective_weight = (
            sum(kind_effective_weights) / float(len(kind_effective_weights))
            if kind_effective_weights
            else 0.0
        )
        return target, {
            "feedback_steering_count": str(len(accepted_records)),
            "feedback_steering_dropped_count": str(dropped_count),
            "feedback_steering_sources": ",".join(sources),
            "feedback_steering_kinds": ",".join(sorted(set(kinds))),
            "feedback_steering_kind_count": str(len(set(kinds))),
            "feedback_steering_effective_weight": str(effective_weight),
            "feedback_steering_weight": str(weight),
            "feedback_steering_max_step": str(max_step),
        }

    def _feedback_embedding_records(self, payload: object) -> list[dict[str, object]]:
        if payload in (None, "", b"", []):
            return []
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        if isinstance(payload, str):
            payload = json.loads(payload)
        if isinstance(payload, Mapping):
            records = []
            for nested_key in (
                "jmcg_feedback",
                "route_humu_feedback",
                "generation_feedback",
                "records",
                "feedback",
                "items",
            ):
                if nested_key in payload:
                    records.extend(self._feedback_embedding_records(payload[nested_key]))
            embedding = payload.get("humu_embedding", payload.get("route_humu_embedding"))
            if isinstance(embedding, list) and embedding:
                source = (
                    _feedback_subject_id(payload)
                    or payload.get("route_id")
                    or payload.get("source")
                    or payload.get("id")
                    or ""
                )
                records.append(
                    {
                        "kind": str(payload.get("kind") or ""),
                        "source": str(source),
                        "embedding": [float(value) for value in embedding],
                        "weight": payload.get("weight", 1.0),
                        "confidence": payload.get("confidence", 1.0),
                        "polarity": str(payload.get("polarity") or "attract"),
                    }
                )
            return records
        if isinstance(payload, list):
            records = []
            for item in payload:
                records.extend(self._feedback_embedding_records(item))
            return records
        raise TypeError(f"Unsupported HFM-3D feedback payload: {type(payload)!r}")

    def _apply_feedback_steering(
        self,
        latent_points: torch.Tensor,
        feedback_target: torch.Tensor,
        steering_weight: str,
        steering_max_step: str,
    ) -> torch.Tensor:
        weight = float(steering_weight)
        max_step = float(steering_max_step)
        if weight <= 0.0 or max_step <= 0.0:
            return latent_points
        target = feedback_target.to(latent_points.device, dtype=latent_points.dtype)
        target = target.unsqueeze(0).expand(latent_points.shape[0], -1)
        direction = self.manifold.logmap(latent_points, target)
        direction_norm = torch.linalg.vector_norm(
            direction,
            dim=-1,
            keepdim=True,
        ).clamp_min(self.manifold.eps)
        requested_step = direction_norm * weight
        bounded_step = torch.minimum(
            requested_step,
            torch.full_like(requested_step, max_step),
        )
        step = direction * (bounded_step / direction_norm)
        forward = self.manifold.expmap(latent_points, step)
        backward = self.manifold.expmap(latent_points, -step)
        base_distance = self.manifold.distance(latent_points, target)
        forward_distance = self.manifold.distance(forward, target)
        backward_distance = self.manifold.distance(backward, target)
        use_forward = forward_distance <= backward_distance
        best = torch.where(use_forward.expand_as(forward), forward, backward)
        best_distance = torch.minimum(forward_distance, backward_distance)
        return torch.where((best_distance < base_distance).expand_as(best), best, latent_points)

    def _load_decoder_artifact(self, decoder_path: str) -> list[dict[str, object]]:
        path = Path(decoder_path)
        if not path.exists():
            raise FileNotFoundError(f"HFM-3D decoder artifact not found: {decoder_path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("HFM-3D decoder artifact must be a JSON object")
        entries = payload.get("entries")
        if not isinstance(entries, list) or not entries:
            raise ValueError("HFM-3D decoder artifact requires entries")
        return [
            self._normalize_decoder_entry(idx, entry)
            for idx, entry in enumerate(entries)
        ]

    def _normalize_decoder_entry(
        self,
        idx: int,
        entry: object,
    ) -> dict[str, object]:
        if not isinstance(entry, Mapping):
            raise ValueError("HFM-3D decoder entry must be a JSON object")
        smiles = entry.get("smiles")
        latent = entry.get("latent")
        if not isinstance(smiles, str) or not smiles:
            raise ValueError("HFM-3D decoder entry requires smiles")
        if not isinstance(latent, list) or len(latent) != 129:
            raise ValueError("HFM-3D decoder entry requires 129-d latent")
        latent_tensor = torch.tensor([float(value) for value in latent], dtype=torch.float32)
        normalized = {
            "id": str(entry.get("id", idx)),
            "smiles": self._canonical_smiles(smiles),
            "latent": latent_tensor,
        }
        sdf_payload = entry.get("sdf_bytes", entry.get("sdf"))
        if sdf_payload is not None:
            if isinstance(sdf_payload, bytes):
                sdf_text = sdf_payload.decode("utf-8")
            elif isinstance(sdf_payload, str) and sdf_payload:
                sdf_text = sdf_payload
            else:
                raise ValueError("HFM-3D decoder entry sdf must be a non-empty string")
            self._validate_decoder_sdf(normalized["smiles"], sdf_text)
            normalized["sdf"] = sdf_text
        return normalized

    def _validate_decoder_sdf(self, smiles: object, sdf_text: str) -> None:
        if Chem is None:
            raise ImportError("RDKit is required for HFM-3D decoder SDF validation")
        mol = Chem.MolFromMolBlock(sdf_text, sanitize=False, removeHs=False)
        if mol is None:
            raise ValueError("HFM-3D decoder entry sdf must be a parseable mol block")
        try:
            Chem.SanitizeMol(mol)
        except Exception as exc:
            raise ValueError("HFM-3D decoder entry sdf failed RDKit sanitization") from exc
        canonical_from_sdf = Chem.MolToSmiles(Chem.RemoveHs(mol))
        if canonical_from_sdf != str(smiles):
            raise ValueError("HFM-3D decoder entry sdf must match smiles")

    def _nearest_decoder_entry(self, embedding: torch.Tensor) -> dict[str, object]:
        embedding_cpu = embedding.detach().cpu().float()
        distances = [
            torch.sum((embedding_cpu - entry["latent"]) ** 2)
            for entry in self._decoder_entries
        ]
        idx = int(torch.argmin(torch.stack(distances)).item())
        return self._decoder_entries[idx]

    def _decode_to_smiles(self, embedding: torch.Tensor) -> tuple[str, str]:
        """Decode a Lorentz embedding to a SMILES string using learned decoder + fingerprint."""
        if self._decoder_entries:
            entry = self._nearest_decoder_entry(embedding)
            return str(entry["smiles"]), str(entry["id"])
        if self._smiles_decoder is not None:
            return self._canonical_smiles(self._smiles_decoder(embedding)), "callable_decoder"
        if self.mode != "local_demo":
            raise RuntimeError(
                "HFM-3D production generation requires a checkpoint and decoder artifact"
            )
        return self._decode_demo_smiles(embedding), "local_demo"

    def _decode_molecule(
        self,
        embedding: torch.Tensor,
        conformer_seed: int,
    ) -> tuple[str, bytes | None, str, dict[str, str]]:
        if self._molecular_decoder is not None:
            return self._decode_with_molecular_decoder(embedding, conformer_seed)
        if self._decoder_entries:
            entry = self._nearest_decoder_entry(embedding)
            smiles = str(entry["smiles"])
            if "sdf" in entry:
                return (
                    smiles,
                    str(entry["sdf"]).encode("utf-8"),
                    str(entry["id"]),
                    {"decoder_mode": "artifact_sdf"},
                )
            return (
                smiles,
                self._build_conformer(smiles, conformer_seed),
                str(entry["id"]),
                {},
            )
        smiles, decoder_entry_id = self._decode_to_smiles(embedding)
        return (
            smiles,
            self._build_conformer(smiles, conformer_seed),
            decoder_entry_id,
            {},
        )

    def _decode_with_molecular_decoder(
        self,
        embedding: torch.Tensor,
        conformer_seed: int,
    ) -> tuple[str, bytes | None, str, dict[str, str]]:
        decoder = self._molecular_decoder
        if decoder is None:
            raise RuntimeError("HFM-3D molecular decoder is not configured")
        decode = getattr(decoder, "decode", None)
        if callable(decode):
            payload = decode(embedding)
        elif callable(decoder):
            payload = decoder(embedding)
        else:
            raise TypeError("HFM-3D molecular decoder must be callable or expose decode()")

        if isinstance(payload, Molecule):
            smiles = self._canonical_smiles(payload.smiles)
            metadata = self._string_metadata(payload.metadata)
            metadata["decoder_mode"] = "molecular_decoder"
            decoder_entry_id = metadata.get("decoder_entry_id", "molecular_decoder")
            sdf_bytes = payload.sdf_bytes or self._build_conformer(smiles, conformer_seed)
            return smiles, sdf_bytes, decoder_entry_id, metadata
        if not isinstance(payload, Mapping):
            raise TypeError("HFM-3D molecular decoder must return Molecule or JSON object")

        smiles_value = payload.get("smiles")
        if not isinstance(smiles_value, str) or not smiles_value:
            raise ValueError("HFM-3D molecular decoder output requires smiles")
        smiles = self._canonical_smiles(smiles_value)

        metadata = self._string_metadata(payload.get("metadata", {}))
        metadata.setdefault("decoder_mode", "molecular_decoder")
        atom_types = payload.get("atom_types")
        coordinates = payload.get("coordinates")
        if atom_types is not None:
            metadata["decoded_atom_types"] = json.dumps(
                self._normalize_atom_types(atom_types)
            )
        if coordinates is not None:
            metadata["decoded_coordinates"] = json.dumps(
                self._normalize_coordinates(coordinates)
            )

        sdf_payload = payload.get("sdf_bytes", payload.get("sdf"))
        if isinstance(sdf_payload, bytes):
            sdf_bytes = sdf_payload
        elif isinstance(sdf_payload, str):
            sdf_bytes = sdf_payload.encode("utf-8")
        elif coordinates is not None:
            sdf_bytes = self._build_sdf_from_decoder_geometry(
                smiles,
                atom_types,
                coordinates,
            )
        else:
            sdf_bytes = self._build_conformer(smiles, conformer_seed)

        decoder_entry_id = payload.get("decoder_entry_id", payload.get("id"))
        if decoder_entry_id is None:
            decoder_entry_id = metadata.get("decoder_entry_id", "molecular_decoder")
        return smiles, sdf_bytes, str(decoder_entry_id), metadata

    def _canonical_smiles(self, smiles: str) -> str:
        if Chem is None:
            raise ImportError("RDKit is required for HFM-3D decoder validity checks")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"HFM-3D decoder produced invalid SMILES: {smiles}")
        return Chem.MolToSmiles(mol)

    def _build_conformer(self, smiles: str, seed: int) -> bytes:
        if Chem is None or AllChem is None:
            raise ImportError("RDKit is required for HFM-3D conformer generation")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"HFM-3D decoder produced invalid SMILES: {smiles}")
        mol = Chem.AddHs(mol)
        status = AllChem.EmbedMolecule(mol, randomSeed=int(seed))
        if status != 0:
            raise RuntimeError(f"HFM-3D conformer generation failed for {smiles}")
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        return Chem.MolToMolBlock(mol).encode("utf-8")

    def _build_sdf_from_decoder_geometry(
        self,
        smiles: str,
        atom_types: object,
        coordinates: object,
    ) -> bytes:
        if Chem is None or Point3D is None:
            raise ImportError("RDKit is required for HFM-3D geometry decoding")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"HFM-3D decoder produced invalid SMILES: {smiles}")
        decoded_coordinates = self._normalize_coordinates(coordinates)
        if len(decoded_coordinates) != mol.GetNumAtoms():
            raise ValueError("HFM-3D decoder coordinates must match molecule atom count")
        if atom_types is not None:
            decoded_atom_types = self._normalize_atom_types(atom_types)
            if len(decoded_atom_types) != mol.GetNumAtoms():
                raise ValueError("HFM-3D decoder atom types must match atom count")
            for atom_idx, atom_type in enumerate(decoded_atom_types):
                if mol.GetAtomWithIdx(atom_idx).GetSymbol() != atom_type:
                    raise ValueError("HFM-3D decoder atom type does not match SMILES")
        conformer = Chem.Conformer(mol.GetNumAtoms())
        for atom_idx, (x_coord, y_coord, z_coord) in enumerate(decoded_coordinates):
            conformer.SetAtomPosition(
                atom_idx,
                Point3D(x_coord, y_coord, z_coord),
            )
        mol.RemoveAllConformers()
        mol.AddConformer(conformer, assignId=True)
        return Chem.MolToMolBlock(mol).encode("utf-8")

    def _normalize_atom_types(self, atom_types: object) -> list[str]:
        if not isinstance(atom_types, list) or not atom_types:
            raise ValueError("HFM-3D decoder atom_types must be a non-empty list")
        normalized = []
        for atom_type in atom_types:
            if not isinstance(atom_type, str) or not atom_type:
                raise ValueError("HFM-3D decoder atom_types must contain strings")
            normalized.append(atom_type)
        return normalized

    def _normalize_coordinates(self, coordinates: object) -> list[list[float]]:
        if not isinstance(coordinates, list) or not coordinates:
            raise ValueError("HFM-3D decoder coordinates must be a non-empty list")
        normalized = []
        for coordinate in coordinates:
            if not isinstance(coordinate, list) or len(coordinate) != 3:
                raise ValueError("HFM-3D decoder coordinates must be 3D points")
            normalized.append([float(value) for value in coordinate])
        return normalized

    def _string_metadata(self, metadata: object) -> dict[str, str]:
        if not isinstance(metadata, Mapping):
            raise ValueError("HFM-3D decoder metadata must be a JSON object")
        return {str(key): str(value) for key, value in metadata.items()}

    def _input_cone_provenance(self, intent_cone: IntentCone | None) -> str:
        if intent_cone is None:
            return ""
        if hasattr(intent_cone, "model_dump_json"):
            return intent_cone.model_dump_json()
        return str(intent_cone)

    def _decode_demo_smiles(self, embedding: torch.Tensor) -> str:
        """Decode by fixed pool for explicit local demo mode only."""
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        h = self._decoder(embedding)
        fp = (h.squeeze(0).cpu() > 0).int()

        smiles_pool = [
            "c1ccccc1", "CCO", "CC(=O)O", "CCN", "CCCl",
            "c1ccncc1", "CC(=O)N", "CCS", "CC(C)C", "c1ccoc1",
            "CN(C)C", "CC(=O)OC", "c1ccco1", "CS(=O)(=O)O",
            "COC", "CC(C)=O", "c1ccsc1", "CC#N", "CCOC",
            "c1ccc2c(c1)ccc1ccccc21", "C[C@@H](O)C(=O)O",
            "c1ccccc1N", "O=C(O)c1ccccc1", "CCCCN",
        ]
        idx = int(fp.sum()) % len(smiles_pool)
        return smiles_pool[idx]

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint to disk."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state = {
            "model": self._model.state_dict() if self._model else {},
            "decoder": self._decoder.state_dict() if self._decoder else {},
            "device": self.device,
        }
        torch.save(state, path)

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint from disk."""
        state = torch.load(path, map_location=self.device, weights_only=True)
        if self._model:
            model_state = state.get("model") or state.get("flow_model")
            if not isinstance(model_state, Mapping) or not model_state:
                raise ValueError("HFM-3D checkpoint requires model state")
            self._model.load_state_dict(model_state, strict=True)
        if self._decoder:
            decoder_state = state.get("decoder")
            if decoder_state is not None and not isinstance(decoder_state, Mapping):
                raise ValueError("HFM-3D checkpoint decoder state must be a mapping")
            if decoder_state:
                self._decoder.load_state_dict(decoder_state, strict=True)
        self._checkpoint_loaded = True

    async def info(self) -> dict:
        return {
            "name": "hfm_3d",
            "version": "0.1.0",
            "description": "Hyperbolic Flow Matching for 3D molecular generation",
            "supported_properties": ["qed", "logp", "sa_score", "mw"],
            "max_batch_size": 1024,
            "supports_streaming": True,
            "requires_gpu": True,
            "has_checkpoint": self._checkpoint_loaded,
        }


async def bootstrap_validation_artifacts(
    target_directory: str | Path,
) -> dict[str, Path]:
    target = Path(target_directory).expanduser().resolve()
    if target.exists():
        return await _validated_existing_validation_artifacts(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent)
    )
    try:
        checkpoint_path = temporary / "hfm_checkpoint.pt"
        decoder_path = temporary / "decoder.json"
        with torch.random.fork_rng():
            torch.manual_seed(_VALIDATION_ARTIFACT_SEED)
            HFM3DGenerator(mode="local_demo").save_checkpoint(str(checkpoint_path))
        checkpoint_state = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        checkpoint_state[_VALIDATION_ARTIFACT_MARKER_KEY] = (
            _validation_artifact_marker()
        )
        torch.save(checkpoint_state, checkpoint_path)
        _write_json(
            decoder_path,
            {
                "schema_version": _VALIDATION_ARTIFACT_SCHEMA,
                "purpose": _VALIDATION_ARTIFACT_PURPOSE,
                "generator": "hfm_3d",
                "seed": _VALIDATION_ARTIFACT_SEED,
                _VALIDATION_ARTIFACT_MARKER_KEY: _validation_artifact_marker(),
                "entries": [
                    {
                        "id": "validation_ethanol",
                        "smiles": "CCO",
                        "latent": [1.0, *([0.0] * 128)],
                    },
                    {
                        "id": "validation_ethylamine",
                        "smiles": "CCN",
                        "latent": [
                            float(np.cosh(0.1)),
                            float(np.sinh(0.1)),
                            *([0.0] * 127),
                        ],
                    },
                ],
            },
        )
        _write_validation_metadata(
            temporary,
            {
                "checkpoint": checkpoint_path,
                "decoder": decoder_path,
            },
        )
        paths = _validation_artifact_paths(temporary)
        await _probe_validation_artifacts(paths)
        _fsync_tree(temporary)
        if target.exists():
            return await _validated_existing_validation_artifacts(target)
        try:
            temporary.rename(target)
        except OSError:
            if not target.exists():
                raise
            return await _validated_existing_validation_artifacts(target)
        _fsync_directory(target.parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return _validation_artifact_paths(target)


async def _validated_existing_validation_artifacts(
    target: Path,
) -> dict[str, Path]:
    try:
        paths = _validation_artifact_paths(target)
        await _probe_validation_artifacts(paths)
    except Exception as exc:
        raise RuntimeError(
            f"HFM validation bootstrap refuses to overwrite existing path: {target}"
        ) from exc
    return paths


def load_validation_artifact_metadata(
    artifact_path: str | Path,
) -> dict[str, object] | None:
    artifact = Path(artifact_path).expanduser().resolve()
    metadata_path = artifact.parent / _VALIDATION_ARTIFACT_METADATA_FILE
    if not metadata_path.is_file():
        return _read_embedded_validation_artifact_metadata(artifact)
    metadata = _read_validation_metadata(metadata_path)
    records = metadata["artifacts"]
    if not any(
        isinstance(record, Mapping) and record.get("file") == artifact.name
        for record in records.values()
    ):
        raise RuntimeError("HFM validation metadata does not reference configured artifact")
    _validate_artifact_records(artifact.parent, records)
    return metadata


def _read_embedded_validation_artifact_metadata(
    artifact_path: Path,
) -> dict[str, object] | None:
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        try:
            payload = torch.load(
                artifact_path,
                map_location="cpu",
                weights_only=True,
            )
        except Exception:
            return None
    return _validation_artifact_marker_from_payload(payload)


def _validation_artifact_marker_from_payload(
    payload: object,
) -> dict[str, object] | None:
    if not isinstance(payload, Mapping):
        return None
    marker_present = _VALIDATION_ARTIFACT_MARKER_KEY in payload
    marker = payload.get(_VALIDATION_ARTIFACT_MARKER_KEY)
    if not marker_present and (
        payload.get("schema_version") == _VALIDATION_ARTIFACT_SCHEMA
        or payload.get("purpose") == _VALIDATION_ARTIFACT_PURPOSE
    ):
        marker_present = True
        marker = payload
    if not marker_present:
        return None
    if (
        not isinstance(marker, Mapping)
        or marker.get("schema_version") != _VALIDATION_ARTIFACT_SCHEMA
        or marker.get("purpose") != _VALIDATION_ARTIFACT_PURPOSE
        or marker.get("generator") != "hfm_3d"
        or marker.get("seed") != _VALIDATION_ARTIFACT_SEED
    ):
        raise RuntimeError("HFM embedded validation artifact marker is invalid")
    return {
        "schema_version": _VALIDATION_ARTIFACT_SCHEMA,
        "purpose": _VALIDATION_ARTIFACT_PURPOSE,
        "generator": "hfm_3d",
        "seed": _VALIDATION_ARTIFACT_SEED,
    }


def _validation_artifact_marker() -> dict[str, object]:
    return {
        "schema_version": _VALIDATION_ARTIFACT_SCHEMA,
        "purpose": _VALIDATION_ARTIFACT_PURPOSE,
        "generator": "hfm_3d",
        "seed": _VALIDATION_ARTIFACT_SEED,
    }


def _validation_artifact_paths(directory: Path) -> dict[str, Path]:
    metadata_path = directory / _VALIDATION_ARTIFACT_METADATA_FILE
    if not metadata_path.is_file():
        raise RuntimeError("HFM validation artifact metadata is missing")
    metadata = _read_validation_metadata(metadata_path)
    records = metadata["artifacts"]
    if set(records) != {"checkpoint", "decoder"}:
        raise RuntimeError("HFM validation artifact metadata has invalid artifact set")
    _validate_artifact_records(directory, records)
    return {
        "checkpoint": directory / str(records["checkpoint"]["file"]),
        "decoder": directory / str(records["decoder"]["file"]),
        "metadata": metadata_path,
    }


def _read_validation_metadata(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("HFM validation artifact metadata is invalid") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != _VALIDATION_ARTIFACT_SCHEMA
        or payload.get("purpose") != _VALIDATION_ARTIFACT_PURPOSE
        or payload.get("generator") != "hfm_3d"
        or payload.get("seed") != _VALIDATION_ARTIFACT_SEED
        or not isinstance(payload.get("artifacts"), dict)
    ):
        raise RuntimeError("HFM validation artifact metadata is invalid")
    return payload


def _validate_artifact_records(
    directory: Path,
    records: Mapping[str, object],
) -> None:
    for record in records.values():
        if not isinstance(record, Mapping):
            raise RuntimeError("HFM validation artifact record is invalid")
        filename = record.get("file")
        expected_sha256 = record.get("sha256")
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
            or not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
        ):
            raise RuntimeError("HFM validation artifact record is invalid")
        artifact_path = directory / filename
        if not artifact_path.is_file():
            raise RuntimeError(f"HFM validation artifact is missing: {artifact_path}")
        if _sha256(artifact_path) != expected_sha256:
            raise RuntimeError(f"HFM validation artifact checksum mismatch: {artifact_path}")


def _write_validation_metadata(
    directory: Path,
    artifacts: Mapping[str, Path],
) -> None:
    _write_json(
        directory / _VALIDATION_ARTIFACT_METADATA_FILE,
        {
            "schema_version": _VALIDATION_ARTIFACT_SCHEMA,
            "purpose": _VALIDATION_ARTIFACT_PURPOSE,
            "generator": "hfm_3d",
            "seed": _VALIDATION_ARTIFACT_SEED,
            "artifacts": {
                name: {
                    "file": path.name,
                    "sha256": _sha256(path),
                }
                for name, path in artifacts.items()
            },
        },
    )


async def _probe_validation_artifacts(paths: Mapping[str, Path]) -> None:
    generator = HFM3DGenerator(
        checkpoint_path=str(paths["checkpoint"]),
        decoder_path=str(paths["decoder"]),
        mode="production_real",
    )
    molecules = await generator.generate(
        batch_size=2,
        sampling_seed=_VALIDATION_ARTIFACT_SEED,
        flow_steps=1,
    )
    if len(molecules) != 2 or any(
        Chem is None or Chem.MolFromSmiles(molecule.smiles) is None
        for molecule in molecules
    ):
        raise RuntimeError("HFM validation artifact production probe failed")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            dict(payload),
            handle,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_tree(directory: Path) -> None:
    for artifact in sorted(path for path in directory.rglob("*") if path.is_file()):
        with artifact.open("rb") as handle:
            os.fsync(handle.fileno())
    _fsync_directory(directory)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(
        directory,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _molecular_decoder_from_env() -> ExternalMolecularDecoder | None:
    command = os.getenv("HFM_MOLECULAR_DECODER_COMMAND", "").strip()
    if not command:
        return None
    return ExternalMolecularDecoder(command)


def _feedback_subject_id(payload: Mapping) -> str:
    subject = payload.get("subject")
    if isinstance(subject, Mapping):
        return str(subject.get("id") or "")
    return ""


def _valid_feedback_records(
    records: list[dict[str, object]],
    embedding_dim: int,
) -> tuple[list[dict[str, object]], int]:
    accepted = []
    dropped = 0
    for record in records:
        embedding = record.get("embedding")
        normalized_embedding = normalize_lorentz_embedding(
            embedding,
            expected_dim=embedding_dim,
            curvature=1.0,
        )
        if normalized_embedding is None:
            dropped += 1
            continue
        try:
            record_weight = float(record.get("weight", 1.0))
            confidence = max(0.0, min(float(record.get("confidence", 1.0)), 1.0))
        except (TypeError, ValueError):
            dropped += 1
            continue
        polarity = str(record.get("polarity") or "attract")
        if record_weight < 0.0 or confidence <= 0.0:
            dropped += 1
            continue
        if polarity not in {"attract", "repel"}:
            dropped += 1
            continue
        effective_weight = record_weight * confidence
        if effective_weight <= 0.0:
            dropped += 1
            continue
        accepted_record = dict(record)
        accepted_record["embedding"] = normalized_embedding
        accepted_record["kind"] = str(record.get("kind") or "unspecified")
        accepted_record["effective_weight"] = effective_weight
        accepted_record["polarity"] = polarity
        accepted.append(accepted_record)
    return accepted, dropped


def _weighted_feedback_embedding(
    record: dict[str, object],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    embedding = torch.tensor(
        record["embedding"],
        dtype=dtype,
        device=device,
    )
    polarity = str(record.get("polarity") or "attract")
    if polarity == "repel":
        embedding = embedding.clone()
        embedding[1:] = -embedding[1:]
    return embedding * float(record["effective_weight"])


def _feedback_kind_weight(kind: str) -> float:
    return {
        "molecule": 1.0,
        "route": 0.8,
        "property": 0.8,
        "pocket": 1.0,
        "intent": 1.0,
        "unspecified": 1.0,
    }.get(kind, 1.0)


def _require_command_available(
    requirement: CommandRequirement,
    command: str,
) -> None:
    required_requirement = CommandRequirement(
        requirement.name,
        requirement.env_var,
        required=True,
    )
    env = {**os.environ, requirement.env_var: command}
    require_available([check_command(required_requirement, env=env)])
