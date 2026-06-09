"""Reference candidate provider and oracle evaluator for local PCBO closed loop.

These implementations connect to existing L0-L3 oracle services via gRPC and provide
a working local closed loop without external production resources.

Callable paths for env injection:
  PARETO_BO_CANDIDATE_PROVIDER=pareto_bo.providers:TangentSpaceNoiseCandidateProvider
  PARETO_BO_ORACLE_EVALUATE=pareto_bo.providers:LocalOracleEvaluator
"""
from __future__ import annotations

import json
import os
from typing import Any

import torch


_NOISE_SCALE_ENV = "PARETO_BO_CANDIDATE_NOISE_SCALE"
_N_CANDIDATES_ENV = "PARETO_BO_CANDIDATE_COUNT"
_ORACLE_LEVEL_ENV = "PARETO_BO_ORACLE_LEVEL"
_ORACLE_TARGET_ENVS = {
    0: "L0_ADMET_ORACLE_TARGET",
    1: "L1_DOCKING_ORACLE_TARGET",
    2: "L2_AFFINITY_ORACLE_TARGET",
    3: "L3_FEP_ORACLE_TARGET",
}
_SMILES_LIST_ENV = "PARETO_BO_SMILES_LIST_JSON"
_ORACLE_PROPERTIES_ENV = "PARETO_BO_ORACLE_PROPERTIES"


class TangentSpaceNoiseCandidateProvider:
    """Generates candidates by adding Gaussian noise to observed embeddings.

    Operates entirely in embedding space; no external service required.

    Environment variables:
      PARETO_BO_CANDIDATE_NOISE_SCALE  float, default 0.1
      PARETO_BO_CANDIDATE_COUNT        int, default 16
    """

    def __init__(
        self,
        noise_scale: float | None = None,
        n_candidates: int | None = None,
    ) -> None:
        self.noise_scale = noise_scale
        self.n_candidates = n_candidates

    def _resolved_noise_scale(self) -> float:
        if self.noise_scale is not None:
            return float(self.noise_scale)
        return float(os.environ.get(_NOISE_SCALE_ENV, "0.1"))

    def _resolved_n_candidates(self) -> int:
        if self.n_candidates is not None:
            return int(self.n_candidates)
        return int(os.environ.get(_N_CANDIDATES_ENV, "16"))

    def propose(self, state: dict) -> torch.Tensor:
        observed = state.get("observed_embeddings")
        if observed is None:
            raise ValueError("TangentSpaceNoiseCandidateProvider requires observed_embeddings")
        observed_tensor = (
            observed
            if isinstance(observed, torch.Tensor)
            else torch.tensor(observed, dtype=torch.float32)
        ).float()
        if observed_tensor.ndim == 1:
            observed_tensor = observed_tensor.unsqueeze(0)
        n = self._resolved_n_candidates()
        noise_scale = self._resolved_noise_scale()
        idx = torch.randint(0, observed_tensor.shape[0], (n,))
        base = observed_tensor.index_select(0, idx)
        noise = torch.randn_like(base) * noise_scale
        return base + noise

    @classmethod
    def from_env(cls) -> TangentSpaceNoiseCandidateProvider:
        return cls()


class SmilesCandidateProvider:
    """Candidate provider backed by a fixed SMILES list encoded via HUMU encoder or fingerprints.

    The provider encodes SMILES to embeddings once per session and maintains an
    index-to-SMILES registry so that LocalOracleEvaluator can decode candidates back
    to SMILES for oracle evaluation.

    Environment variables:
      PARETO_BO_SMILES_LIST_JSON  JSON file path containing a list of SMILES strings
    """

    def __init__(self, smiles_list: list[str]) -> None:
        if not smiles_list:
            raise ValueError("SmilesCandidateProvider requires at least one SMILES")
        self.smiles_list = list(smiles_list)
        self._embeddings: torch.Tensor | None = None
        self._smiles_registry: dict[int, str] = {}

    @classmethod
    def from_env(cls) -> SmilesCandidateProvider:
        path = os.environ.get(_SMILES_LIST_ENV, "").strip()
        if not path:
            raise RuntimeError(
                f"{_SMILES_LIST_ENV} must point to a JSON file containing a SMILES list"
            )
        with open(path, encoding="utf-8") as fh:
            smiles_list = json.load(fh)
        if not isinstance(smiles_list, list):
            raise ValueError(f"{_SMILES_LIST_ENV}: expected a JSON list of SMILES strings")
        return cls(smiles_list)

    def propose(self, state: dict) -> torch.Tensor:
        if self._embeddings is None:
            self._embeddings = self._encode_smiles()
        self._smiles_registry = {i: s for i, s in enumerate(self.smiles_list)}
        return self._embeddings

    def smiles_for_index(self, candidate_index: int) -> str | None:
        return self._smiles_registry.get(int(candidate_index))

    def _encode_smiles(self) -> torch.Tensor:
        try:
            return self._encode_via_humu()
        except Exception:
            return self._encode_via_fingerprint()

    def _encode_via_humu(self) -> torch.Tensor:
        import grpc
        from mf_core.proto_gen.moleculeforge.v1.humu import humu_encoder_pb2, humu_encoder_pb2_grpc

        target = os.environ.get("HUMU_ENCODER_TARGET", "").strip()
        if not target:
            raise RuntimeError("HUMU_ENCODER_TARGET not configured")
        channel = grpc.insecure_channel(target)
        stub = humu_encoder_pb2_grpc.HumuEncoderServiceStub(channel)
        embeddings = []
        for smiles in self.smiles_list:
            response = stub.Encode(
                humu_encoder_pb2.EncodeRequest(input_type="molecule", smiles=smiles)
            )
            embeddings.append(list(response.humu_embedding))
        return torch.tensor(embeddings, dtype=torch.float32)

    def _encode_via_fingerprint(self) -> torch.Tensor:
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except ImportError as exc:
            raise RuntimeError(
                "SmilesCandidateProvider: rdkit is required when HUMU_ENCODER_TARGET is not set"
            ) from exc
        fps = []
        for smiles in self.smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                fps.append([0.0] * 2048)
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                fps.append(list(fp))
        return torch.tensor(fps, dtype=torch.float32)


class LocalOracleEvaluator:
    """Oracle evaluator that calls a local L0-L3 gRPC oracle service.

    When a SmilesCandidateProvider is shared as `smiles_provider`, the evaluator looks
    up the SMILES for each candidate index and calls the oracle gRPC service.

    Without SMILES lookup, falls back to embedding-distance proxy scores so the
    optimization loop can run locally without any oracle service.

    Environment variables:
      PARETO_BO_ORACLE_LEVEL        int (0-3), default 0
      L0_ADMET_ORACLE_TARGET        gRPC target for L0 oracle
      L1_DOCKING_ORACLE_TARGET      gRPC target for L1 oracle
      L2_AFFINITY_ORACLE_TARGET     gRPC target for L2 oracle
      L3_FEP_ORACLE_TARGET          gRPC target for L3 oracle
      PARETO_BO_ORACLE_PROPERTIES   comma-separated property names to request
    """

    def __init__(
        self,
        smiles_provider: SmilesCandidateProvider | None = None,
        oracle_level: int | None = None,
        oracle_properties: list[str] | None = None,
    ) -> None:
        self.smiles_provider = smiles_provider
        self.oracle_level = oracle_level
        self.oracle_properties = oracle_properties or []

    @classmethod
    def from_env(cls) -> LocalOracleEvaluator:
        return cls()

    def _resolved_level(self) -> int:
        if self.oracle_level is not None:
            return int(self.oracle_level)
        return int(os.environ.get(_ORACLE_LEVEL_ENV, "0"))

    def _resolved_properties(self) -> list[str]:
        if self.oracle_properties:
            return self.oracle_properties
        props_str = os.environ.get(_ORACLE_PROPERTIES_ENV, "").strip()
        if props_str:
            return [p.strip() for p in props_str.split(",") if p.strip()]
        return []

    def _oracle_target(self) -> str:
        level = self._resolved_level()
        env_var = _ORACLE_TARGET_ENVS.get(level, _ORACLE_TARGET_ENVS[0])
        return os.environ.get(env_var, "").strip()

    async def __call__(self, request: dict) -> dict[str, Any]:
        smiles = self._smiles_for_request(request)
        if smiles and self._oracle_target():
            return await self._evaluate_via_grpc(smiles)
        return self._embedding_proxy_scores(request)

    def _smiles_for_request(self, request: dict) -> str | None:
        smiles = request.get("candidate_smiles")
        if smiles:
            return str(smiles)
        if self.smiles_provider is not None:
            idx = request.get("candidate_index")
            if idx is not None:
                return self.smiles_provider.smiles_for_index(int(idx))
        return None

    async def _evaluate_via_grpc(self, smiles: str) -> dict[str, Any]:
        import grpc
        from mf_core.proto_gen.moleculeforge.v1.oracle import oracle_pb2, oracle_pb2_grpc

        target = self._oracle_target()
        level = self._resolved_level()
        properties = self._resolved_properties()
        level_proto = _oracle_level_proto(level)

        channel = grpc.aio.insecure_channel(target)
        stub = oracle_pb2_grpc.OracleServiceStub(channel)
        response = await stub.Evaluate(
            oracle_pb2.OracleBatchRequest(
                molecule_smiles=[smiles],
                level=level_proto,
                requested_properties=properties,
            )
        )
        evaluations = list(response.evaluations)
        if not evaluations:
            raise RuntimeError(f"Oracle returned no evaluations for SMILES: {smiles}")
        ev = evaluations[0]
        scores = dict(ev.scores)
        objectives = [float(v) for v in scores.values()] if scores else [0.0, 0.0]
        if len(objectives) < 2:
            objectives = objectives + [0.0] * (2 - len(objectives))
        constraint_names = [k for k in scores if "constraint" in k.lower()]
        constraints = [float(scores[k]) for k in constraint_names] if constraint_names else [0.5]
        return {
            "objectives": objectives[:2],
            "constraints": constraints[:1],
            "oracle_name": str(ev.oracle_name),
            "smiles": smiles,
            "scores": scores,
        }

    def _embedding_proxy_scores(self, request: dict) -> dict[str, Any]:
        embedding = request.get("candidate_embedding", [])
        tensor = torch.tensor(embedding, dtype=torch.float32) if embedding else torch.zeros(1)
        norm = float(tensor.norm().item())
        mean_val = float(tensor.mean().item())
        return {
            "objectives": [norm, mean_val],
            "constraints": [0.5],
            "source": "embedding_proxy",
        }


def _oracle_level_proto(level: int) -> int:
    from mf_core.proto_gen.moleculeforge.v1.oracle import oracle_pb2

    mapping = {
        0: oracle_pb2.L0_RDKIT,
        1: oracle_pb2.L1_ML_SURROGATE,
        2: oracle_pb2.L2_DOCKING,
        3: oracle_pb2.L3_FEP,
    }
    return mapping.get(level, oracle_pb2.L0_RDKIT)
