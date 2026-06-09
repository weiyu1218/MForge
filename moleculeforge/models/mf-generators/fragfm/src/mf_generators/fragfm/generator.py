"""FragFM: Fragment-based discrete flow matching for molecular generation."""
from __future__ import annotations

import inspect
import json
from collections.abc import Mapping
from pathlib import Path

import torch
from mf_core.plugins.generator import GeneratorPlugin
from mf_core.types.humu import IntentCone
from mf_core.types.molecule import Molecule
from mf_generators.fragfm.model.fragment_vocabulary import FragmentVocabulary
from mf_generators.fragfm.model.sa_aware_rate_matrix import SAAwareRateMatrix
from mf_generators.fragfm.model.two_level_dfm import TwoLevelDFM

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover
    Chem = None


class FragFMGenerator(GeneratorPlugin):
    name = "fragfm"

    def __init__(
        self,
        checkpoint_path: str = "",
        device: str = "cpu",
        vocab_path: str = "",
        rate_matrix_path: str = "",
        mode: str = "production_real",
        model=None,
        rate_matrix=None,
        decoder=None,
        humu_latent_sampler=None,
    ):
        if mode not in {"production_real", "local_demo"}:
            raise ValueError(f"Unknown FragFMGenerator mode: {mode}")
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.vocab_path = vocab_path
        self.rate_matrix_path = rate_matrix_path
        self.mode = mode
        self._model = model
        self.decoder = decoder
        self.humu_latent_sampler = humu_latent_sampler
        self._scored_rule_cache: list[tuple[dict[str, object], list[int], float]] | None = None
        if vocab_path:
            self.vocab, self.assembly_rules = self._load_artifact(vocab_path)
        elif mode == "local_demo":
            self.vocab = FragmentVocabulary(["CC", "O"])
            self.assembly_rules = [
                {"id": "local_demo_ethanol", "fragments": ["CC", "O"], "product": "CCO"}
            ]
        else:
            raise RuntimeError(
                "FragFM production generation requires a vocabulary artifact"
            )
        self.rate_matrix = rate_matrix or SAAwareRateMatrix(vocab_size=len(self.vocab))
        if rate_matrix_path:
            path = Path(rate_matrix_path)
            if not path.exists():
                raise FileNotFoundError(
                    f"FragFM rate matrix artifact not found: {rate_matrix_path}"
                )
            state = torch.load(path, map_location=device, weights_only=True)
            self._validate_rate_matrix_state(state)
            self.rate_matrix.load_state_dict(state, strict=False)
        self.rate_matrix.to(device)
        if self._model is None and checkpoint_path:
            path = Path(checkpoint_path)
            if not path.exists():
                raise FileNotFoundError(
                    f"FragFM checkpoint artifact not found: {checkpoint_path}"
                )
            self._model = self._load_model_from_checkpoint(checkpoint_path, device)

    def _load_model_from_checkpoint(self, checkpoint_path: str, device: str):
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self._validate_checkpoint_state(state)
        hidden_dim = self._checkpoint_hidden_dim(state)
        model = TwoLevelDFM(vocab_size=len(self.vocab), hidden_dim=hidden_dim)
        model.load_state_dict(state, strict=False)
        model.to(device)
        return model

    def _validate_checkpoint_state(self, state: object) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("FragFM checkpoint must load to a state-dict mapping")
        weight = state.get("fragment_encoder.weight")
        if weight is None or not hasattr(weight, "shape"):
            raise ValueError("FragFM checkpoint requires fragment_encoder.weight")
        shape = tuple(int(dim) for dim in weight.shape)
        if len(shape) != 2 or shape[1] <= 0:
            raise ValueError(
                "FragFM checkpoint fragment_encoder.weight must be a 2D tensor "
                "with positive hidden dimension"
            )
        if shape[0] != len(self.vocab):
            raise ValueError(
                "FragFM checkpoint fragment vocabulary size "
                f"{shape[0]} does not match vocab artifact {len(self.vocab)}"
            )

    def _checkpoint_hidden_dim(self, state: Mapping[str, object]) -> int:
        weight = state.get("fragment_encoder.weight")
        if weight is None or not hasattr(weight, "shape"):
            return 256
        shape = tuple(int(dim) for dim in weight.shape)
        if len(shape) != 2 or shape[1] <= 0:
            return 256
        return shape[1]

    def _validate_rate_matrix_state(self, state: object) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("FragFM rate matrix must load to a state-dict mapping")
        base_rate = state.get("base_rate")
        if base_rate is None or not hasattr(base_rate, "shape"):
            raise ValueError("FragFM rate matrix requires base_rate")
        expected_base_shape = (len(self.vocab), len(self.vocab))
        base_shape = tuple(int(dim) for dim in base_rate.shape)
        if base_shape != expected_base_shape:
            raise ValueError(
                "FragFM rate matrix base_rate shape "
                f"{base_shape} does not match vocab artifact {expected_base_shape}"
            )
        sa_embedding = state.get("sa_score_embedding.weight")
        if sa_embedding is None or not hasattr(sa_embedding, "shape"):
            raise ValueError("FragFM rate matrix requires sa_score_embedding.weight")
        expected_sa_shape = (10, len(self.vocab) * len(self.vocab))
        sa_shape = tuple(int(dim) for dim in sa_embedding.shape)
        if sa_shape != expected_sa_shape:
            raise ValueError(
                "FragFM rate matrix sa_score_embedding.weight shape "
                f"{sa_shape} does not match expected {expected_sa_shape}"
            )

    def _load_artifact(self, vocab_path: str) -> tuple[FragmentVocabulary, list[dict[str, object]]]:
        path = Path(vocab_path)
        if not path.exists():
            raise FileNotFoundError(f"FragFM vocabulary artifact not found: {vocab_path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("FragFM vocabulary artifact must be a JSON object")
        fragments = payload.get("fragments")
        rules = payload.get("assembly_rules")
        if not isinstance(fragments, list) or not fragments:
            raise ValueError("FragFM vocabulary artifact requires fragments")
        if not isinstance(rules, list) or not rules:
            raise ValueError("FragFM vocabulary artifact requires assembly_rules")
        vocab = FragmentVocabulary([str(fragment) for fragment in fragments])
        assembly_rules = [
            self._normalize_rule(idx, rule, vocab)
            for idx, rule in enumerate(rules)
        ]
        return vocab, assembly_rules

    def _normalize_rule(
        self,
        idx: int,
        rule: object,
        vocab: FragmentVocabulary,
    ) -> dict[str, object]:
        if not isinstance(rule, Mapping):
            raise ValueError("FragFM assembly rule must be a JSON object")
        fragments = rule.get("fragments")
        product = rule.get("product")
        if not isinstance(fragments, list) or not fragments:
            raise ValueError("FragFM assembly rule requires fragments")
        if not isinstance(product, str) or not product:
            raise ValueError("FragFM assembly rule requires product")
        rule_fragments = [str(fragment) for fragment in fragments]
        missing = [fragment for fragment in rule_fragments if not vocab.contains(fragment)]
        if missing:
            raise ValueError(f"FragFM assembly rule references unknown fragments: {missing}")
        canonical_product = self._canonical_smiles(product)
        sa_score_bin = int(rule.get("sa_score_bin", 5))
        if not 0 <= sa_score_bin <= 9:
            raise ValueError("FragFM assembly rule sa_score_bin must be in [0, 9]")
        normalized_rule: dict[str, object] = {
            "id": str(rule.get("id", idx)),
            "fragments": rule_fragments,
            "product": canonical_product,
            "sa_score_bin": sa_score_bin,
        }
        humu_embedding = rule.get("humu_embedding")
        if humu_embedding is not None:
            if not isinstance(humu_embedding, list) or not humu_embedding:
                raise ValueError("FragFM assembly rule humu_embedding must be a non-empty list")
            if not all(isinstance(value, int | float) for value in humu_embedding):
                raise ValueError("FragFM assembly rule humu_embedding must contain numbers")
            normalized_rule["humu_embedding"] = [float(value) for value in humu_embedding]
        return normalized_rule

    def _canonical_smiles(self, smiles: str) -> str:
        if Chem is None:
            raise ImportError("RDKit is required for FragFM validity checks")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"FragFM assembly rule produced invalid SMILES: {smiles}")
        return Chem.MolToSmiles(mol)

    async def generate(
        self,
        batch_size: int,
        intent_cone: IntentCone | None = None,
        **kwargs,
    ) -> list[Molecule]:
        """Generate molecules via two-level discrete flow matching with fragment assembly."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        humu_latents = await self._sample_humu_latents(
            batch_size=batch_size,
            intent_cone=intent_cone,
        )
        ranked_rules = self._rank_rules(intent_cone=intent_cone)
        results = []
        for i in range(batch_size):
            humu_latent = None
            if humu_latents is not None:
                humu_latent = humu_latents[i]
                ranked_rules = self._rank_rules(
                    intent_cone=intent_cone,
                    humu_latent=humu_latent,
                )
            rule, fragment_indices, _transition_score, condition_score = ranked_rules[
                i % len(ranked_rules)
            ]
            smiles = str(rule["product"])
            if self._model is not None and self.decoder is not None:
                smiles = self._decode_with_model(fragment_indices, rule)
            metadata = {
                "generator_name": self.name,
                "fragment_vocabulary": self.vocab_path,
                "assembly_rule_id": str(rule["id"]),
                "rate_matrix_applied": "true",
                "fragment_indices": ",".join(str(idx) for idx in fragment_indices),
                "humu_condition_score": str(condition_score),
            }
            if humu_latent is not None:
                metadata["humu_latent"] = ",".join(str(float(value)) for value in humu_latent)
            results.append(
                Molecule(
                    smiles=smiles,
                    metadata=metadata,
                )
            )
        return results

    def _rank_rules(
        self,
        intent_cone: IntentCone | None = None,
        humu_latent: list[float] | None = None,
    ) -> list[tuple[dict[str, object], list[int], float, float]]:
        ranked = []
        for rule, fragment_indices, transition_score in self._scored_rules():
            condition_score = self._intent_rule_alignment(
                intent_cone,
                rule,
                humu_latent=humu_latent,
            )
            ranked.append((rule, fragment_indices, transition_score, condition_score))
        return sorted(ranked, key=lambda item: item[2] + item[3], reverse=True)

    def _scored_rules(self) -> list[tuple[dict[str, object], list[int], float]]:
        if self._scored_rule_cache is not None:
            return self._scored_rule_cache
        encoded_rules = [
            (
                rule,
                [self.vocab.encode(fragment) for fragment in rule["fragments"]],
            )
            for rule in self.assembly_rules
        ]
        transition_scores = self._transition_scores(encoded_rules)
        self._scored_rule_cache = sorted(
            (
                (rule, fragment_indices, transition_score)
                for (rule, fragment_indices), transition_score in zip(
                    encoded_rules,
                    transition_scores,
                    strict=True,
                )
            ),
            key=lambda item: item[2],
            reverse=True,
        )
        return self._scored_rule_cache

    def _transition_scores(
        self,
        encoded_rules: list[tuple[dict[str, object], list[int]]],
    ) -> list[float]:
        if hasattr(self.rate_matrix, "base_rate") and hasattr(
            self.rate_matrix,
            "sa_score_embedding",
        ):
            return self._sparse_transition_scores(encoded_rules)
        return [
            self._transition_score(
                fragment_indices,
                sa_score_bin=int(rule.get("sa_score_bin", 5)),
            )
            for rule, fragment_indices in encoded_rules
        ]

    def _sparse_transition_scores(
        self,
        encoded_rules: list[tuple[dict[str, object], list[int]]],
    ) -> list[float]:
        scores = [0.0] * len(encoded_rules)
        left_values: list[int] = []
        right_values: list[int] = []
        sa_bins: list[int] = []
        rule_indices: list[int] = []
        for rule_index, (rule, fragment_indices) in enumerate(encoded_rules):
            if len(fragment_indices) < 2:
                continue
            sa_score_bin = max(0, min(9, int(rule.get("sa_score_bin", 5))))
            for left, right in zip(fragment_indices, fragment_indices[1:], strict=False):
                left_values.append(left)
                right_values.append(right)
                sa_bins.append(sa_score_bin)
                rule_indices.append(rule_index)
        if not left_values:
            return scores

        base_rate = self.rate_matrix.base_rate
        sa_weights = self.rate_matrix.sa_score_embedding.weight
        device = base_rate.device
        vocab_size = int(base_rate.shape[0])
        left_indices = torch.tensor(left_values, dtype=torch.long, device=device)
        right_indices = torch.tensor(right_values, dtype=torch.long, device=device)
        bin_indices = torch.tensor(sa_bins, dtype=torch.long, device=device)
        target_rules = torch.tensor(rule_indices, dtype=torch.long, device=device)
        flat_indices = left_indices * vocab_size + right_indices
        with torch.no_grad():
            base_values = base_rate[left_indices, right_indices]
            modulation = sa_weights[bin_indices, flat_indices]
            values = base_values * (1 + torch.tanh(modulation))
            accumulated = torch.zeros(len(encoded_rules), dtype=values.dtype, device=device)
            accumulated.index_add_(0, target_rules, values)
        return [float(value) for value in accumulated.detach().cpu().tolist()]

    def _transition_score(
        self,
        fragment_indices: list[int],
        *,
        sa_score_bin: int,
    ) -> float:
        if len(fragment_indices) < 2:
            return 0.0
        if hasattr(self.rate_matrix, "base_rate") and hasattr(
            self.rate_matrix,
            "sa_score_embedding",
        ):
            return self._sparse_transition_score(fragment_indices, sa_score_bin)
        sa_score = torch.tensor(
            [sa_score_bin],
            dtype=torch.long,
            device=self.device,
        )
        rate_matrix = self.rate_matrix(sa_score)
        transition_score = 0.0
        for left, right in zip(fragment_indices, fragment_indices[1:], strict=False):
            transition_score += float(rate_matrix[0, left, right].detach().cpu().item())
        return transition_score

    def _sparse_transition_score(
        self,
        fragment_indices: list[int],
        sa_score_bin: int,
    ) -> float:
        base_rate = self.rate_matrix.base_rate
        sa_weights = self.rate_matrix.sa_score_embedding.weight
        vocab_size = int(base_rate.shape[0])
        left_indices = torch.tensor(
            fragment_indices[:-1],
            dtype=torch.long,
            device=base_rate.device,
        )
        right_indices = torch.tensor(
            fragment_indices[1:],
            dtype=torch.long,
            device=base_rate.device,
        )
        flat_indices = left_indices * vocab_size + right_indices
        sa_index = max(0, min(9, int(sa_score_bin)))
        with torch.no_grad():
            base_values = base_rate[left_indices, right_indices]
            modulation = sa_weights[sa_index, flat_indices]
            values = base_values * (1 + torch.tanh(modulation))
            return float(values.sum().detach().cpu().item())

    def _intent_rule_alignment(
        self,
        intent_cone: IntentCone | None,
        rule: dict[str, object],
        humu_latent: list[float] | None = None,
    ) -> float:
        if humu_latent is not None:
            return self._condition_vector_rule_alignment(humu_latent, rule)
        if intent_cone is None:
            return 0.0
        return self._condition_vector_rule_alignment(intent_cone.axis, rule)

    def _condition_vector_rule_alignment(
        self,
        condition_vector: list[float],
        rule: dict[str, object],
    ) -> float:
        embedding = rule.get("humu_embedding")
        if not isinstance(embedding, list) or not embedding:
            return 0.0
        axis = condition_vector
        if len(axis) == len(embedding) + 1:
            axis = axis[1:]
        dim = min(len(axis), len(embedding))
        if dim == 0:
            return 0.0
        axis_vector = torch.tensor(axis[:dim], dtype=torch.float32)
        embedding_vector = torch.tensor(embedding[:dim], dtype=torch.float32)
        axis_norm = torch.linalg.vector_norm(axis_vector)
        embedding_norm = torch.linalg.vector_norm(embedding_vector)
        if float(axis_norm.item()) == 0.0 or float(embedding_norm.item()) == 0.0:
            return 0.0
        similarity = torch.dot(axis_vector, embedding_vector) / (axis_norm * embedding_norm)
        return float(similarity.detach().cpu().item())

    async def _sample_humu_latents(
        self,
        *,
        batch_size: int,
        intent_cone: IntentCone | None,
    ) -> list[list[float]] | None:
        sampler = self.humu_latent_sampler
        if sampler is None:
            return None
        if hasattr(sampler, "sample"):
            latents = sampler.sample(batch_size=batch_size, intent_cone=intent_cone)
        elif callable(sampler):
            latents = sampler(batch_size=batch_size, intent_cone=intent_cone)
        else:
            raise TypeError("FragFM HUMU latent sampler must be callable or expose sample()")
        if inspect.isawaitable(latents):
            latents = await latents
        if latents is None:
            return None
        if isinstance(latents, torch.Tensor):
            latents = latents.detach().cpu().tolist()
        if not isinstance(latents, list) or len(latents) != batch_size:
            raise ValueError("FragFM HUMU latent sampler must return one latent per sample")
        normalized = []
        for latent in latents:
            if not isinstance(latent, list) or not latent:
                raise ValueError("FragFM HUMU latent sampler returned an invalid latent")
            if not all(isinstance(value, int | float) for value in latent):
                raise ValueError("FragFM HUMU latent values must be numeric")
            normalized.append([float(value) for value in latent])
        return normalized

    def _decode_with_model(self, fragment_indices: list[int], rule: dict[str, object]) -> str:
        fragment_ids = torch.tensor([fragment_indices], dtype=torch.long, device=self.device)
        molecule_ids = torch.zeros(
            (1, max(1, len(fragment_indices)), self._model.fragment_encoder.embedding_dim),
            dtype=torch.float32,
            device=self.device,
        )
        logits = self._model(fragment_ids, molecule_ids)
        decoded = self.decoder(logits, rule=rule, vocab=self.vocab)
        if inspect.isawaitable(decoded):
            raise RuntimeError("FragFM decoder must be synchronous")
        if isinstance(decoded, str):
            return self._canonical_smiles(decoded)
        if isinstance(decoded, list) and decoded:
            return self._canonical_smiles(str(decoded[0]))
        raise ValueError("FragFM decoder returned no SMILES")

    async def info(self) -> dict:
        return {
            "name": "fragfm",
            "version": "0.1.0",
            "description": "Two-level Discrete Flow Matching for fragment-based generation",
            "supported_properties": ["qed", "sa_score", "mw", "logp"],
            "max_batch_size": 512,
            "supports_streaming": True,
            "requires_gpu": False,
        }
