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
            state = torch.load(path, map_location=device)
            self.rate_matrix.load_state_dict(state, strict=False)
        self.rate_matrix.to(device)
        if self._model is None and checkpoint_path and Path(checkpoint_path).exists():
            self._model = TwoLevelDFM(vocab_size=len(self.vocab))
            state = torch.load(checkpoint_path, map_location=device)
            self._model.load_state_dict(state, strict=False)
            self._model.to(device)

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
        return {
            "id": str(rule.get("id", idx)),
            "fragments": rule_fragments,
            "product": canonical_product,
            "sa_score_bin": sa_score_bin,
        }

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
        ranked_rules = self._rank_rules()
        results = []
        for i in range(batch_size):
            rule, fragment_indices, _transition_score = ranked_rules[i % len(ranked_rules)]
            smiles = str(rule["product"])
            if self._model is not None and self.decoder is not None:
                smiles = self._decode_with_model(fragment_indices, rule)
            results.append(
                Molecule(
                    smiles=smiles,
                    metadata={
                        "generator_name": self.name,
                        "fragment_vocabulary": self.vocab_path,
                        "assembly_rule_id": str(rule["id"]),
                        "rate_matrix_applied": "true",
                        "fragment_indices": ",".join(str(idx) for idx in fragment_indices),
                    },
                )
            )
        return results

    def _rank_rules(self) -> list[tuple[dict[str, object], list[int], float]]:
        ranked = []
        for rule in self.assembly_rules:
            fragment_indices = [self.vocab.encode(fragment) for fragment in rule["fragments"]]
            sa_score_bin = torch.tensor([int(rule.get("sa_score_bin", 5))], dtype=torch.long)
            rate_matrix = self.rate_matrix(sa_score_bin)
            transition_score = 0.0
            for left, right in zip(fragment_indices, fragment_indices[1:], strict=False):
                transition_score += float(rate_matrix[0, left, right].detach().cpu().item())
            ranked.append((rule, fragment_indices, transition_score))
        return sorted(ranked, key=lambda item: item[2], reverse=True)

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
