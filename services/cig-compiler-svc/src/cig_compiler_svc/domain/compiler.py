"""CIGCompiler — main compiler (async interface)."""
from __future__ import annotations

import importlib
import inspect
import os
from collections.abc import Callable
from enum import StrEnum
from typing import Any

from mf_core.types.cig import ChemicalIntentGraph
from mf_core.types.humu import HCIV, IntentCone

from cig_compiler_svc.domain.hciv_encoder import (
    hash_encode_hciv,
    load_hciv_encoder_checkpoint,
)
from cig_compiler_svc.domain.hciv_generator import generate_intent_cone, generate_random_hciv
from cig_compiler_svc.domain.stages.stage1_semantic import _heuristic_extract
from cig_compiler_svc.domain.stages.stage1b_grounding import (
    ALL_GROUNDING_SOURCES,
    GroundingSource,
    ground_knowledge,
)
from cig_compiler_svc.domain.stages.stage2_cig_build import build_cig


class EncodingMode(StrEnum):
    LEARNED = "learned"
    HASH = "hash"
    RANDOM = "random"


class CompilerMode(StrEnum):
    PRODUCTION_REAL = "production_real"
    LOCAL_DEMO = "local_demo"


SemanticParser = Callable[[str], dict[str, Any]]


class ProductionSemanticParserAdapter:
    def __init__(self, uri: str | None = None) -> None:
        self.uri = uri
        self._parser: SemanticParser | None = None

    def __call__(self, nl_text: str) -> dict[str, Any]:
        parser = self._load_parser()
        parsed = parser(nl_text)
        if inspect.isawaitable(parsed):
            return parsed
        if not isinstance(parsed, dict):
            raise RuntimeError("production semantic parser must return a dict")
        return parsed

    def _load_parser(self) -> SemanticParser:
        if self._parser is not None:
            return self._parser
        uri = self.uri or os.environ.get("CIG_SEMANTIC_PARSER_URI")
        if not uri:
            raise RuntimeError(
                "CIG_SEMANTIC_PARSER_URI is required for production_real "
                "semantic parsing"
            )

        target = uri.removeprefix("python://").removeprefix("python:")
        if ":" not in target:
            raise RuntimeError(
                "CIG_SEMANTIC_PARSER_URI must reference a Python callable as "
                "module:function"
            )
        module_name, function_name = target.split(":", 1)
        module = importlib.import_module(module_name)
        parser = getattr(module, function_name, None)
        if not callable(parser):
            raise RuntimeError(
                f"CIG_SEMANTIC_PARSER_URI target is not callable: {uri}"
            )
        self._parser = parser
        return parser


class CIGCompiler:
    def __init__(
        self,
        encoding_mode: str | EncodingMode | None = None,
        hciv_dim: int = 128,
        learned_encoder: object | None = None,
        semantic_parser: SemanticParser | None = None,
        enable_grounding: bool = True,
        grounding_sources: tuple[GroundingSource, ...] | None = None,
        mode: str | CompilerMode = CompilerMode.PRODUCTION_REAL,
    ) -> None:
        self.mode = CompilerMode(mode)
        default_encoding = (
            EncodingMode.HASH
            if self.mode == CompilerMode.LOCAL_DEMO
            else EncodingMode.LEARNED
        )
        self.encoding_mode = EncodingMode(encoding_mode or default_encoding)
        self._validate_mode()
        self.hciv_dim = hciv_dim
        self.learned_encoder = learned_encoder
        self.semantic_parser = self._resolve_semantic_parser(semantic_parser)
        self.enable_grounding = enable_grounding
        self.grounding_sources = grounding_sources or self._default_grounding_sources()

    def _validate_mode(self) -> None:
        if (
            self.mode == CompilerMode.PRODUCTION_REAL
            and self.encoding_mode in (EncodingMode.HASH, EncodingMode.RANDOM)
        ):
            raise ValueError(
                f"encoding_mode='{self.encoding_mode.value}' is only allowed "
                "in local_demo mode"
            )

    def _resolve_semantic_parser(
        self,
        semantic_parser: SemanticParser | None,
    ) -> SemanticParser:
        if semantic_parser is not None:
            return semantic_parser
        if self.mode == CompilerMode.LOCAL_DEMO:
            return _heuristic_extract
        return ProductionSemanticParserAdapter()

    def _default_grounding_sources(self) -> tuple[GroundingSource, ...]:
        if self.mode == CompilerMode.PRODUCTION_REAL:
            return ALL_GROUNDING_SOURCES
        return ("uniprot",)

    async def compile(
        self,
        nl_text: str,
        seed: int | None = None,
    ) -> tuple[ChemicalIntentGraph, HCIV, IntentCone]:
        parsed = self.semantic_parser(nl_text)
        extracted = await parsed if inspect.isawaitable(parsed) else parsed
        if self.enable_grounding:
            extracted = await ground_knowledge(extracted, sources=self.grounding_sources)
        cig = build_cig(extracted, source=nl_text)
        hciv, encoder_cone = self._encode_hciv(cig, seed=seed)
        cone = encoder_cone or generate_intent_cone(hciv, dim=self.hciv_dim, seed=seed)
        return cig, hciv, cone

    def _encode_hciv(
        self,
        cig: ChemicalIntentGraph,
        seed: int | None,
    ) -> tuple[HCIV, IntentCone | None]:
        if self.encoding_mode == EncodingMode.LEARNED:
            encoder = self._resolve_learned_encoder()
            encoded = encoder.encode(cig)
            if isinstance(encoded, tuple):
                hciv, cone = encoded
                return hciv, cone
            return encoded, None
        elif self.encoding_mode == EncodingMode.HASH:
            return hash_encode_hciv(cig, dim=self.hciv_dim, seed=seed or 42), None
        elif self.encoding_mode == EncodingMode.RANDOM:
            return generate_random_hciv(self.hciv_dim, seed=seed), None
        else:
            raise ValueError(f"Unknown encoding_mode: {self.encoding_mode}")

    def _resolve_learned_encoder(self):
        if self.learned_encoder is not None:
            return self.learned_encoder
        if self.mode == CompilerMode.PRODUCTION_REAL:
            checkpoint_path = os.environ.get("HCIV_CHECKPOINT_PATH")
            if not checkpoint_path:
                raise RuntimeError(
                    "HCIV_CHECKPOINT_PATH is required for production_real "
                    "learned HCIV encoding"
                )
            self.learned_encoder = load_hciv_encoder_checkpoint(
                checkpoint_path,
                dim=self.hciv_dim,
            )
            return self.learned_encoder
        raise RuntimeError(
            "learned HCIV encoder is required when encoding_mode='learned'"
        )
