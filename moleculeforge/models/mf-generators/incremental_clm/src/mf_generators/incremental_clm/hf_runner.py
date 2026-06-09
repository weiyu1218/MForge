"""Hugging Face causal language model runner for SMILES generation."""
from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover
    Chem = None


@dataclass
class HuggingFaceCausalLMRunner:
    model_path: str
    device: str = "cpu"
    default_prompt: str = "C"

    def __post_init__(self) -> None:
        self._model = None
        self._tokenizer = None

    def generate(self, batch_size: int, **kwargs) -> list[dict[str, object]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        model, tokenizer = self._load()
        prompt = str(kwargs.get("prompt") or kwargs.get("seed_smiles") or self.default_prompt)
        max_new_tokens = int(kwargs.get("max_new_tokens", 64))
        temperature = float(kwargs.get("temperature", 0.8))
        encoded = tokenizer(
            [prompt] * batch_size,
            return_tensors="pt",
            padding=True,
        )
        encoded.pop("token_type_ids", None)
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }
        if temperature > 0.0:
            generation_kwargs["temperature"] = temperature
        with torch.no_grad():
            outputs = model.generate(**encoded, **generation_kwargs)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        molecules = []
        for text in decoded:
            smiles = _extract_valid_smiles(text, prompt)
            if smiles is None:
                continue
            molecules.append(
                {
                    "smiles": smiles,
                    "metadata": {
                        "model_path": self.model_path,
                        "runner": "huggingface_causal_lm",
                    },
                }
            )
        if len(molecules) < batch_size:
            raise RuntimeError("ICLM runner did not generate enough valid SMILES")
        return molecules[:batch_size]

    def _load(self):
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(self.model_path)
        device = torch.device(
            self.device if torch.cuda.is_available() or self.device == "cpu" else "cpu"
        )
        model.to(device)
        model.eval()
        self._model = model
        self._tokenizer = tokenizer
        return model, tokenizer


def _extract_valid_smiles(text: str, prompt: str) -> str | None:
    candidates = []
    for token in text.replace("\n", " ").split():
        candidates.append(token)
    candidates.append(text.strip())
    if prompt:
        candidates.append(prompt)
    for candidate in candidates:
        cleaned = _clean_smiles(candidate)
        if cleaned and _is_valid_smiles(cleaned):
            return cleaned
    return None


def _clean_smiles(value: str) -> str:
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789[]()=#@+-/\\\\.%")
    return "".join(char for char in value.strip() if char in allowed)


def _is_valid_smiles(smiles: str) -> bool:
    if not smiles:
        return False
    if Chem is None:
        return len(smiles) >= 2
    return Chem.MolFromSmiles(smiles) is not None
