"""Fragment vocabulary for FragFM."""
from __future__ import annotations


class FragmentVocabulary:
    def __init__(self, fragment_smiles_list: list[str] | None = None):
        self.fragments = fragment_smiles_list or [
            "*C*",
            "*CC*",
            "*c1ccccc1*",
            "*C(=O)*",
            "*CN*",
            "*O*",
            "*Cl*",
            "*Br*",
        ]
        self._frag_to_idx = {f: i for i, f in enumerate(self.fragments)}

    def encode(self, fragment_smiles: str) -> int:
        return self._frag_to_idx.get(fragment_smiles, 0)

    def decode(self, idx: int) -> str:
        return self.fragments[idx] if 0 <= idx < len(self.fragments) else ""

    def contains(self, fragment_smiles: str) -> bool:
        return fragment_smiles in self._frag_to_idx

    def __len__(self) -> int:
        return len(self.fragments)
