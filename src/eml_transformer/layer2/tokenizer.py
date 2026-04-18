from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eml_transformer.compiler.catalog import CatalogEntry


class Layer2Tokenizer:
    """Tokenizer for Layer 2 which treats catalog entries as atomic tokens.

    This tokenizer has a fixed vocabulary including:
    - Special tokens: <pad>, <bos>, <eos>, <unk>
    - Variables: x, y, z
    - Literal: 1.0
    - Operator: E
    - Catalog Names: All entries in the provided catalog.
    """

    def __init__(
        self, vocab: dict[str, int], catalog: list[CatalogEntry] | None = None
    ):
        self._vocab = vocab
        self._id_to_token = {v: k for k, v in vocab.items()}
        self.catalog = catalog or []

    @classmethod
    def from_catalog(cls, catalog: list[CatalogEntry]) -> Layer2Tokenizer:
        """Create a tokenizer from a list of catalog entries."""
        # Start with special tokens
        vocab = {
            "<pad>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "<unk>": 3,
        }

        # Add variables and fixed atoms
        # Order is deterministic to ensure stable vocab across runs if needed
        fixed_tokens = ["x", "y", "z", "1.0", "E"]
        for token in fixed_tokens:
            if token not in vocab:
                vocab[token] = len(vocab)

        # Add catalog names sorted for stability
        catalog_names = sorted(entry.name for entry in catalog)
        for name in catalog_names:
            if name not in vocab:
                vocab[name] = len(vocab)

        return cls(vocab, catalog)

    def parse_to_tree(self, rpn: str):
        """Helper to parse RPN using the tokenizer's catalog."""
        from eml_transformer.compiler import build_registry, parse_and_expand

        if not self.catalog:
            return None
        try:
            registry = build_registry(self.catalog)
            return parse_and_expand(rpn, registry)
        except Exception:
            return None

    def encode(self, rpn: str, *, add_special: bool = True) -> list[int]:
        """Convert an RPN string into a list of token IDs."""
        tokens = rpn.split()
        ids = []
        if add_special:
            ids.append(self.bos_id)

        for t in tokens:
            ids.append(self._vocab.get(t, self.unk_id))

        if add_special:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], *, strip_special: bool = True) -> str:
        """Convert a list of token IDs back into an RPN string."""
        tokens = []
        for i in ids:
            if strip_special and i in {self.pad_id, self.bos_id, self.eos_id}:
                continue
            token = self._id_to_token.get(i, "<unk>")
            tokens.append(token)
        return " ".join(tokens)

    @property
    def vocab_size(self) -> int:
        """Total number of unique tokens in the vocabulary."""
        return len(self._vocab)

    @property
    def pad_id(self) -> int:
        """ID of the <pad> token."""
        return self._vocab["<pad>"]

    @property
    def bos_id(self) -> int:
        """ID of the <bos> token."""
        return self._vocab["<bos>"]

    @property
    def eos_id(self) -> int:
        """ID of the <eos> token."""
        return self._vocab["<eos>"]

    @property
    def unk_id(self) -> int:
        """ID of the <unk> token."""
        return self._vocab["<unk>"]
