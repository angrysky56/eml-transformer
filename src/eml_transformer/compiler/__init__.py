"""EML-compiled transformer: a tiny computer inside a transformer.

This package analytically compiles an EML expression tree into the weights
of a small decoder-only transformer. No training is performed — given an
EML tree from the catalog, ``compile_tree`` returns a PyTorch module whose
forward pass computes the exact function encoded by that tree, verifiable
to machine epsilon at the ``primitives.TEST_POINTS``.

Design overview
---------------
Each RPN token in the EML stream occupies one sequence position. After
layer processing, position ``i``'s residual stream holds the complex value
computed at the EML node corresponding to that token.

* **Leaf tokens** (``1``, ``x``, ``y``, constant) have their value injected
  at position ``i`` by a layer-0 "injection" FFN reading from the token
  embedding.
* **EML tokens** (``E``) at position ``i`` use attention to route the
  already-computed values from the two child positions ``(j_left,
  j_right)`` into position ``i``'s hidden state, then the FFN applies
  the EML operator ``exp(a) - ln(b)``.

The attention pattern for every position is set analytically at compile
time from the tree's structure — there is no learned attention and no
training step.

See :mod:`eml_transformer.compiler.compiler` for the construction itself
and :mod:`eml_transformer.compiler.machine` for the minimal transformer
architecture it targets.
"""

from eml_transformer.compiler.catalog import (
    TEST_POINTS,
    CatalogEntry,
    load_catalog,
    load_entry,
    signature_bindings,
)
from eml_transformer.compiler.composer import (
    CalleeSpec,
    CallResolutionError,
    build_registry,
    compose,
    expand_calls,
    parse_and_expand,
    substitute_vars,
)
from eml_transformer.compiler.machine import (
    CompiledNode,
    CompiledProgram,
    EMLMachine,
    MachineConfig,
    compile_tree,
)
from eml_transformer.compiler.rpn import (
    EMLNode,
    RPNParseError,
    Token,
    TokenKind,
    parse_rpn_to_tree,
    tokenize_rpn,
    tree_to_rpn,
)
from eml_transformer.compiler.verify import (
    EntryResult,
    PointResult,
    format_summary,
    verify_catalog,
    verify_entry,
)

__all__ = [
    "CallResolutionError",
    "CalleeSpec",
    "CatalogEntry",
    "CompiledNode",
    "CompiledProgram",
    "EMLMachine",
    "EMLNode",
    "EntryResult",
    "MachineConfig",
    "PointResult",
    "RPNParseError",
    "TEST_POINTS",
    "Token",
    "TokenKind",
    "build_registry",
    "compile_tree",
    "compose",
    "expand_calls",
    "format_summary",
    "load_catalog",
    "load_entry",
    "parse_and_expand",
    "parse_rpn_to_tree",
    "signature_bindings",
    "substitute_vars",
    "tokenize_rpn",
    "tree_to_rpn",
    "verify_catalog",
    "verify_entry",
]
