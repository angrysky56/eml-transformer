"""Dataset generation for Layer 2.

This module enumerates EML RPN programs up to a given composition depth
and computes their numerical signatures for training.
"""

from __future__ import annotations

import cmath
import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable

from eml_transformer.compiler import (
    TEST_POINTS,
    TokenKind,
    build_registry,
    load_catalog,
    parse_and_expand,
    signature_bindings,
)

if TYPE_CHECKING:
    from eml_transformer.compiler import CalleeSpec, EMLNode


@dataclass(frozen=True)
class SignatureProgramPair:
    """A numerical signature and its corresponding RPN program."""

    signature: tuple[complex, ...]
    rpn: str


@dataclass(frozen=True)
class GeneratorConfig:
    """Configuration for dataset generation."""

    max_composition_depth: int = 2
    max_args_per_call: int = 2
    exclude_entries: frozenset[str] = field(default_factory=frozenset)
    unique_signatures_only: bool = True
    seed: int = 0


def iter_composition_rpns(
    registry: dict[str, CalleeSpec],
    max_depth: int,
    include_variables: tuple[str, ...] = ("x", "y"),
) -> Iterable[str]:
    """Iterate over RPN programs up to max_depth.

    Depth 0: Variables (x, y).
    Depth 1: primitives called with variables.
    Depth 2: primitives called with at least one Depth 1 expression.
    """
    # Depth 0: Variables
    yield from include_variables
    if max_depth == 0:
        return

    # To generate higher depths, we need pools of expressions at each depth
    depth_pools: list[list[str]] = [list(include_variables)]

    for d in range(1, max_depth + 1):
        current_depth_rpns: list[str] = []
        
        # We can use any expression from [0...d-1] as an argument,
        # but at least one must be from pool [d-1].
        all_prev_rpns = list(itertools.chain.from_iterable(depth_pools))
        last_depth_pool = set(depth_pools[-1])

        # For each primitive in the registry
        for name, spec in registry.items():
            arity = spec.arity
            if arity == 0:
                # Constants or 0-arity functions are depth 1.
                if d == 1:
                    rpn = name
                    yield rpn
                    current_depth_rpns.append(rpn)
                continue

            # Generate all combinations of length 'arity'
            for args in itertools.product(all_prev_rpns, repeat=arity):
                # Check if at least one arg is from the previous depth pool
                if any(arg in last_depth_pool for arg in args):
                    # Construct RPN: "arg1 arg2 name"
                    rpn = " ".join(args) + f" {name}"
                    yield rpn
                    current_depth_rpns.append(rpn)
        
        depth_pools.append(current_depth_rpns)


def _safe_eml(a: complex, b: complex) -> complex:
    """EML operator with clamping and ln(0) handling matching EMLMachine."""
    # Match machine.py: EXP_CLAMP_MAX = 700.0, EXP_CLAMP_MIN = -700.0
    a_real = max(min(a.real, 700.0), -700.0)
    ea = cmath.exp(complex(a_real, a.imag))
    if b == 0j:
        lb = complex(float("-inf"), 0.0)
    else:
        lb = cmath.log(b)
    return ea - lb


def _evaluate_tree(node: EMLNode, bindings: dict[str, complex]) -> complex:
    """Recursive evaluator for EML trees, matching EMLMachine semantics."""
    if node.kind is TokenKind.CONST:
        return node.value  # type: ignore
    if node.kind is TokenKind.VAR:
        # If var is not in bindings, we fall back to 'x' as per signature_bindings
        return bindings.get(node.var_name, bindings.get("x", 0j)) # type: ignore
    if node.kind is TokenKind.EML:
        left_val = _evaluate_tree(node.left, bindings)  # type: ignore
        right_val = _evaluate_tree(node.right, bindings)  # type: ignore
        return _safe_eml(left_val, right_val)
    raise ValueError(f"unsupported node kind for evaluation: {node.kind}")


def generate_pairs(cfg: GeneratorConfig) -> list[SignatureProgramPair]:
    """Generate unique (signature, RPN) pairs based on config."""
    registry = build_registry(load_catalog())
    
    # Filter registry if needed
    if cfg.exclude_entries:
        registry = {
            k: v for k, v in registry.items() 
            if k not in cfg.exclude_entries
        }

    # Map from signature key to (rpn, signature)
    deduped: dict[tuple[tuple[float, float], ...], tuple[str, tuple[complex, ...]]] = {}
    pairs: list[SignatureProgramPair] = []

    for rpn in iter_composition_rpns(registry, cfg.max_composition_depth):
        try:
            # 1. Expand tree
            tree = parse_and_expand(rpn, registry)
            
            # 2. Extract variables for stable signature bindings
            # We can use a simple walk to get variable names
            vars_in_tree: set[str] = set()
            def collect_vars(n: EMLNode):
                if n.kind is TokenKind.VAR:
                    vars_in_tree.add(n.var_name)  # type: ignore
                elif n.kind is TokenKind.EML:
                    collect_vars(n.left)  # type: ignore
                    collect_vars(n.right)  # type: ignore
            collect_vars(tree)
            vars_sorted = tuple(sorted(vars_in_tree))
            
            # 3. Evaluate at test points
            sig_list: list[complex] = []
            for p in TEST_POINTS:
                bindings = signature_bindings(p, vars_sorted)
                val = _evaluate_tree(tree, bindings)
                sig_list.append(val)
            
            signature = tuple(sig_list)
            
            if cfg.unique_signatures_only:
                # Round for stable comparison (tolerance 10^-10 as per plan)
                sig_key = tuple(
                    (round(c.real, 10), round(c.imag, 10)) 
                    for c in signature
                )
                
                if sig_key in deduped:
                    existing_rpn, _ = deduped[sig_key]
                    # Keep shorter RPN, tiebreak lexicographically
                    if len(rpn) < len(existing_rpn) or (
                        len(rpn) == len(existing_rpn) and rpn < existing_rpn
                    ):
                        deduped[sig_key] = (rpn, signature)
                else:
                    deduped[sig_key] = (rpn, signature)
            else:
                pairs.append(SignatureProgramPair(signature, rpn))

        except Exception:
            continue

    if cfg.unique_signatures_only:
        return [
            SignatureProgramPair(sig, rpn) 
            for rpn, sig in deduped.values()
        ]
    else:
        return pairs
