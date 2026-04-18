"""Composition layer: CALL-node resolution and tree-algebra helpers.

This module is what makes catalog entries *callable*. Without composition,
every program had to be written as one giant monolithic RPN string.
With composition, the learned generator (next phase) can emit short
token streams like ``x ln sin`` — three tokens — and the executor resolves
that to a concrete EML tree before compilation.

The central operation is :func:`expand_calls`: given a tree that may
contain unresolved CALL nodes and a registry mapping catalog names to
their EML trees, return a new tree where every CALL has been replaced
by the callee's body with appropriate variable substitution.

Semantics:

* Each catalog entry has a *sorted* variable list — eml-mcp's
  ``sorted(get_variables())`` convention.
* A CALL node supplies argument subtrees *positionally*, in the same
  order as that sorted variable list.
* Resolution substitutes arg[j] for every occurrence of the j-th
  variable in the callee body.

α-renaming comes for free because we work on the tree representation:
variable *names* in the callee's body are just string labels; substituting
subtrees for them is a deep-copy operation that never needs fresh names.
The only invariant we care about is that the same variable name inside a
callee body always refers to the same argument subtree — which is
guaranteed by the positional mapping above.

Name collisions between caller and callee variables are not a problem
because callee variables are substituted *before* the callee body is
merged into the caller's tree. After expansion there are no more
references to the callee's original variable names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from eml_transformer.compiler.catalog import CatalogEntry
from eml_transformer.compiler.rpn import EMLNode, TokenKind, parse_rpn_to_tree


class CallResolutionError(ValueError):
    """Raised when a CALL node can't be expanded.

    Common causes: the called name isn't in the registry, the arg count
    doesn't match the callee's variable count, or a callee body refers
    to a variable that isn't listed in its declared variable set.
    """


@dataclass(frozen=True)
class CalleeSpec:
    """A registered callable catalog entry.

    Attributes:
        name: The public name used in CALL tokens (e.g. ``"sin"``).
        body: The callee's expression tree. May itself contain CALL nodes,
            which will be resolved recursively during expansion.
        variables: The callee's variable list in the *exact order* that
            CALL-node arguments are matched against. By convention this
            follows eml-mcp's ``sorted(get_variables())``.
    """

    name: str
    body: EMLNode
    variables: tuple[str, ...]

    @property
    def arity(self) -> int:
        return len(self.variables)


def build_registry(
    entries: list[CatalogEntry],
    *,
    inner_call_arities: dict[str, int] | None = None,
) -> dict[str, CalleeSpec]:
    """Parse a list of catalog entries into a CALL registry.

    Args:
        entries: Catalog entries as returned by
            :func:`eml_transformer.compiler.catalog.load_catalog`.
        inner_call_arities: Optional arity table *for CALL tokens inside
            the parsed catalog RPN*. In practice the raw catalog never
            contains CALL tokens (they're all fully-expanded EML trees),
            so the default ``None`` is correct. The parameter exists so
            this function can be used with a "layered" library where
            later catalog entries call earlier ones by name.

    Returns:
        A dict mapping each entry's name to a :class:`CalleeSpec`, with
        the body parsed and variables in sorted order.
    """
    registry: dict[str, CalleeSpec] = {}
    for entry in entries:
        body = parse_rpn_to_tree(entry.rpn, call_arities=inner_call_arities)
        # Match eml-mcp's signature-binding convention: variables are
        # sorted for the stable positional mapping used by CALL args.
        sorted_vars = tuple(sorted(entry.variables))
        registry[entry.name] = CalleeSpec(
            name=entry.name, body=body, variables=sorted_vars
        )
    return registry


def _deep_copy(node: EMLNode) -> EMLNode:
    """Deep-copy an EMLNode subtree. Avoids aliasing when we splice trees."""
    if node.kind is TokenKind.EML:
        return EMLNode(
            kind=TokenKind.EML,
            left=_deep_copy(node.left),  # type: ignore[arg-type]
            right=_deep_copy(node.right),  # type: ignore[arg-type]
        )
    if node.kind is TokenKind.CALL:
        return EMLNode(
            kind=TokenKind.CALL,
            call_name=node.call_name,
            call_args=tuple(_deep_copy(a) for a in node.call_args or ()),
        )
    if node.kind is TokenKind.CONST:
        return EMLNode(kind=TokenKind.CONST, value=node.value)
    if node.kind is TokenKind.VAR:
        return EMLNode(kind=TokenKind.VAR, var_name=node.var_name)
    raise CallResolutionError(f"unknown node kind: {node.kind}")


def substitute_vars(
    tree: EMLNode,
    bindings: Mapping[str, EMLNode],
) -> EMLNode:
    """Return a new tree with each VAR node substituted by its binding.

    Variables not in ``bindings`` pass through unchanged — useful for
    partial substitution during staged expansion.

    This is α-renaming done at the tree level: a VAR named ``"x"`` in the
    callee body is replaced by a fresh deep-copy of ``bindings["x"]``.
    Different occurrences of the same VAR name all map to copies of the
    same bound subtree (not to the same object — we deep-copy per
    occurrence so later mutations in one branch can't affect another).
    """
    if tree.kind is TokenKind.VAR:
        name = tree.var_name
        if name is not None and name in bindings:
            return _deep_copy(bindings[name])
        return _deep_copy(tree)
    if tree.kind is TokenKind.EML:
        return EMLNode(
            kind=TokenKind.EML,
            left=substitute_vars(tree.left, bindings),  # type: ignore[arg-type]
            right=substitute_vars(tree.right, bindings),  # type: ignore[arg-type]
        )
    if tree.kind is TokenKind.CALL:
        return EMLNode(
            kind=TokenKind.CALL,
            call_name=tree.call_name,
            call_args=tuple(
                substitute_vars(a, bindings) for a in tree.call_args or ()
            ),
        )
    # CONST: no vars possible
    return _deep_copy(tree)


def expand_calls(
    tree: EMLNode,
    registry: Mapping[str, CalleeSpec],
    *,
    max_depth: int = 64,
) -> EMLNode:
    """Recursively resolve every CALL node in ``tree`` against ``registry``.

    The returned tree is a pure CONST/VAR/EML tree ready for the compiler.
    Resolution proceeds depth-first:

    1. For each CALL node, first expand its argument subtrees (which may
       themselves contain CALLs).
    2. Look up the callee body in ``registry``. Arg count must match the
       callee's variable count.
    3. Substitute each callee variable with the corresponding fully-expanded
       arg subtree.
    4. Recursively expand calls in the resulting tree in case the callee's
       body itself referenced other named entries.

    Args:
        tree: The tree to expand. May contain any mix of node kinds.
        registry: Mapping of name → :class:`CalleeSpec`.
        max_depth: Guard against accidentally infinite expansion (e.g., a
            mis-configured registry with a mutual recursion). Decremented
            on each recursive expansion; raises if it hits zero.

    Returns:
        A new tree with no CALL nodes.

    Raises:
        CallResolutionError: On any inconsistency (unknown name, arity
            mismatch, missing body variable, or recursion depth exceeded).
    """
    if max_depth <= 0:
        raise CallResolutionError(
            "expand_calls hit max_depth; registry may contain a cycle"
        )

    if tree.kind is TokenKind.CONST:
        return _deep_copy(tree)
    if tree.kind is TokenKind.VAR:
        return _deep_copy(tree)
    if tree.kind is TokenKind.EML:
        return EMLNode(
            kind=TokenKind.EML,
            left=expand_calls(tree.left, registry, max_depth=max_depth),  # type: ignore[arg-type]
            right=expand_calls(tree.right, registry, max_depth=max_depth),  # type: ignore[arg-type]
        )
    # CALL
    if tree.call_name not in registry:
        raise CallResolutionError(
            f"unknown callee {tree.call_name!r}; registry has: "
            f"{sorted(registry.keys())}"
        )
    spec = registry[tree.call_name]
    args = tree.call_args or ()
    if len(args) != spec.arity:
        raise CallResolutionError(
            f"call to {spec.name!r} supplied {len(args)} args but arity is "
            f"{spec.arity} (variables={spec.variables})"
        )
    # Step 1: expand calls in args first, so substitutions use resolved trees.
    expanded_args = tuple(
        expand_calls(a, registry, max_depth=max_depth - 1) for a in args
    )
    # Step 2: build the substitution and apply to a copy of the body.
    bindings = {var: arg for var, arg in zip(spec.variables, expanded_args)}
    substituted = substitute_vars(spec.body, bindings)
    # Step 3: body may have had its own CALLs; expand them in the new tree.
    return expand_calls(substituted, registry, max_depth=max_depth - 1)


def compose(
    outer: CalleeSpec,
    *inner_trees: EMLNode,
) -> EMLNode:
    """Convenience: build ``outer(inner_1, ..., inner_n)`` as a fresh tree.

    Useful for programmatically constructing compositions without writing
    RPN: ``compose(sin_spec, ln_of_x_tree)`` returns a tree for ``sin(ln(x))``.

    The returned tree is already expanded — no CALL nodes remain if the
    ``outer`` body doesn't reference other callables. If ``outer.body``
    itself contains CALLs, call :func:`expand_calls` on the result with
    a registry that resolves those.

    Args:
        outer: The callable spec to apply.
        *inner_trees: Argument subtrees, one per outer variable, in the
            order given by ``outer.variables``.

    Returns:
        The composed tree.

    Raises:
        CallResolutionError: On arity mismatch.
    """
    if len(inner_trees) != outer.arity:
        raise CallResolutionError(
            f"compose({outer.name!r}) expects arity {outer.arity}, "
            f"got {len(inner_trees)} arg tree(s)"
        )
    bindings = {var: tree for var, tree in zip(outer.variables, inner_trees)}
    return substitute_vars(outer.body, bindings)


def parse_and_expand(
    rpn: str,
    registry: Mapping[str, CalleeSpec],
) -> EMLNode:
    """End-to-end: parse an RPN string that may contain CALL tokens, expand.

    The natural pipeline for a learned generator's output: emit a short
    token string with catalog names in it, pass that string through this
    function, and hand the resulting fully-expanded tree to
    :func:`eml_transformer.compiler.machine.compile_tree`.

    Example RPN (after training a generator)::

        x ln sin      →  sin(ln(x)) as a fully-expanded tree

    Args:
        rpn: RPN source possibly containing callee names as tokens.
        registry: Catalog registry defining valid callee names and bodies.

    Returns:
        A fully-expanded EMLNode ready for compilation.

    Raises:
        RPNParseError or CallResolutionError as appropriate.
    """
    arities = {name: spec.arity for name, spec in registry.items()}
    tree = parse_rpn_to_tree(rpn, call_arities=arities)
    return expand_calls(tree, registry)
