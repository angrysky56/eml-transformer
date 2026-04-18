"""Tests for the EML compiler package: RPN, machine, composer, verifier.

These tests are independent of the eml-mcp catalog where possible (we use
hand-built trees and minimal registries) so they run fast and don't require
the catalog DB file to exist for most cases.
"""

from __future__ import annotations

import math

import pytest

from eml_transformer.compiler import (
    CalleeSpec,
    CallResolutionError,
    EMLMachine,
    EMLNode,
    RPNParseError,
    TokenKind,
    compile_tree,
    compose,
    expand_calls,
    parse_and_expand,
    parse_rpn_to_tree,
    substitute_vars,
    tokenize_rpn,
    tree_to_rpn,
)


# ---------------------------------------------------------------------------
# Helper: build small trees without RPN so tests stay readable.
# ---------------------------------------------------------------------------


def _const(v: complex | float) -> EMLNode:
    return EMLNode(kind=TokenKind.CONST, value=complex(v))


def _var(name: str) -> EMLNode:
    return EMLNode(kind=TokenKind.VAR, var_name=name)


def _eml(a: EMLNode, b: EMLNode) -> EMLNode:
    return EMLNode(kind=TokenKind.EML, left=a, right=b)


def _call(name: str, *args: EMLNode) -> EMLNode:
    return EMLNode(kind=TokenKind.CALL, call_name=name, call_args=tuple(args))


# ---------------------------------------------------------------------------
# RPN parser tests.
# ---------------------------------------------------------------------------


def test_parse_leaf_only() -> None:
    """A single ``1.0`` is a valid minimal RPN tree."""
    tree = parse_rpn_to_tree("1.0")
    assert tree.kind is TokenKind.CONST
    assert tree.value == complex(1.0)


def test_parse_exp_tree() -> None:
    """``x 1.0 E`` is ``eml(x, 1)`` == ``exp(x)``."""
    tree = parse_rpn_to_tree("x 1.0 E")
    assert tree.kind is TokenKind.EML
    assert tree.left.kind is TokenKind.VAR and tree.left.var_name == "x"
    assert tree.right.kind is TokenKind.CONST


def test_parse_complex_literal() -> None:
    """Parser must handle ``(0.45+1.57j)``-style complex literals."""
    tree = parse_rpn_to_tree("(0.451582705289455+1.5707963267948966j)")
    assert tree.kind is TokenKind.CONST
    assert abs(tree.value.real - 0.451582705289455) < 1e-15
    assert abs(tree.value.imag - 1.5707963267948966) < 1e-15


def test_parse_rejects_underflow() -> None:
    """``E`` with fewer than 2 operands must raise."""
    with pytest.raises(RPNParseError):
        parse_rpn_to_tree("1.0 E")


def test_rpn_round_trip() -> None:
    """Linearize-then-parse should recover the original tree structure."""
    # ln(x) from the catalog.
    original = "1.0 1.0 x E 1.0 E E"
    tree = parse_rpn_to_tree(original)
    # Linearization is semantically equal; stringwise it may differ in
    # numeric formatting, so re-parse and compare structurally.
    linearized = tree_to_rpn(tree)
    reparsed = parse_rpn_to_tree(linearized)
    assert tree_to_rpn(reparsed) == linearized


# ---------------------------------------------------------------------------
# Compiler + machine tests: known-answer checks on hand-built trees.
# ---------------------------------------------------------------------------


def test_machine_computes_exp_of_x() -> None:
    """eml(x, 1) = exp(x) - ln(1) = exp(x)."""
    tree = _eml(_var("x"), _const(1.0))
    machine = EMLMachine(compile_tree(tree))
    for t in (0.0, 0.5, 1.0, 2.5, -1.5):
        result = machine({"x": t})
        expected = math.exp(t)
        assert abs(result - expected) < 1e-12, (t, result, expected)


def test_machine_computes_ln_composition() -> None:
    """Catalog's ln(x) tree must match math.log at multiple points."""
    # 1.0 1.0 x E 1.0 E E == ln(x)
    tree = parse_rpn_to_tree("1.0 1.0 x E 1.0 E E")
    machine = EMLMachine(compile_tree(tree))
    for t in (0.5, 1.0, 2.5, 10.0):
        result = machine({"x": t})
        expected = math.log(t)
        assert abs(result - expected) < 1e-10, (t, result, expected)


def test_machine_has_correct_layer_count() -> None:
    """Layer count is tree_depth + 1 (one leaf-injection layer + depth)."""
    tree = parse_rpn_to_tree("1.0 1.0 x E 1.0 E E")  # depth 3
    machine = EMLMachine(compile_tree(tree))
    assert machine.config.num_layers == 4  # depth 3 + 1


def test_machine_rejects_missing_bindings() -> None:
    """Missing a required variable must raise ValueError."""
    tree = _eml(_var("x"), _const(1.0))
    machine = EMLMachine(compile_tree(tree))
    with pytest.raises(ValueError, match="missing"):
        machine({})


def test_machine_handles_constants_only() -> None:
    """A constant-only tree (e.g. ``e``) needs no bindings."""
    # eml(1, 1) = exp(1) - ln(1) = e - 0 = e
    tree = _eml(_const(1.0), _const(1.0))
    machine = EMLMachine(compile_tree(tree))
    result = machine({})  # no bindings needed
    assert abs(result - math.e) < 1e-12


# ---------------------------------------------------------------------------
# Composer tests: substitution, expansion, composition.
# ---------------------------------------------------------------------------


def _ln_spec() -> CalleeSpec:
    """Minimal catalog-free spec for ln(x)."""
    body = parse_rpn_to_tree("1.0 1.0 x E 1.0 E E")
    return CalleeSpec(name="ln", body=body, variables=("x",))


def _exp_spec() -> CalleeSpec:
    body = parse_rpn_to_tree("x 1.0 E")
    return CalleeSpec(name="exp", body=body, variables=("x",))


def test_substitute_vars_deep_copies() -> None:
    """Substitution must not alias — mutating a source subtree after
    substitution must not affect the substituted tree."""
    source = _eml(_var("x"), _var("x"))  # eml(x, x)
    arg = _const(5.0)
    substituted = substitute_vars(source, {"x": arg})
    # Mutating the arg's value shouldn't propagate into the substituted tree.
    arg.value = complex(99.0)
    # Both positions in ``substituted`` should still hold 5.0.
    assert substituted.left.value == complex(5.0)
    assert substituted.right.value == complex(5.0)


def test_expand_calls_resolves_single_level() -> None:
    """A lone CALL node becomes the callee body with arg substitution."""
    registry = {"ln": _ln_spec()}
    tree = _call("ln", _var("y"))
    expanded = expand_calls(tree, registry)
    assert not expanded.has_calls()
    # The expanded tree should compute ln(y) — verify numerically.
    machine = EMLMachine(compile_tree(expanded))
    for t in (0.5, 1.0, 2.5):
        result = machine({"y": t})
        assert abs(result - math.log(t)) < 1e-10


def test_expand_calls_resolves_nested() -> None:
    """sin(ln(x)) via ln call inside sin: but we test the simpler exp(ln(x))."""
    registry = {"ln": _ln_spec(), "exp": _exp_spec()}
    # CALL tree: exp(ln(x))  — should compute to x
    tree = _call("exp", _call("ln", _var("x")))
    expanded = expand_calls(tree, registry)
    assert not expanded.has_calls()
    machine = EMLMachine(compile_tree(expanded))
    for t in (0.5, 1.0, 2.5, 10.0):
        result = machine({"x": t})
        # exp(ln(t)) == t, to numerical precision
        assert abs(result - t) < 1e-10, (t, result)


def test_compose_builds_sin_of_expression() -> None:
    """compose(exp_spec, some_tree) should produce exp-applied-to-that-tree."""
    exp_spec = _exp_spec()
    # Pass a non-trivial arg: ln of y, which the compose won't expand
    # (it just does var substitution). The returned tree therefore has
    # ln of y plugged into exp's body, which after compilation is exp(ln(y)).
    ln_of_y = parse_rpn_to_tree("1.0 1.0 y E 1.0 E E")
    composed = compose(exp_spec, ln_of_y)
    # composed should have no CALL nodes (exp body has none either).
    assert not composed.has_calls()
    machine = EMLMachine(compile_tree(composed))
    for t in (0.5, 1.0, 2.5):
        result = machine({"y": t})
        assert abs(result - t) < 1e-10


def test_compose_arity_mismatch() -> None:
    """compose must raise if arg count disagrees with spec.arity."""
    ln_spec = _ln_spec()  # arity 1
    with pytest.raises(CallResolutionError, match="arity"):
        compose(ln_spec, _var("x"), _var("y"))  # too many args


def test_expand_calls_unknown_name() -> None:
    """Unknown callee must raise with a helpful message."""
    tree = _call("nonexistent", _var("x"))
    with pytest.raises(CallResolutionError, match="unknown callee"):
        expand_calls(tree, {})


def test_expand_calls_arity_mismatch() -> None:
    """arg count vs variable count mismatch raises."""
    registry = {"ln": _ln_spec()}  # arity 1
    tree = _call("ln", _var("x"), _var("y"))  # 2 args
    with pytest.raises(CallResolutionError, match="arity"):
        expand_calls(tree, registry)


def test_parse_and_expand_end_to_end() -> None:
    """RPN ``y ln exp`` with a full registry should compute exp(ln(y)) = y."""
    registry = {"ln": _ln_spec(), "exp": _exp_spec()}
    tree = parse_and_expand("y ln exp", registry)
    assert not tree.has_calls()
    machine = EMLMachine(compile_tree(tree))
    for t in (0.5, 1.0, 2.5, 10.0):
        result = machine({"y": t})
        assert abs(result - t) < 1e-10


def test_expand_max_depth_guard() -> None:
    """Recursion guard must trip on a self-referential registry."""
    # Build a pathological spec whose body calls itself.
    self_call = EMLNode(
        kind=TokenKind.CALL, call_name="bomb", call_args=(_var("x"),)
    )
    bomb = CalleeSpec(name="bomb", body=self_call, variables=("x",))
    tree = _call("bomb", _var("x"))
    with pytest.raises(CallResolutionError, match="max_depth"):
        expand_calls(tree, {"bomb": bomb}, max_depth=8)
