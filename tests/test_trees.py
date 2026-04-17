"""Tests for the EML tree data structure and linearization.

The single most important thing these tests verify is that `linearize(tree)`
produces depth labels that agree with the `.depth` property of each node in
postorder. Without that invariant, the Effort Evaluator would be learning
noise instead of subtree depth.
"""

from __future__ import annotations

import math
import random

import pytest

from eml_transformer.data.trees import (
    EMLNode,
    NodeType,
    const_one,
    eml,
    linearize,
    random_tree,
    var,
)


def test_leaf_depths_are_zero() -> None:
    assert const_one().depth == 0
    assert var("x").depth == 0


def test_single_eml_depth_is_one() -> None:
    tree = eml(var("x"), const_one())  # this is exp(x) in EML
    assert tree.depth == 1


def test_nested_eml_depth() -> None:
    # eml(eml(x, 1), 1) -- a depth-2 tree
    tree = eml(eml(var("x"), const_one()), const_one())
    assert tree.depth == 2


def test_asymmetric_depth_uses_max() -> None:
    # Left branch has depth 2, right branch has depth 1.
    # Root depth should be 1 + max(2, 1) = 3.
    deep_left = eml(eml(var("x"), const_one()), const_one())
    shallow_right = eml(var("y"), const_one())
    tree = eml(deep_left, shallow_right)
    assert tree.depth == 3
    assert deep_left.depth == 2
    assert shallow_right.depth == 1


def test_linearize_rpn_order_is_postorder() -> None:
    """Tokens must come out in RPN: left subtree, right subtree, then op."""
    tree = eml(var("x"), const_one())  # exp(x)
    lin = linearize(tree)
    assert lin.tokens == ["x", "1", "E"]


def test_linearize_nested_tree_order() -> None:
    """More complex check: eml(eml(x, 1), y) -> x 1 E y E."""
    tree = eml(eml(var("x"), const_one()), var("y"))
    lin = linearize(tree)
    assert lin.tokens == ["x", "1", "E", "y", "E"]


def test_linearize_depths_match_node_property() -> None:
    """The killer invariant: per-token depths == node.depth in postorder."""
    tree = eml(eml(var("x"), const_one()), var("y"))
    lin = linearize(tree)
    # Expected subtree depths at each RPN position:
    #   'x' -> 0, '1' -> 0, inner 'E' (eml(x,1)) -> 1, 'y' -> 0, root 'E' -> 2
    assert lin.depths == [0, 0, 1, 0, 2]
    assert lin.root_depth() == 2


def test_linearize_length_equals_node_count() -> None:
    tree = eml(eml(var("x"), const_one()), eml(var("y"), const_one()))
    lin = linearize(tree)
    assert len(lin) == tree.node_count


def test_random_tree_respects_max_depth() -> None:
    rng = random.Random(0)
    for _ in range(50):
        tree = random_tree(max_depth=4, variables=["x", "y"], rng=rng)
        assert 1 <= tree.depth <= 4


def test_random_tree_determinism() -> None:
    """Same seed -> same tree. This is what EffortDataset relies on."""
    rng_a = random.Random(123)
    rng_b = random.Random(123)
    tree_a = random_tree(max_depth=5, variables=["x"], rng=rng_a)
    tree_b = random_tree(max_depth=5, variables=["x"], rng=rng_b)
    assert tree_a.to_expression() == tree_b.to_expression()


def test_known_formula_exp_evaluates_correctly() -> None:
    """eml(x, 1) should compute exp(x) - ln(1) = exp(x)."""
    tree = eml(var("x"), const_one())
    for x in (0.0, 1.0, 2.0, -1.0):
        assert tree.evaluate({"x": x}) == pytest.approx(math.exp(x), abs=1e-12)


def test_evaluate_raises_on_unbound_variable() -> None:
    tree = eml(var("x"), const_one())
    with pytest.raises(ValueError, match="Unbound variable"):
        tree.evaluate({})


def test_max_depth_below_one_rejected() -> None:
    with pytest.raises(ValueError):
        random_tree(max_depth=0, variables=["x"], rng=random.Random(0))
