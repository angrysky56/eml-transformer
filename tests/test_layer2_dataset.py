"""Tests for Layer 2 dataset generation."""

import pytest
from eml_transformer.compiler import build_registry, load_catalog
from eml_transformer.layer2.dataset import (
    GeneratorConfig,
    generate_pairs,
    iter_composition_rpns,
)


def test_iter_composition_rpns_depth_1():
    # Setup a tiny registry
    registry = build_registry(load_catalog(include_names={"sin", "add"}))
    # sin arity 1, add arity 2
    
    rpns = list(iter_composition_rpns(registry, max_depth=1, include_variables=("x",)))
    
    # Depth 0: x
    # Depth 1: x sin, x x add
    assert "x" in rpns
    assert "x sin" in rpns
    assert "x x add" in rpns
    assert len(rpns) == 3


def test_iter_composition_rpns_depth_2():
    registry = build_registry(load_catalog(include_names={"sin"}))
    rpns = list(iter_composition_rpns(registry, max_depth=2, include_variables=("x",)))
    
    # Depth 0: x
    # Depth 1: x sin
    # Depth 2: x sin sin
    assert "x" in rpns
    assert "x sin" in rpns
    assert "x sin sin" in rpns
    assert len(rpns) == 3


def test_generate_pairs_deduplication():
    cfg = GeneratorConfig(max_composition_depth=1)
    # Using a small subset to be fast
    pairs = generate_pairs(cfg)
    
    # Check if RPNs are unique
    rpns = [p.rpn for p in pairs]
    assert len(rpns) == len(set(rpns))
    
    # Check if signatures are unique
    signatures = [tuple(round(c.real, 10) + round(c.imag, 10)*1j for c in p.signature) for p in pairs]
    assert len(signatures) == len(set(signatures))


def test_generate_pairs_depth_2_small_registry():
    # Verify depth 2 works with a limited registry to stay fast
    cfg = GeneratorConfig(
        max_composition_depth=2,
        exclude_entries=frozenset(c.name for c in load_catalog() if c.name not in {"sin", "cos", "add"})
    )
    pairs = generate_pairs(cfg)
    assert len(pairs) > 5
    assert any(p.rpn == "x" for p in pairs)
    assert any(p.rpn == "x sin sin" for p in pairs)
    assert any(p.rpn == "x y add" for p in pairs)
