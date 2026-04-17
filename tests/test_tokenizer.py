"""Tests for the EML tokenizer."""

from __future__ import annotations

import pytest

from eml_transformer.data.tokenizer import (
    BOS_TOKEN,
    CONST_ONE_TOKEN,
    EML_OPCODE_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
    EMLTokenizer,
)


def test_pad_is_always_id_zero() -> None:
    """Loss functions that use padding rely on this."""
    tok = EMLTokenizer.from_variables(["x", "y"])
    assert tok.pad_id == 0
    assert tok.id_to_token[0] == PAD_TOKEN


def test_reserved_tokens_have_stable_low_ids() -> None:
    tok = EMLTokenizer.from_variables(["x"])
    assert tok.token_to_id[PAD_TOKEN] == 0
    assert tok.token_to_id[BOS_TOKEN] == 1
    assert tok.token_to_id[EOS_TOKEN] == 2
    assert tok.token_to_id[UNK_TOKEN] == 3


def test_vocab_layout_reserved_then_semantic_then_variables() -> None:
    tok = EMLTokenizer.from_variables(["x", "y"])
    expected = (
        PAD_TOKEN,
        BOS_TOKEN,
        EOS_TOKEN,
        UNK_TOKEN,
        CONST_ONE_TOKEN,
        EML_OPCODE_TOKEN,
        "x",
        "y",
    )
    assert tok.id_to_token == expected
    assert tok.vocab_size == len(expected)


def test_encode_roundtrip_preserves_tokens() -> None:
    tok = EMLTokenizer.from_variables(["x", "y"])
    tokens = ["x", "1", "E", "y", "E"]
    ids = tok.encode(tokens, add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id
    decoded = tok.decode(ids)
    assert decoded == tokens


def test_unknown_token_maps_to_unk_id() -> None:
    tok = EMLTokenizer.from_variables(["x"])
    ids = tok.encode(["z"], add_bos=False, add_eos=False)
    assert ids == [tok.unk_id]


def test_variable_colliding_with_reserved_rejected() -> None:
    with pytest.raises(ValueError, match="collides"):
        EMLTokenizer.from_variables(["E"])
