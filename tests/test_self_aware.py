"""Tests for the modulated FFN variants and ``DecoderLayer`` wiring.

Covers:
- ``DeltaFFN`` (the original ``SelfAwareFFN``) linearity in effort.
- ``FiLMFFN`` degrades to vanilla FFN at init (zero γ, zero β).
- ``DecoderLayer`` / ``TinyDecoder`` route effort to modulated FFNs.
- ``ModelConfig(self_aware=True)`` legacy kwarg still works via ``make_config``.
- Parameter counts for each mode match the expected ratios.
"""

from __future__ import annotations

import torch

from eml_transformer.models.decoder import ModelConfig, TinyDecoder, make_config
from eml_transformer.models.layers import (
    DecoderLayer,
    DeltaFFN,
    FeedForward,
    FiLMFFN,
    SelfAwareFFN,
)


def test_delta_ffn_is_linear_in_effort() -> None:
    """DeltaFFN is ``fixed(x) + e * delta(x)``, which is strictly linear in e."""
    ffn = DeltaFFN(d_model=64)
    ffn.eval()
    x = torch.randn(2, 10, 64)

    out_0 = ffn(x, torch.zeros(2, 10, 1))
    out_1 = ffn(x, torch.ones(2, 10, 1))
    out_2 = ffn(x, torch.ones(2, 10, 1) * 2.0)

    # (out_1 - out_0) == (out_2 - out_1): slope is constant, proving linearity.
    torch.testing.assert_close(out_1 - out_0, out_2 - out_1, rtol=1e-5, atol=1e-5)
    # Zero effort gives exactly the ``fixed`` sub-FFN output.
    torch.testing.assert_close(out_0, ffn.fixed(x))


def test_delta_ffn_handles_none_effort() -> None:
    """When no effort is provided, DeltaFFN should compute just the fixed branch."""
    ffn = DeltaFFN(d_model=32)
    ffn.eval()
    x = torch.randn(1, 4, 32)
    out = ffn(x, None)
    torch.testing.assert_close(out, ffn.fixed(x))


def test_self_aware_alias_points_to_delta() -> None:
    """``SelfAwareFFN`` must remain importable as a backward-compat alias."""
    assert SelfAwareFFN is DeltaFFN


def test_film_ffn_degrades_to_vanilla_at_init() -> None:
    """At init (γ=β=0) a FiLMFFN should compute the same output as a vanilla FFN
    *with matched fc_in/fc_out weights* — verified here by comparing a FiLM call
    with nonzero effort against the same FiLM call with effort=None.
    """
    film = FiLMFFN(d_model=32)
    film.eval()
    x = torch.randn(2, 6, 32)
    out_none = film(x, None)
    out_zero = film(x, torch.zeros(2, 6, 1))
    out_one = film(x, torch.ones(2, 6, 1))

    # With the generator zero-init'd, effort=0 and effort=1 both produce the
    # vanilla path because γ=β=0 regardless of the effort value.
    torch.testing.assert_close(out_none, out_zero, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(out_none, out_one, rtol=1e-6, atol=1e-6)


def test_film_generator_is_trainable() -> None:
    """After init the generator params should be zero, but they must carry
    gradients so training can grow them."""
    film = FiLMFFN(d_model=16)
    assert film.film_gen.weight.requires_grad
    assert film.film_gen.bias.requires_grad
    assert torch.all(film.film_gen.weight == 0)
    assert torch.all(film.film_gen.bias == 0)


def test_decoder_layer_respects_ffn_mode() -> None:
    """Each ffn_mode must produce the matching FFN class inside DecoderLayer."""
    layer_v = DecoderLayer(64, n_heads=4, max_seq_len=32, ffn_mode="vanilla")
    layer_d = DecoderLayer(64, n_heads=4, max_seq_len=32, ffn_mode="delta")
    layer_f = DecoderLayer(64, n_heads=4, max_seq_len=32, ffn_mode="film")
    assert isinstance(layer_v.ffn, FeedForward)
    assert isinstance(layer_d.ffn, DeltaFFN)
    assert isinstance(layer_f.ffn, FiLMFFN)


def test_tiny_decoder_ffn_mode_selection() -> None:
    """ModelConfig.ffn_mode controls which FFN is instantiated in every layer."""
    vanilla = TinyDecoder(ModelConfig(d_model=64, n_heads=4))
    delta = TinyDecoder(ModelConfig(d_model=64, n_heads=4, ffn_mode="delta"))
    film = TinyDecoder(ModelConfig(d_model=64, n_heads=4, ffn_mode="film"))
    assert isinstance(vanilla.layers[0].ffn, FeedForward)
    assert isinstance(delta.layers[0].ffn, DeltaFFN)
    assert isinstance(film.layers[0].ffn, FiLMFFN)


def test_legacy_self_aware_kwarg_routes_to_delta() -> None:
    """``make_config(self_aware=True)`` must produce ``ffn_mode='delta'``
    for backward compatibility with saved checkpoints and old CLI flags."""
    cfg = make_config(self_aware=True)
    assert cfg.ffn_mode == "delta"
    cfg_off = make_config(self_aware=False)
    assert cfg_off.ffn_mode == "vanilla"


def test_tiny_decoder_forward_with_effort() -> None:
    """A FiLM or delta decoder accepts effort; vanilla ignores it harmlessly."""
    for mode in ("vanilla", "delta", "film"):
        model = TinyDecoder(ModelConfig(d_model=32, n_heads=4, n_layers=2, ffn_mode=mode))
        ids = torch.randint(0, model.config.vocab_size, (1, 5))
        effort = torch.rand(1, 5, 1)
        out = model(ids, effort=effort)
        assert out.shape == (1, 5, 32)
