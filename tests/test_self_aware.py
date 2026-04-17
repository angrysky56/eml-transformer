import torch

from eml_transformer.models.decoder import ModelConfig, TinyDecoder
from eml_transformer.models.layers import DecoderLayer, FeedForward, SelfAwareFFN


def test_self_aware_ffn_scaling():
    """Verify that SelfAwareFFN output scales with effort."""
    d_model = 64
    ffn = SelfAwareFFN(d_model)
    x = torch.randn(2, 10, d_model)

    effort_0 = torch.zeros(2, 10, 1)
    effort_1 = torch.ones(2, 10, 1)
    effort_2 = torch.ones(2, 10, 1) * 2.0

    out_0 = ffn(x, effort_0)
    out_1 = ffn(x, effort_1)
    out_2 = ffn(x, effort_2)

    # Check that out_1 = fixed + delta and out_2 = fixed + 2*delta
    # Therefore out_2 - out_1 should equal out_1 - out_0
    delta_1 = out_1 - out_0
    delta_2 = out_2 - out_1

    torch.testing.assert_close(delta_1, delta_2, rtol=1e-5, atol=1e-5)

    # Also check that out_0 is exactly the output of the fixed sub-FFN
    torch.testing.assert_close(out_0, ffn.fixed(x))


def test_decoder_layer_self_aware_propagation():
    """Verify that DecoderLayer passes effort to its FFN."""
    d_model = 64
    layer = DecoderLayer(d_model, n_heads=4, max_seq_len=128, self_aware=True)
    x = torch.randn(1, 5, d_model)
    effort = torch.ones(1, 5, 1)

    # If effort=0, it should differ from effort=1
    out_0 = layer(x, effort=torch.zeros_like(effort))
    out_1 = layer(x, effort=effort)

    assert not torch.allclose(out_0, out_1)


def test_tiny_decoder_self_aware_flag():
    """Verify that TinyDecoder respects the self_aware config flag."""
    config_normal = ModelConfig(self_aware=False)
    decoder_normal = TinyDecoder(config_normal)
    assert isinstance(decoder_normal.layers[0].ffn, FeedForward)

    config_aware = ModelConfig(self_aware=True)
    decoder_aware = TinyDecoder(config_aware)
    assert isinstance(decoder_aware.layers[0].ffn, SelfAwareFFN)


def test_tiny_decoder_forward_with_effort():
    """Verify TinyDecoder forward pass with effort tensor."""
    config = ModelConfig(self_aware=True)
    decoder = TinyDecoder(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 10))
    effort = torch.randn(1, 10, 1)

    # Should run without error
    out = decoder(input_ids, effort=effort)
    assert out.shape == (1, 10, config.d_model)
