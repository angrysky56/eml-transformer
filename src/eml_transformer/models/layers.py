"""Small rotary-positional + RMSNorm transformer building blocks.

These are intentionally plain implementations. The Effort Evaluator's job is
to test whether a learned head can recover EML subtree depth from an RPN
sequence — a minimal architecture keeps that question clean. No dropout,
no bias terms on linear layers (a modern small-LM convention), no
torch.compile for the first proof-of-life.

Effort-conditioning mechanisms
------------------------------
Three modulation modes are supported via the ``ffn_mode`` choice on
``ModelConfig``:

* ``"vanilla"``   — no effort input; standard GELU FFN.
* ``"delta"``     — Gemini's original construction: ``fixed(x) + e * delta(x)``.
                    Duplicates the FFN (2× params) to give the model a second
                    pathway that scales linearly with effort.
* ``"film"``      — feature-wise linear modulation (Perez et al. 2018). A tiny
                    generator turns the scalar effort into per-channel γ and β
                    that scale and shift the FFN's hidden activations before
                    the non-linearity. Adds ``~2 * (d_model * expansion)``
                    params per layer plus a small MLP — far less than ``delta``
                    but with more expressive per-channel routing.

FiLM is the conventional choice in the literature. ``delta`` is kept as an
ablation so we can actually measure which construction helps (if any).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings for Q/K (Su et al. 2021).

    We precompute cos/sin tables up to ``max_seq_len`` and register them as
    non-persistent buffers so they move with ``.to(device)`` but don't bloat
    the checkpoint. The ``base`` constant (10000) matches the GPT-NeoX /
    LLaMA convention.
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for rotary; got {head_dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)  # (T, head_dim / 2)
        # Duplicate so we can pair x1,x2 along the last axis at apply time.
        emb = torch.cat([freqs, freqs], dim=-1)  # (T, head_dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        """Rotate the two halves of the last dim: [x1, x2] -> [-x2, x1]."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        """Apply rotary to Q and K.

        Shapes: q, k are (batch, heads, seq_len, head_dim).
        Returns rotated (q, k) with identical shapes.
        """
        seq_len = q.size(-2)
        if seq_len > self.cos_cached.size(0):
            raise ValueError(
                f"sequence length {seq_len} exceeds cached rotary table "
                f"({self.cos_cached.size(0)}); increase max_seq_len"
            )
        cos = self.cos_cached[:seq_len].to(dtype=q.dtype)[None, None, :, :]
        sin = self.sin_cached[:seq_len].to(dtype=q.dtype)[None, None, :, :]
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention using torch's fused SDPA kernel.

    The padding mask from the dataset is fused with the causal mask into a
    single additive bias. We don't use ``is_causal=True`` because combining it
    with a per-batch key-padding mask isn't supported in all PyTorch 2.x
    builds — constructing the full bias is reliable and negligible cost at
    this model size.
    """

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, d_model) input tensor.
            key_padding_mask: (batch, seq_len) bool; True means *real* (attend),
                False means *pad* (mask out). None means no padding.

        Returns:
            (batch, seq_len, d_model) output tensor.
        """
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, T, H, Hd)
        q = q.transpose(1, 2)  # (B, H, T, Hd)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = self.rope(q, k)

        # Build combined (causal + padding) additive bias in float dtype.
        causal = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        if key_padding_mask is not None:
            pad_cols = ~key_padding_mask  # (B, T) True = pad column to mask
            # Broadcast to (B, 1, T, T) so every query row masks the same pad keys.
            mask_bool = causal[None, None, :, :] | pad_cols[:, None, None, :]
        else:
            mask_bool = causal[None, None, :, :].expand(B, 1, T, T)
        bias = torch.zeros(mask_bool.shape, dtype=q.dtype, device=x.device)
        bias.masked_fill_(mask_bool, float("-inf"))

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)


class FeedForward(nn.Module):
    """Position-wise GELU FFN. Standard 4x expansion ratio.

    Signature supports an optional ``effort`` argument for interface symmetry
    with the modulated variants, but ignores it.
    """

    def __init__(self, d_model: int, expansion: int = 4) -> None:
        super().__init__()
        hidden = d_model * expansion
        self.fc_in = nn.Linear(d_model, hidden, bias=False)
        self.fc_out = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: Tensor, effort: Tensor | None = None) -> Tensor:  # noqa: ARG002
        return self.fc_out(F.gelu(self.fc_in(x)))


class DeltaFFN(nn.Module):
    """Duplicated-FFN modulation: ``fixed(x) + effort * delta(x)``.

    This is the original construction from the first Phase-2 attempt. It's
    kept as an ablation target: fair comparisons should report it alongside
    FiLM to measure whether modulation helps more than the extra parameters.
    Note this doubles the FFN parameter count vs vanilla.
    """

    def __init__(self, d_model: int, expansion: int = 4) -> None:
        super().__init__()
        self.fixed = FeedForward(d_model, expansion)
        self.delta = FeedForward(d_model, expansion)

    def forward(self, x: Tensor, effort: Tensor | None = None) -> Tensor:
        f_out = self.fixed(x)
        if effort is None:
            return f_out
        d_out = self.delta(x)
        return f_out + effort * d_out


class FiLMFFN(nn.Module):
    """Feature-wise Linear Modulation FFN (Perez et al. 2018).

    A single linear layer turns the scalar effort into per-channel ``(γ, β)``
    that modulate the hidden activations between ``fc_in`` and the GELU
    non-linearity::

        h = fc_in(x)
        γ, β = linear(effort).chunk(2, dim=-1)
        h = (1 + γ) * h + β
        h = gelu(h)
        out = fc_out(h)

    The ``1 +`` initialization on γ means that at init (zero effort, zero γ-bias)
    the layer behaves as a plain FFN — so a frozen-evaluator run with untrained
    effort signal degrades gracefully to vanilla. Applying modulation *before*
    the non-linearity is what lets a linear parameterization compose into a
    nonlinear change downstream, which is the key insight from the original
    FiLM retrospective.

    Parameter cost per layer: ``d_model * 2 * hidden`` for the generator,
    compared to ``DeltaFFN``'s duplicate ``~2 * 4 * d_model^2``. For ``d_model=128``
    and ``expansion=4``: FiLM adds ~1K generator params; Delta adds ~130K.
    """

    def __init__(self, d_model: int, expansion: int = 4, effort_dim: int = 1) -> None:
        super().__init__()
        hidden = d_model * expansion
        self.fc_in = nn.Linear(d_model, hidden, bias=False)
        self.fc_out = nn.Linear(hidden, d_model, bias=False)
        # Generator: effort scalar -> (γ, β) concatenated along last dim.
        self.film_gen = nn.Linear(effort_dim, 2 * hidden, bias=True)
        # Tag the generator so the decoder's global ``_init_weights`` pass
        # doesn't overwrite the zero init below. See the FiLM retrospective's
        # note on regularization-at-init: large γ/β at init destabilize
        # training. Zero init means the layer begins as a plain FFN and
        # learns modulation from scratch.
        self.film_gen._is_film_gen = True  # type: ignore[attr-defined]
        nn.init.zeros_(self.film_gen.weight)
        nn.init.zeros_(self.film_gen.bias)
        self.hidden = hidden

    def forward(self, x: Tensor, effort: Tensor | None = None) -> Tensor:
        """Forward pass with FiLM modulation.

        Args:
            x: (batch, seq_len, d_model) input.
            effort: (batch, seq_len, 1) per-token effort. If ``None``, the layer
                computes a plain FFN (γ=0, β=0), matching ``FeedForward`` exactly.

        Returns:
            (batch, seq_len, d_model) output.
        """
        h = self.fc_in(x)
        if effort is not None:
            params = self.film_gen(effort)  # (B, T, 2 * hidden)
            gamma, beta = params.chunk(2, dim=-1)  # each (B, T, hidden)
            h = (1.0 + gamma) * h + beta
        h = F.gelu(h)
        return self.fc_out(h)


def make_ffn(d_model: int, mode: str, expansion: int = 4) -> nn.Module:
    """Factory for the FFN variants. Keeps DecoderLayer mode-agnostic."""
    if mode == "vanilla":
        return FeedForward(d_model, expansion)
    if mode == "delta":
        return DeltaFFN(d_model, expansion)
    if mode == "film":
        return FiLMFFN(d_model, expansion)
    raise ValueError(f"unknown ffn_mode {mode!r}; expected vanilla|delta|film")


# Backward-compat alias: the pre-refactor Phase-2 name for DeltaFFN. Tests
# and external callers that import ``SelfAwareFFN`` keep working.
SelfAwareFFN = DeltaFFN


class DecoderLayer(nn.Module):
    """Pre-norm transformer decoder layer with optional effort modulation.

    The FFN is chosen by ``ffn_mode``. All three variants accept the same
    ``(x, effort)`` signature, so the layer's forward is mode-agnostic.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        eps: float = 1e-5,
        ffn_mode: str = "vanilla",
        ffn_expansion: int = 4,
    ) -> None:
        super().__init__()
        self.ffn_mode = ffn_mode
        self.norm_attn = nn.RMSNorm(d_model, eps=eps)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len)
        self.norm_ffn = nn.RMSNorm(d_model, eps=eps)
        self.ffn = make_ffn(d_model, ffn_mode, expansion=ffn_expansion)

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Tensor | None = None,
        effort: Tensor | None = None,
    ) -> Tensor:
        x = x + self.attn(self.norm_attn(x), key_padding_mask=key_padding_mask)
        x = x + self.ffn(self.norm_ffn(x), effort)
        return x


class LMHead(nn.Module):
    """Simple linear head mapping hidden states to vocabulary logits."""

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(x)
