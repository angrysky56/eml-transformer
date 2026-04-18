from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from eml_transformer.models.layers import CausalSelfAttention, FeedForward


@dataclass(frozen=True)
class Layer2Config:
    """Configuration for the Layer 2 signature-to-program model."""

    vocab_size: int
    signature_dim: int = 12
    d_model: int = 128
    n_heads: int = 4
    n_decoder_layers: int = 4
    max_target_length: int = 16
    dropout: float = 0.1
    eps: float = 1e-5


class SignatureEncoder(nn.Module):
    """Encodes a 12-dimensional numerical signature into a d_model context token."""

    def __init__(self, cfg: Layer2Config):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cfg.signature_dim, cfg.d_model, bias=False),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model, bias=False),
            nn.RMSNorm(cfg.d_model, eps=cfg.eps),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: (batch, 12) signature tensor.

        Returns:
            (batch, 1, d_model) context token.
        """
        return self.mlp(x).unsqueeze(1)


class CrossAttention(nn.Module):
    """Multi-head cross-attention layer."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, d_model) decoder stream.
            context: (batch, context_len, d_model) encoder output.

        Returns:
            (batch, seq_len, d_model) context-aware representation.
        """
        batch_size, seq_len, _ = x.shape
        _, context_len, _ = context.shape

        q = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        kv = (
            self.kv_proj(context)
            .view(batch_size, context_len, 2, self.n_heads, self.head_dim)
            .transpose(1, 3)
        )
        k, v = kv.unbind(dim=2)  # each (B, H, C, Hd)

        # Cross-attention has no causal mask
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(out)


class Layer2DecoderLayer(nn.Module):
    """Decoder layer with self-attention, cross-attention, and FFN."""

    def __init__(self, cfg: Layer2Config):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model, eps=cfg.eps)
        self.self_attn = CausalSelfAttention(
            cfg.d_model, cfg.n_heads, cfg.max_target_length
        )

        self.norm2 = nn.RMSNorm(cfg.d_model, eps=cfg.eps)
        self.cross_attn = CrossAttention(cfg.d_model, cfg.n_heads)

        self.norm3 = nn.RMSNorm(cfg.d_model, eps=cfg.eps)
        self.ffn = FeedForward(cfg.d_model)

    def forward(
        self, x: Tensor, context: Tensor, key_padding_mask: Tensor | None = None
    ) -> Tensor:
        """Forward pass with residual connections."""
        x = x + self.self_attn(self.norm1(x), key_padding_mask=key_padding_mask)
        x = x + self.cross_attn(self.norm2(x), context)
        x = x + self.ffn(self.norm3(x))
        return x


class SignatureProgramModel(nn.Module):
    """Encoder-Decoder model for mapping signatures to EML programs."""

    def __init__(self, cfg: Layer2Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.encoder = SignatureEncoder(cfg)
        self.decoder_layers = nn.ModuleList(
            [Layer2DecoderLayer(cfg) for _ in range(cfg.n_decoder_layers)]
        )
        self.ln_f = nn.RMSNorm(cfg.d_model, eps=cfg.eps)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying (Embedding and LM Head)
        self.tok_emb.weight = self.head.weight

    def forward(
        self, signature: Tensor, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        """Forward pass.

        Args:
            signature: (batch, 12) target function signature.
            input_ids: (batch, seq_len) RPN token IDs.
            attention_mask: (batch, seq_len) bool; True means *real*, False means *pad*.

        Returns:
            (batch, seq_len, vocab_size) logits.
        """
        x = self.tok_emb(input_ids)
        context = self.encoder(signature)

        for layer in self.decoder_layers:
            x = layer(x, context, key_padding_mask=attention_mask)

        return self.head(self.ln_f(x))

    @torch.no_grad()
    def generate(
        self,
        signature: Tensor,
        *,
        max_length: int = 16,
        bos_id: int = 1,
        eos_id: int = 2,
        temperature: float = 0.0,
    ) -> list[list[int]]:
        """Autoregressive generation of RPN token IDs.

        Returns:
            A batch of token ID lists, each stopping at EOS or max_length.
        """
        batch_size = signature.shape[0]
        device = signature.device

        # Start with BOS
        input_ids = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        results: list[list[int]] = [[] for _ in range(batch_size)]

        for _ in range(max_length):
            # Pass full sequence so far to handle causal mask/positional info correctly
            logits = self.forward(signature, input_ids)[:, -1, :]

            if temperature == 0.0:
                next_tokens = torch.argmax(logits, dim=-1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

            # Record tokens and check for EOS
            for i in range(batch_size):
                if not finished[i]:
                    token_id = int(next_tokens[i].item())
                    if token_id == eos_id:
                        finished[i] = True
                    else:
                        results[i].append(token_id)

            if finished.all():
                break

            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(1)], dim=1)

        return results
