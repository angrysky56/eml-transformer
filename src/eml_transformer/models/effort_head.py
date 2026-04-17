"""Effort regression head for the Effort Evaluator.

Rather than a direct scalar regressor, we use a classifier over discrete
depth bins and compute a continuous effort score via the softmax-weighted
mean of bin values. This gives us:

1. Cross-entropy training loss (natural for integer-valued depth labels
   with heavy class imbalance toward depth 0).
2. A calibrated probability distribution over depths we can inspect
   during training to catch pathological collapse to a single bin.
3. A continuous scalar `effort in [0, 1]` usable downstream by
   Self-Aware, Spock, and Maybe Math (future phases).

The number of bins should cover the max depth that can appear at eval time,
not just training time, so that out-of-distribution (deeper) trees can in
principle be predicted correctly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class EffortHead(nn.Module):
    """Linear classifier mapping hidden states to depth-bin logits.

    `num_bins` should equal `max_expected_depth + 1` so all valid integer
    depth labels (0, 1, ..., max_expected_depth) are representable.
    """

    def __init__(self, d_model: int, num_bins: int) -> None:
        super().__init__()
        if num_bins < 2:
            raise ValueError(f"num_bins must be >= 2; got {num_bins}")
        self.num_bins = num_bins
        self.classifier = nn.Linear(d_model, num_bins, bias=True)
        # Cache the bin values as a buffer for the weighted-mean readout.
        # dtype=float32 is intentional; we cast to match q.dtype at call time.
        bin_values = torch.arange(num_bins, dtype=torch.float32)
        self.register_buffer("bin_values", bin_values, persistent=False)

    def forward(self, hidden: Tensor) -> Tensor:
        """Return raw logits over depth bins.

        Args:
            hidden: (batch, seq_len, d_model) hidden states from the decoder.

        Returns:
            (batch, seq_len, num_bins) unnormalized logits. Feed these to
            `F.cross_entropy` with `ignore_index=-100`.
        """
        return self.classifier(hidden)

    def predict_depth(self, hidden: Tensor) -> Tensor:
        """Hard predictions via argmax over bins.

        Returns:
            (batch, seq_len) long tensor of predicted integer depths.
        """
        return self.forward(hidden).argmax(dim=-1)

    def effort_scalar(self, hidden: Tensor, normalize: bool = True) -> Tensor:
        """Continuous effort score via softmax-weighted mean over bins.

        This is the scalar the downstream Self-Aware layer, Spock attention,
        and Maybe Math gate will consume in later phases. It's exactly the
        expected value of depth under the predicted distribution, optionally
        normalized to [0, 1] by dividing by `num_bins - 1`.

        Args:
            hidden: (batch, seq_len, d_model) hidden states.
            normalize: if True, divide by (num_bins - 1) so result is in [0, 1].

        Returns:
            (batch, seq_len) float tensor of effort scores.
        """
        logits = self.forward(hidden)
        probs = F.softmax(logits, dim=-1)  # (B, T, num_bins)
        bin_values = self.bin_values.to(dtype=probs.dtype)  # (num_bins,)
        expected = (probs * bin_values).sum(dim=-1)  # (B, T)
        if normalize:
            expected = expected / max(self.num_bins - 1, 1)
        return expected
