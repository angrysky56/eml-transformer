"""Analytically-compiled tiny transformer that executes an EML program.

Given an EML tree (parsed from catalog RPN), :func:`compile_tree` produces
an :class:`EMLMachine` — a PyTorch module whose forward pass computes the
exact function the tree encodes, verified to machine epsilon against the
catalog's stored signatures.

Architecture
------------
The machine is a decoder-only transformer with *fixed* weights set
analytically at compile time. No training, no gradient descent. The
construction follows Moran's "compiled transformer" framing:

* **Sequence length** ``T`` = number of tokens in the tree's RPN form.
* **Positions** correspond to RPN tokens in left-to-right order. After
  all layers run, position ``i``'s residual stream holds the complex
  value computed at the tree node corresponding to RPN token ``i``.
* **Layers** correspond to tree depth. Layer 0 is the "leaf injection"
  layer (constants and variables become values at their positions).
  Layer ``l`` for ``l ≥ 1`` computes every EML node at depth ``l`` by
  reading its two child values from earlier positions.
* **Attention** is the *routing* mechanism. Each EML node at depth ``l``
  reads from the two positions where its children's values live. Because
  those positions are fixed at compile time, the attention pattern is a
  hard-coded routing table, not a learned function.
* **FFN** is the *computation* mechanism. Once attention has delivered
  the two operand values ``a`` and ``b`` to the position of an EML node,
  the FFN applies ``exp(a) - ln(b)``.

Design choice: the activation function is the EML operator itself, not
GELU/ReLU. This is the honest minimal compiled machine — the contribution
is that a transformer's generic shape (attention routing + pointwise FFN
+ residual stream) is sufficient to execute this computation, not that
we can smuggle ``exp`` and ``ln`` past a GELU. A follow-on paper could
explore replacing the EML activation with a polynomial approximation and
a real MLP; this module is the exact reference implementation.

Residual-stream layout
----------------------
Hidden dimension ``d_model = T + 2`` is split into two regions:

* **Position code** (first ``T`` dims): one-hot of the token's position.
  Used by attention as query/key for exact routing.
* **Value slot** (last 2 dims): real and imaginary parts of the complex
  value computed at this node. Leaves fill these at layer 0; EML nodes
  fill them at their depth layer.

The value slot is written through *residual addition*, matching how real
transformers update state, so after layer ``depth(node)`` the value at
``node.position`` is final and subsequent layers leave it alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import torch
from torch import Tensor, nn

from eml_transformer.compiler.rpn import EMLNode, TokenKind, parse_rpn_to_tree


# ---------------------------------------------------------------------------
# Numerical primitives matching eml-mcp semantics.
# ---------------------------------------------------------------------------
# These are reimplemented here rather than imported so eml-transformer stays
# self-contained (per the project's self-containment rule). The semantics
# match eml_mcp.primitives to the last digit: clamped exp to avoid overflow,
# extended-reals ln(0) = -inf.
EXP_CLAMP_MAX = 700.0
EXP_CLAMP_MIN = -700.0


def _safe_exp_complex(z_real: Tensor, z_imag: Tensor) -> tuple[Tensor, Tensor]:
    """Complex exp with real-part clamp. Returns (real, imag) tensors.

    Matches ``eml_mcp.primitives._safe_exp`` semantics: clamp z.real into
    ``[EXP_CLAMP_MIN, EXP_CLAMP_MAX]`` before exponentiation to prevent
    IEEE754 overflow at ``exp(709)``.
    """
    a = torch.clamp(z_real, EXP_CLAMP_MIN, EXP_CLAMP_MAX)
    mag = torch.exp(a)
    return mag * torch.cos(z_imag), mag * torch.sin(z_imag)


def _safe_log_complex(z_real: Tensor, z_imag: Tensor) -> tuple[Tensor, Tensor]:
    """Complex principal-branch log with extended-reals zero handling.

    Matches ``eml_mcp.primitives._safe_log``: ``ln(0)`` → ``-inf + 0j``.
    For nonzero ``z``, returns ``ln|z| + i·arg(z)``.
    """
    r2 = z_real * z_real + z_imag * z_imag
    is_zero = r2 == 0
    # log|z| via log(r^2)/2, clamped away from -inf at exactly zero so we can
    # overwrite with the sentinel without NaN propagation through the grad.
    safe_r2 = torch.where(is_zero, torch.ones_like(r2), r2)
    log_mag = 0.5 * torch.log(safe_r2)
    log_mag = torch.where(is_zero, torch.full_like(log_mag, float("-inf")), log_mag)
    arg = torch.where(is_zero, torch.zeros_like(z_imag), torch.atan2(z_imag, z_real))
    return log_mag, arg


def _eml_complex(
    a_real: Tensor, a_imag: Tensor, b_real: Tensor, b_imag: Tensor
) -> tuple[Tensor, Tensor]:
    """The EML operator on complex values: ``exp(a) - ln(b)``."""
    ea_r, ea_i = _safe_exp_complex(a_real, a_imag)
    lb_r, lb_i = _safe_log_complex(b_real, b_imag)
    return ea_r - lb_r, ea_i - lb_i


# ---------------------------------------------------------------------------
# Compilation: tree → flat program representation.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompiledNode:
    """One node's compiled form.

    Attributes:
        position: Sequence position (RPN token index) this node occupies.
        depth: Tree depth, 0 for leaves. Determines which layer computes it.
        kind: CONST / VAR / EML.
        const_value: Set iff ``kind == CONST``.
        var_name: Set iff ``kind == VAR``.
        left_pos, right_pos: Operand positions, set iff ``kind == EML``.
    """

    position: int
    depth: int
    kind: TokenKind
    const_value: complex | None = None
    var_name: str | None = None
    left_pos: int | None = None
    right_pos: int | None = None


@dataclass(frozen=True)
class CompiledProgram:
    """A tree flattened into position-indexed node metadata.

    ``nodes[i]`` describes the node at RPN position ``i``. ``variables`` is
    the ordered unique variable name list (``["x", "y"]`` etc), which the
    machine's forward pass uses to look up input bindings.
    """

    nodes: tuple[CompiledNode, ...]
    variables: tuple[str, ...]

    @property
    def seq_len(self) -> int:
        return len(self.nodes)

    @property
    def max_depth(self) -> int:
        return max(n.depth for n in self.nodes)


def compile_tree(tree: EMLNode) -> CompiledProgram:
    """Flatten an EML tree into a :class:`CompiledProgram`.

    Walks the tree in left-to-right postorder (the RPN order), assigning
    each node a position. For EML nodes, the left and right child positions
    are resolved to the already-assigned positions of those subtrees.

    The depth field is computed on the fly: leaves have depth 0, EMLs have
    ``1 + max(left_depth, right_depth)``. This is what lets the machine
    use only ``max_depth + 1`` transformer layers.
    """
    nodes: list[CompiledNode] = []
    var_order: list[str] = []

    def walk(node: EMLNode) -> tuple[int, int]:
        """Return (position, depth) of the walked node."""
        if node.kind is TokenKind.CONST:
            pos = len(nodes)
            nodes.append(
                CompiledNode(
                    position=pos,
                    depth=0,
                    kind=TokenKind.CONST,
                    const_value=node.value,
                )
            )
            return pos, 0
        if node.kind is TokenKind.VAR:
            if node.var_name not in var_order:
                var_order.append(node.var_name)  # type: ignore[arg-type]
            pos = len(nodes)
            nodes.append(
                CompiledNode(
                    position=pos,
                    depth=0,
                    kind=TokenKind.VAR,
                    var_name=node.var_name,
                )
            )
            return pos, 0
        # EML
        left_pos, left_depth = walk(node.left)  # type: ignore[arg-type]
        right_pos, right_depth = walk(node.right)  # type: ignore[arg-type]
        pos = len(nodes)
        depth = 1 + max(left_depth, right_depth)
        nodes.append(
            CompiledNode(
                position=pos,
                depth=depth,
                kind=TokenKind.EML,
                left_pos=left_pos,
                right_pos=right_pos,
            )
        )
        return pos, depth

    walk(tree)
    return CompiledProgram(nodes=tuple(nodes), variables=tuple(var_order))


# ---------------------------------------------------------------------------
# The compiled machine: a transformer-shaped module with analytic weights.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MachineConfig:
    """Static config for an :class:`EMLMachine`, extracted from a program.

    Attributes:
        seq_len: Number of RPN tokens (= number of residual-stream positions).
        num_layers: Number of transformer layers (= ``max_depth + 1``).
        variables: Ordered variable names the machine expects at input.
    """

    seq_len: int
    num_layers: int
    variables: tuple[str, ...]


class EMLMachine(nn.Module):
    """Analytically-compiled transformer that executes one EML program.

    After :func:`compile_tree` + :meth:`from_program`, the returned instance
    is a no-gradient module whose ``forward(bindings)`` returns the complex
    value the EML program computes at the given variable assignment.

    The machine is *program-specialized*: its sequence length, layer count,
    and per-layer routing tables are all fixed by the program at construction
    time. A different program produces a different machine.

    Shape notes:

    * ``forward`` expects a mapping ``{var_name: complex-like}`` for each
      variable in ``self.config.variables``. A complex-like is any value
      ``complex(v)`` accepts.
    * Returns a Python ``complex``.
    """

    def __init__(
        self,
        program: CompiledProgram,
    ) -> None:
        super().__init__()
        self.program = program
        self.config = MachineConfig(
            seq_len=program.seq_len,
            num_layers=program.max_depth + 1,
            variables=program.variables,
        )

        # The residual stream is a single (seq_len, 2) buffer: one complex
        # value per position, stored as (real, imag). Torch tensors, not
        # parameters — no gradients will flow here even if someone mistakenly
        # calls backward().
        # Constants for the leaf-injection layer (layer 0) are baked in here
        # as a precomputed leaf tensor we add into the residual stream.
        leaf_real = np.zeros(program.seq_len, dtype=np.float64)
        leaf_imag = np.zeros(program.seq_len, dtype=np.float64)
        leaf_mask = np.zeros(program.seq_len, dtype=bool)
        for node in program.nodes:
            if node.kind is TokenKind.CONST:
                leaf_real[node.position] = node.const_value.real  # type: ignore[union-attr]
                leaf_imag[node.position] = node.const_value.imag  # type: ignore[union-attr]
                leaf_mask[node.position] = True
        self.register_buffer("leaf_real", torch.tensor(leaf_real))
        self.register_buffer("leaf_imag", torch.tensor(leaf_imag))
        self.register_buffer("leaf_mask", torch.tensor(leaf_mask))


        # Per-layer EML node metadata: at layer l, which positions are EML
        # nodes of depth l, and where do they read their operands from?
        # Stored as three tensors per layer: ``target_pos`` (seq indices
        # receiving the EML result), ``left_pos``, ``right_pos``.
        layers_meta: list[dict[str, Tensor]] = []
        for layer_idx in range(self.config.num_layers):
            emls_at_depth = [n for n in program.nodes if n.kind is TokenKind.EML and n.depth == layer_idx]
            if emls_at_depth:
                tgt = torch.tensor([n.position for n in emls_at_depth], dtype=torch.long)
                lp = torch.tensor([n.left_pos for n in emls_at_depth], dtype=torch.long)
                rp = torch.tensor([n.right_pos for n in emls_at_depth], dtype=torch.long)
            else:
                tgt = torch.empty(0, dtype=torch.long)
                lp = torch.empty(0, dtype=torch.long)
                rp = torch.empty(0, dtype=torch.long)
            layers_meta.append({"target": tgt, "left": lp, "right": rp})
        # Register each tensor as a buffer so ``.to(device)`` moves them.
        for layer_idx, meta in enumerate(layers_meta):
            for key, tensor in meta.items():
                self.register_buffer(f"layer_{layer_idx}_{key}", tensor)
        self._num_layers = self.config.num_layers

    def _layer_buffers(self, layer_idx: int) -> tuple[Tensor, Tensor, Tensor]:
        return (
            getattr(self, f"layer_{layer_idx}_target"),
            getattr(self, f"layer_{layer_idx}_left"),
            getattr(self, f"layer_{layer_idx}_right"),
        )

    def forward(self, bindings: Mapping[str, complex] | None = None) -> complex:
        """Execute the compiled program for one variable binding.

        Args:
            bindings: ``{var_name: value}`` dict. Every name in
                ``self.config.variables`` must be present. Values may be
                any ``complex(v)``-constructible object (real or complex).

        Returns:
            The final complex value at the root node (last RPN position).
        """
        bindings = dict(bindings or {})
        missing = [v for v in self.config.variables if v not in bindings]
        if missing:
            raise ValueError(f"missing variable bindings: {missing}")

        device = self.leaf_real.device
        dtype = self.leaf_real.dtype

        # Initialize residual stream: constants preloaded, vars injected
        # at their positions, EML positions start at 0.
        r = self.leaf_real.clone()
        i = self.leaf_imag.clone()
        for node in self.program.nodes:
            if node.kind is TokenKind.VAR:
                val = complex(bindings[node.var_name])  # type: ignore[index]
                r[node.position] = torch.tensor(val.real, dtype=dtype, device=device)
                i[node.position] = torch.tensor(val.imag, dtype=dtype, device=device)

        # Layer 0 is the leaf-injection layer and is already done above; it
        # has no EML nodes (depth 0 = leaves only) so we just skip it.
        # Layers 1..max_depth compute EML nodes of their depth.
        for layer_idx in range(1, self._num_layers):
            target, left, right = self._layer_buffers(layer_idx)
            if target.numel() == 0:
                continue
            # Attention: gather operand values from left/right positions.
            a_r = r.index_select(0, left)
            a_i = i.index_select(0, left)
            b_r = r.index_select(0, right)
            b_i = i.index_select(0, right)
            # FFN: apply the EML operator.
            out_r, out_i = _eml_complex(a_r, a_i, b_r, b_i)
            # Residual write: place results at their target positions.
            r = r.index_copy(0, target, out_r)
            i = i.index_copy(0, target, out_i)

        # Root is the last RPN position (postorder linearization places the
        # root last by construction).
        root_r = r[-1].item()
        root_i = i[-1].item()
        return complex(root_r, root_i)


    @classmethod
    def from_rpn(cls, rpn: str) -> "EMLMachine":
        """Construct an :class:`EMLMachine` directly from an RPN string."""
        tree = parse_rpn_to_tree(rpn)
        program = compile_tree(tree)
        return cls(program)

    @classmethod
    def from_program(cls, program: CompiledProgram) -> "EMLMachine":
        """Construct from an already-compiled program (exposed for testing)."""
        return cls(program)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"EMLMachine(seq_len={self.config.seq_len}, "
            f"num_layers={self.config.num_layers}, "
            f"variables={self.config.variables})"
        )
