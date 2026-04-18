"""RPN (reverse Polish notation) parsing for EML trees.

The catalog stores EML expressions as space-separated RPN token strings:

* ``x 1.0 E``              — ``eml(x, 1)`` = ``exp(x)``
* ``1.0 1.0 x E 1.0 E E``  — ``eml(1, eml(eml(1, x), 1))`` = ``ln(x)``

Tokens:

* ``E``                    — apply the EML operator to the two previous stack
                             values: ``eml(left, right)`` where ``left`` was
                             pushed first.
* A variable name          — push the named variable (``x``, ``y``, ``z``).
* A numeric literal        — push a constant. Real literals look like ``1.0``
                             or ``-697.281718171541``; complex literals look
                             like ``(0.451582705289455+1.5707963267948966j)``.

This module provides:

* :class:`Token` / :class:`TokenKind` — a structured RPN token.
* :func:`tokenize_rpn`    — split a string into tokens.
* :func:`parse_rpn_to_tree` — build an :class:`EMLNode` from an RPN string.
* :func:`tree_to_rpn`     — inverse; linearizes a tree back to an RPN string.

The RPN parser is deliberately strict: malformed input (extra tokens,
unknown identifiers, an ``E`` with fewer than two operands on the stack)
raises :class:`RPNParseError` with enough context to locate the issue in
a catalog row. This keeps the compiler's failure modes clear — silent
"evaluated something plausible-looking" is much worse than a loud parse
error pointing at the offending token.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class RPNParseError(ValueError):
    """Raised when an RPN token string can't be parsed into a valid tree."""


class TokenKind(Enum):
    """The four categories of RPN token.

    The core EML grammar has only three (CONST / VAR / EML). CALL is a
    compositional extension: it lets one catalog entry appear as a single
    token that references another entry by name, so the learned generator
    can emit ``sin ln x`` instead of the 39-token expanded ``sin`` tree.
    CALL tokens are resolved away by :func:`compose` / :func:`expand_calls`
    before the tree reaches the compiler — the machine itself only executes
    the pure CONST / VAR / EML grammar.
    """

    EML = "eml"        # the operator token "E"
    CONST = "const"    # a numeric literal (real or complex)
    VAR = "var"        # a variable name (one of {x, y, z, ...})
    CALL = "call"      # reference to a named catalog entry (unresolved)


@dataclass(frozen=True)
class Token:
    """One RPN token, post-tokenization.

    Exactly one of ``value``, ``var_name``, ``call_name`` is non-None
    depending on :attr:`kind`; ``EML`` tokens have none of them. The raw
    source string is preserved in :attr:`raw` for error messages.
    """

    kind: TokenKind
    raw: str
    value: complex | None = None
    var_name: str | None = None
    call_name: str | None = None


def _parse_numeric(tok: str) -> complex:
    """Parse a numeric RPN token as a complex number.

    Accepts:

    * Real literals: ``1.0``, ``-0.3665129205816644``, ``2.718281828459045``.
    * Python-style complex literals with explicit parentheses:
      ``(0.451582705289455+1.5707963267948966j)``.
    * Bare complex literals without parens: ``0.45+1.57j`` (just in case).

    Raises :class:`RPNParseError` on anything else.
    """
    raw = tok.strip()
    # Strip a single pair of surrounding parens if present.
    if raw.startswith("(") and raw.endswith(")"):
        raw = raw[1:-1]
    try:
        # Complex first — Python's complex() accepts real-only strings too,
        # but we route real literals through float() for slightly better
        # precision of common values like "1.0".
        if "j" in raw.lower():
            return complex(raw)
        return complex(float(raw))
    except ValueError as exc:
        raise RPNParseError(f"could not parse numeric token {tok!r}") from exc


def tokenize_rpn(
    rpn: str,
    *,
    call_names: frozenset[str] | None = None,
) -> list[Token]:
    """Split an RPN string into structured tokens.

    The catalog's RPN format is whitespace-separated, but complex literals
    contain parentheses (``(0.45+1.57j)``) which may or may not include
    internal whitespace depending on how they were serialized. This tokenizer
    handles both cases by treating a ``(`` as opening a "protected" region
    that ends at the matching ``)``, allowing whitespace within.

    Args:
        rpn: The whitespace-separated RPN source string.
        call_names: Optional set of catalog-entry names that should be
            tokenized as ``CALL`` tokens rather than ``VAR`` tokens. When
            None (default), any identifier becomes a VAR — this preserves
            backward compatibility with raw catalog RPN strings, which never
            contain CALL tokens. Note: identifying something as a CALL is
            purely lexical here; arity-checking and resolution happen at
            :func:`parse_rpn_to_tree` / :func:`expand_calls` time.

    Returns:
        A list of :class:`Token` objects in source order.

    Raises:
        RPNParseError: On unbalanced parens or unrecognized token forms.
    """
    # First pass: split into raw token strings, respecting parens.
    raws: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in rpn:
        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth -= 1
            if depth < 0:
                raise RPNParseError("unbalanced close-paren in RPN string")
            buf.append(ch)
        elif ch.isspace() and depth == 0:
            if buf:
                raws.append("".join(buf))
                buf = []
        else:
            buf.append(ch)
    if depth != 0:
        raise RPNParseError("unbalanced open-paren in RPN string")
    if buf:
        raws.append("".join(buf))

    # Second pass: classify each raw token.
    out: list[Token] = []
    for r in raws:
        if r == "E":
            out.append(Token(kind=TokenKind.EML, raw=r))
        elif _looks_numeric(r):
            out.append(Token(kind=TokenKind.CONST, raw=r, value=_parse_numeric(r)))
        elif r.isidentifier():
            if call_names is not None and r in call_names:
                out.append(Token(kind=TokenKind.CALL, raw=r, call_name=r))
            else:
                out.append(Token(kind=TokenKind.VAR, raw=r, var_name=r))
        else:
            raise RPNParseError(f"unrecognized RPN token {r!r}")
    return out


def _looks_numeric(tok: str) -> bool:
    """Cheap predicate: does this string look like a numeric literal?

    Accepts signed digits, decimals, exponent markers, ``j`` for imaginary
    unit, and wrapping parens. Does *not* accept bare identifiers; those
    go to the VAR branch. A stricter check happens in :func:`_parse_numeric`.
    """
    stripped = tok.strip("()")
    if not stripped:
        return False
    # Has to start with a digit, sign, or decimal point — rules out idents.
    return stripped[0].isdigit() or stripped[0] in "+-." or "j" in stripped.lower()


@dataclass
class EMLNode:
    """Minimal EML tree node for the compiler.

    Self-contained (no import from eml-mcp) to keep eml-transformer's
    dependencies tight. Semantically identical to :class:`eml_mcp.trees.EMLNode`
    for the CONST / VAR / EML cases. The CALL case is a compositional
    extension — it names another catalog entry and supplies argument
    subtrees by position. CALL nodes must be resolved by
    :func:`eml_transformer.compiler.composer.expand_calls` before the tree
    reaches the compiler; the machine itself only executes CONST/VAR/EML.

    Attributes:
        kind: The node kind (CONST/VAR/EML/CALL).
        value: The complex constant for ``CONST`` leaves.
        var_name: The variable name for ``VAR`` leaves.
        left, right: The two subtrees for ``EML`` internal nodes. ``left``
            feeds ``exp`` (the first argument); ``right`` feeds ``ln``.
        call_name: The catalog-entry name for ``CALL`` nodes (e.g. ``"sin"``).
        call_args: Positional argument subtrees for a ``CALL`` node,
            in the order matching the callee's sorted variable list.
    """

    kind: TokenKind
    value: complex | None = None
    var_name: str | None = None
    left: "EMLNode | None" = None
    right: "EMLNode | None" = None
    call_name: str | None = None
    call_args: "tuple[EMLNode, ...] | None" = None

    def depth(self) -> int:
        """Maximum subtree depth. Leaves have depth 0.

        For a tree containing unresolved CALL nodes, this returns the
        *structural* depth of the written form — not the depth after
        expansion. Call :func:`expand_calls` first if you want
        post-expansion depth.
        """
        if self.kind is TokenKind.EML:
            return 1 + max(self.left.depth(), self.right.depth())  # type: ignore[union-attr]
        if self.kind is TokenKind.CALL:
            if not self.call_args:
                return 1
            return 1 + max(a.depth() for a in self.call_args)
        return 0

    def size(self) -> int:
        """Total node count including self (pre-expansion for CALL nodes)."""
        if self.kind is TokenKind.EML:
            return 1 + self.left.size() + self.right.size()  # type: ignore[union-attr]
        if self.kind is TokenKind.CALL:
            return 1 + sum(a.size() for a in (self.call_args or ()))
        return 1

    def has_calls(self) -> bool:
        """True iff any subtree contains a CALL node."""
        if self.kind is TokenKind.CALL:
            return True
        if self.kind is TokenKind.EML:
            return self.left.has_calls() or self.right.has_calls()  # type: ignore[union-attr]
        return False


def parse_rpn_to_tree(
    rpn: str,
    *,
    call_arities: dict[str, int] | None = None,
) -> EMLNode:
    """Build an :class:`EMLNode` tree from an RPN token string.

    Standard stack-based RPN: scan tokens left to right, push leaves onto
    a stack, and at each operator pop its arity in operands and push a
    composed node. The stack should have exactly one element at the end —
    that's the tree root.

    For CALL-token support (optional), pass ``call_arities`` mapping each
    known catalog name to its variable count. The parser will then recognize
    those names as CALL tokens, pop that many operands, and emit a CALL
    node whose ``call_args`` are those operands in input order.

    Args:
        rpn: Whitespace-separated RPN source.
        call_arities: Optional ``{name: arity}`` mapping enabling CALL
            tokens. When None, identifiers become VARs (backward-compat
            with raw catalog RPN).

    Returns:
        The root :class:`EMLNode`. May contain unresolved CALL nodes if
        ``call_arities`` was provided; call :func:`expand_calls` to resolve.

    Raises:
        RPNParseError: On parse failure or stack underflow.
    """
    call_names = frozenset(call_arities.keys()) if call_arities else None
    tokens = tokenize_rpn(rpn, call_names=call_names)
    if not tokens:
        raise RPNParseError("empty RPN string")

    stack: list[EMLNode] = []
    for i, tok in enumerate(tokens):
        if tok.kind is TokenKind.CONST:
            stack.append(EMLNode(kind=TokenKind.CONST, value=tok.value))
        elif tok.kind is TokenKind.VAR:
            stack.append(EMLNode(kind=TokenKind.VAR, var_name=tok.var_name))
        elif tok.kind is TokenKind.EML:
            if len(stack) < 2:
                raise RPNParseError(
                    f"at token {i} ({tok.raw!r}): EML requires 2 operands, "
                    f"stack has {len(stack)}"
                )
            right = stack.pop()
            left = stack.pop()
            stack.append(EMLNode(kind=TokenKind.EML, left=left, right=right))
        elif tok.kind is TokenKind.CALL:
            assert call_arities is not None  # tokenizer guarantees this
            arity = call_arities[tok.call_name]  # type: ignore[index]
            if len(stack) < arity:
                raise RPNParseError(
                    f"at token {i} ({tok.raw!r}): call to {tok.call_name!r} "
                    f"requires {arity} operands, stack has {len(stack)}"
                )
            args = tuple(stack[-arity:]) if arity > 0 else ()
            if arity > 0:
                del stack[-arity:]
            stack.append(
                EMLNode(
                    kind=TokenKind.CALL,
                    call_name=tok.call_name,
                    call_args=args,
                )
            )
        else:  # pragma: no cover - defensive
            raise RPNParseError(f"unknown token kind {tok.kind}")

    if len(stack) != 1:
        raise RPNParseError(
            f"RPN did not reduce to a single tree: {len(stack)} items left on stack"
        )
    return stack[0]


def tree_to_rpn(tree: EMLNode) -> str:
    """Linearize a tree back into whitespace-separated RPN. Inverse of parse."""
    tokens: list[str] = []
    _linearize(tree, tokens)
    return " ".join(tokens)


def _linearize(node: EMLNode, out: list[str]) -> None:
    if node.kind is TokenKind.CONST:
        out.append(_format_const(node.value))
    elif node.kind is TokenKind.VAR:
        out.append(node.var_name or "?")
    elif node.kind is TokenKind.EML:
        if node.left is None or node.right is None:  # pragma: no cover
            raise RPNParseError("EML node missing child")
        _linearize(node.left, out)
        _linearize(node.right, out)
        out.append("E")
    elif node.kind is TokenKind.CALL:
        for arg in node.call_args or ():
            _linearize(arg, out)
        out.append(node.call_name or "?")


def _format_const(val: complex | None) -> str:
    """Serialize a constant the way the catalog stores it.

    Real constants render as ``float`` reprs (``1.0``, ``-697.281718171541``).
    Complex constants render as parenthesized Python complex literals.
    """
    if val is None:
        return "?"
    if val.imag == 0.0:
        return repr(val.real)
    return f"({val.real!r}{'+' if val.imag >= 0 else ''}{val.imag!r}j)"


def iter_leaves(tree: EMLNode) -> Iterable[EMLNode]:
    """Yield all leaf (CONST/VAR) nodes in left-to-right postorder.

    CALL nodes are traversed through (their arg subtrees are walked) but
    not yielded — they aren't leaves in the pre-expansion tree.
    """
    if tree.kind is TokenKind.EML:
        yield from iter_leaves(tree.left)  # type: ignore[arg-type]
        yield from iter_leaves(tree.right)  # type: ignore[arg-type]
    elif tree.kind is TokenKind.CALL:
        for arg in tree.call_args or ():
            yield from iter_leaves(arg)
    else:
        yield tree
