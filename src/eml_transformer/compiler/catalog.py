"""Load and iterate over the EML formula catalog.

The catalog lives at ``eml-mcp/eml_formulas.db`` and contains 27 verified
EML expressions — the "standard library" an :class:`EMLMachine` can be
compiled from. Each row records:

* ``name`` — short identifier (``sin``, ``cos``, ``ln``, ``discovered_*``).
* ``rpn`` — whitespace-separated RPN token string for the tree.
* ``variables`` — JSON list of input variable names.
* ``signature`` — JSON list of ``{real, imag}`` dicts, the values the
  formula takes at ``eml_mcp.primitives.TEST_POINTS``. Used for verification.
* ``depth``, ``k`` — tree depth and Kolmogorov complexity.

The :class:`CatalogEntry` dataclass parses and exposes these fields; the
:func:`load_catalog` helper returns all rows from a given DB path.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

# Canonical default catalog location relative to this workspace.
# Both projects live under ai_workspace/; eml-transformer imports the
# catalog as read-only data, not as a Python module, so no circular
# dependency risk.
DEFAULT_CATALOG_PATH = Path(
    "/home/ty/Repositories/ai_workspace/eml-mcp/eml_formulas.db"
)

# The six transcendental test points used by eml-mcp's verifier. Matching
# these exactly is the verification contract. Order matters: signatures
# are stored as a list in this order.
TEST_POINTS: tuple[complex, ...] = (
    complex(0.5772156649015329),   # Euler-Mascheroni
    complex(1.2824271291006226),   # Glaisher-Kinkelin
    complex(1.4142135623730951),   # sqrt(2)
    complex(1.6180339887498949),   # Golden ratio
    complex(2.5),
    complex(0.1),
)


@dataclass(frozen=True)
class CatalogEntry:
    """One row of ``eml_formulas.db``, parsed.

    Attributes:
        name: Identifier like ``"sin"`` or ``"discovered_d432aaea"``.
        description: Human-readable description string.
        rpn: RPN token string ready for :func:`parse_rpn_to_tree`.
        expression: Nested ``eml(...)`` form for display.
        variables: Input variable names in canonical order.
        depth: Tree depth as stored in the catalog.
        k: Kolmogorov complexity as stored.
        signature: The value the function takes at each :data:`TEST_POINTS`
            entry, as a tuple of ``complex``. Used for verification.
        note: Free-form note (e.g. "Simplified from K=171 to K=39 by ...").
    """

    name: str
    description: str
    rpn: str
    expression: str
    variables: tuple[str, ...]
    depth: int
    k: int
    signature: tuple[complex, ...]
    note: str | None = None


def _parse_signature(raw: str) -> tuple[complex, ...]:
    """Parse the JSON-encoded signature field into a tuple of complex.

    Signatures in the DB look like ``[{"real": ..., "imag": ...}, ...]``.
    A missing or unparseable field becomes an empty tuple.
    """
    if not raw:
        return ()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return ()
    out: list[complex] = []
    for entry in data:
        if isinstance(entry, dict) and "real" in entry and "imag" in entry:
            out.append(complex(entry["real"], entry["imag"]))
        else:  # unlikely but don't crash
            return ()
    return tuple(out)


def _parse_variables(raw: str) -> tuple[str, ...]:
    """Parse the JSON-encoded variable list field."""
    if not raw:
        return ()
    try:
        return tuple(json.loads(raw))
    except json.JSONDecodeError:
        return ()


def load_catalog(
    db_path: Path | str | None = None,
    *,
    include_names: set[str] | None = None,
    exclude_names: set[str] | None = None,
) -> list[CatalogEntry]:
    """Load all catalog entries from the eml-mcp SQLite database.

    Args:
        db_path: Path to ``eml_formulas.db``. Defaults to the canonical
            location under ``eml-mcp/``.
        include_names: If set, only return entries whose name is in this set.
        exclude_names: If set, drop entries whose name is in this set.

    Returns:
        List of :class:`CatalogEntry`, ordered by the DB's natural row order
        (chronological by ``created_at`` in practice).
    """
    path = Path(db_path) if db_path is not None else DEFAULT_CATALOG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"catalog not found at {path}. "
            "Provide db_path or ensure eml-mcp is checked out at the expected location."
        )
    con = sqlite3.connect(str(path))
    try:
        cur = con.execute(
            "SELECT name, description, rpn, expression, variables, "
            "depth, k, signature, note FROM formulas"
        )
        out: list[CatalogEntry] = []
        for row in cur.fetchall():
            name = row[0]
            if include_names is not None and name not in include_names:
                continue
            if exclude_names is not None and name in exclude_names:
                continue
            out.append(
                CatalogEntry(
                    name=name,
                    description=row[1] or "",
                    rpn=row[2] or "",
                    expression=row[3] or "",
                    variables=_parse_variables(row[4] or ""),
                    depth=int(row[5] or 0),
                    k=int(row[6] or 0),
                    signature=_parse_signature(row[7] or ""),
                    note=row[8],
                )
            )
        return out
    finally:
        con.close()


def load_entry(name: str, db_path: Path | str | None = None) -> CatalogEntry:
    """Convenience: load a single entry by name, raising KeyError if missing."""
    entries = load_catalog(db_path=db_path, include_names={name})
    if not entries:
        raise KeyError(f"catalog has no entry named {name!r}")
    return entries[0]


def signature_bindings(
    test_point: complex, variables: tuple[str, ...]
) -> dict[str, complex]:
    """Reproduce eml-mcp's signature binding convention.

    For a formula with variables ``variables`` evaluated at ``test_point p``,
    eml-mcp sorts the variable names alphabetically and binds the j-th
    variable (0-indexed) to::

        p * (1.1 + 0.13*j) + 0.071*j

    Constant-only formulas are treated as if they had a single variable ``"x"``.

    This convention lives in ``eml_mcp.trees.EMLNode.to_signature`` — matching
    it exactly is the contract the catalog's signatures encode, so every
    verifier in the ecosystem must use this binding function.

    Args:
        test_point: One of :data:`TEST_POINTS` (or any complex).
        variables: The formula's variable tuple (unsorted; we sort internally).

    Returns:
        A dict suitable for passing to ``EMLMachine.forward()``.
    """
    vars_sorted = sorted(variables) if variables else ["x"]
    return {
        v: test_point * (1.1 + 0.13 * j) + 0.071 * j
        for j, v in enumerate(vars_sorted)
    }
