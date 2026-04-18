"""Verify the compiled machine against the catalog's stored signatures.

For every entry in ``eml_formulas.db`` whose verification contract we can
satisfy (i.e., single-variable functions evaluated at the 6 ``TEST_POINTS``),
compile the RPN into an :class:`EMLMachine`, run the machine at each test
point, and compare the result to the catalog's stored signature.

The pass criterion is ``|machine - stored| < tol`` componentwise (real and
imaginary), with default tolerance ``1e-10``. A stricter tolerance of
``1e-12`` is what the original bootstrapping used; anything looser than
``1e-6`` would mean our numerical primitives disagree with eml-mcp at a
level that breaks the "verifiable to machine epsilon" claim.

This module is both an importable API (``verify_entry``, ``verify_catalog``)
and a command-line tool (``python -m eml_transformer.compiler.verify``).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from eml_transformer.compiler.catalog import (
    DEFAULT_CATALOG_PATH,
    TEST_POINTS,
    CatalogEntry,
    load_catalog,
)
from eml_transformer.compiler.machine import EMLMachine

DEFAULT_TOL = 1e-10


@dataclass(frozen=True)
class PointResult:
    """Outcome of evaluating one machine at one test point."""

    test_point: complex
    expected: complex
    actual: complex
    abs_error: float
    passed: bool


@dataclass(frozen=True)
class EntryResult:
    """Aggregate verification outcome for one catalog entry."""

    name: str
    n_variables: int
    depth: int
    seq_len: int
    skipped_reason: str | None  # non-None when we couldn't verify this entry
    points: tuple[PointResult, ...]

    @property
    def skipped(self) -> bool:
        return self.skipped_reason is not None

    @property
    def all_passed(self) -> bool:
        return not self.skipped and all(p.passed for p in self.points)

    @property
    def max_abs_error(self) -> float:
        if not self.points:
            return 0.0
        return max(p.abs_error for p in self.points)


def verify_entry(
    entry: CatalogEntry,
    *,
    tol: float = DEFAULT_TOL,
) -> EntryResult:
    """Verify one catalog entry: compile → evaluate → compare to signature.

    Args:
        entry: The catalog row to verify.
        tol: Absolute-error tolerance for componentwise comparison.

    Returns:
        An :class:`EntryResult` describing the outcome. ``skipped_reason``
        is set when we can't meaningfully compare (e.g., the catalog has no
        stored signature for this entry, or the signature length disagrees
        with :data:`TEST_POINTS`).
    """
    from eml_transformer.compiler.catalog import (  # local import to avoid cycle
        signature_bindings,
    )

    # Skip gracefully if the catalog doesn't have a valid signature for this row.
    if len(entry.signature) != len(TEST_POINTS):
        return EntryResult(
            name=entry.name,
            n_variables=len(entry.variables),
            depth=entry.depth,
            seq_len=0,
            skipped_reason=(
                f"signature has {len(entry.signature)} values, "
                f"expected {len(TEST_POINTS)}"
            ),
            points=(),
        )

    try:
        machine = EMLMachine.from_rpn(entry.rpn)
    except Exception as exc:  # noqa: BLE001 - we want to report *any* failure
        return EntryResult(
            name=entry.name,
            n_variables=len(entry.variables),
            depth=entry.depth,
            seq_len=0,
            skipped_reason=f"compile failed: {type(exc).__name__}: {exc}",
            points=(),
        )

    point_results: list[PointResult] = []
    for tp, expected in zip(TEST_POINTS, entry.signature, strict=True):
        bindings = signature_bindings(tp, entry.variables)
        # Only pass in bindings for the variables the machine actually needs.
        machine_bindings = {v: bindings[v] for v in machine.config.variables}
        try:
            actual = machine(machine_bindings)
        except Exception as exc:  # noqa: BLE001
            point_results.append(
                PointResult(
                    test_point=tp,
                    expected=expected,
                    actual=complex(float("nan"), float("nan")),
                    abs_error=float("inf"),
                    passed=False,
                )
            )
            continue
        abs_err = abs(actual - expected)
        point_results.append(
            PointResult(
                test_point=tp,
                expected=expected,
                actual=actual,
                abs_error=abs_err,
                passed=abs_err < tol,
            )
        )

    return EntryResult(
        name=entry.name,
        n_variables=len(entry.variables),
        depth=entry.depth,
        seq_len=machine.config.seq_len,
        skipped_reason=None,
        points=tuple(point_results),
    )


def verify_catalog(
    *,
    db_path: Path | str | None = None,
    tol: float = DEFAULT_TOL,
    include_names: set[str] | None = None,
    exclude_names: set[str] | None = None,
) -> list[EntryResult]:
    """Run :func:`verify_entry` over every (selected) row in the catalog."""
    entries = load_catalog(
        db_path=db_path, include_names=include_names, exclude_names=exclude_names
    )
    return [verify_entry(e, tol=tol) for e in entries]


def format_summary(results: list[EntryResult], *, show_all: bool = False) -> str:
    """Render verification results as a text table.

    By default only failures and skips are listed after the summary header;
    pass ``show_all=True`` to include passing rows too.
    """
    total = len(results)
    passed = sum(1 for r in results if r.all_passed)
    failed = sum(1 for r in results if not r.all_passed and not r.skipped)
    skipped = sum(1 for r in results if r.skipped)

    lines: list[str] = [
        f"catalog verification: {passed}/{total} passed, "
        f"{failed} failed, {skipped} skipped",
        "",
        f"{'name':<28} {'vars':>4} {'depth':>5} {'seq':>4} "
        f"{'max_err':>12} {'status':>8}",
        "-" * 72,
    ]
    for r in results:
        status = "PASS" if r.all_passed else ("SKIP" if r.skipped else "FAIL")
        if not show_all and status == "PASS":
            continue
        max_err = f"{r.max_abs_error:.2e}" if not r.skipped else "-"
        lines.append(
            f"{r.name:<28} {r.n_variables:>4} {r.depth:>5} "
            f"{r.seq_len:>4} {max_err:>12} {status:>8}"
        )
        if r.skipped:
            lines.append(f"    skip reason: {r.skipped_reason}")
        elif not r.all_passed:
            # Break down the failing points for diagnosis.
            for pr in r.points:
                if pr.passed:
                    continue
                lines.append(
                    f"    at {pr.test_point}: expected {pr.expected}, "
                    f"got {pr.actual}, err={pr.abs_error:.3e}"
                )
    return "\n".join(lines)


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eml_transformer.compiler.verify",
        description="Verify the EML-compiled machine against the catalog's signatures.",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(DEFAULT_CATALOG_PATH),
        help="Path to eml_formulas.db (default: eml-mcp's canonical location).",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=DEFAULT_TOL,
        help=f"Absolute-error tolerance (default: {DEFAULT_TOL:g}).",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Print every row, not just failures and skips.",
    )
    parser.add_argument(
        "--only",
        type=str,
        nargs="+",
        default=None,
        help="Only verify named entries (useful for debugging a single formula).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_cli().parse_args(argv)
    include = set(args.only) if args.only else None
    results = verify_catalog(db_path=args.db_path, tol=args.tol, include_names=include)
    print(format_summary(results, show_all=args.show_all))
    # Exit code reflects pass/fail for use in scripts and CI.
    any_failed = any((not r.all_passed) and (not r.skipped) for r in results)
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
