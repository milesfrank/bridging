"""Run proportional and minimum-gamma audits for a list of alternatives."""

from __future__ import annotations

import argparse
import csv
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

from generic_dc_mpjr_min_gamma import minimum_gamma_dc_mpjr_indices
from proportional_audit import audit_exact, load_csv, resolve_candidate


OUTPUT_COLUMNS = (
    "alternative",
    # "score",
    "approver_count",
    # "population_size",
    "rho",
    "minimum_gamma",
    # "rho_worst_candidate_row",
    # "coalition_entitlement",
)


def load_alternatives(path: Path) -> list[str]:
    """Load one alternative per line, ignoring blank lines."""
    with path.open(encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def output_sort_key(
    row: dict[str, str | int | float], column: str
) -> str | int | float | Fraction:
    """Return a correctly typed sort key for an output column."""
    value = row[column]
    if column == "alternative":
        return str(value).casefold()
    if column == "minimum_gamma":
        return float("inf") if value == "infinity" else Fraction(str(value))
    return value


def audit_alternative(
    names: list[str], points: np.ndarray, alternative: str
) -> dict[str, str | int | float]:
    """Audit one alternative and return a row suitable for CSV output."""
    column = resolve_candidate(names, alternative)
    approving = points[:, column] == 1
    approver_count = int(approving.sum())
    if approver_count == 0:
        raise ValueError(f"No voters approved {names[column]}")

    # M is the set of potential approvers: one counterfactual center per
    # voter, obtained by forcing the audited alternative's coordinate to 1.
    potential_approvers = points.copy()
    potential_approvers[:, column] = 1
    result = audit_exact(
        points,
        points[approving],
        candidate_centers=potential_approvers,
        k=approver_count,
    )
    minimum_gamma = minimum_gamma_dc_mpjr_indices(
        agents=points,
        candidate_centers=potential_approvers,
        selected_indices=np.flatnonzero(approving),
    )
    return {
        "alternative": names[column],
        # "score": result.score,
        "approver_count": approver_count,
        # "population_size": len(points),
        "rho": result.rho,
        "minimum_gamma": (
            str(minimum_gamma.gamma)
            if minimum_gamma.finite
            else "infinity"
        ),
        # "rho_worst_candidate_row": result.worst_center,
        # "coalition_entitlement": result.coalition_size,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run proportional and minimum-gamma DC-mPJR+ audits for "
            "multiple alternatives."
        )
    )
    parser.add_argument("csv", type=Path, help="Point-matrix CSV to audit")
    parser.add_argument(
        "alternatives",
        nargs="*",
        help=(
            "Alternative names to audit, in the desired output order; "
            "defaults to every CSV column"
        ),
    )
    parser.add_argument(
        "--alternatives-file",
        type=Path,
        help="UTF-8 text file containing one alternative name per line",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write results to this CSV instead of standard output",
    )
    parser.add_argument(
        "--sort-by",
        choices=OUTPUT_COLUMNS,
        help="Sort output rows by this column",
    )
    parser.add_argument(
        "--descending",
        action="store_true",
        help="Reverse the --sort-by order",
    )
    args = parser.parse_args()
    if args.descending and args.sort_by is None:
        parser.error("--descending requires --sort-by")
    return args


def main() -> int:
    args = parse_args()
    names, points = load_csv(args.csv)

    alternatives = list(args.alternatives)
    if args.alternatives_file is not None:
        alternatives.extend(load_alternatives(args.alternatives_file))
    if not alternatives:
        alternatives = list(names)

    rows: list[dict[str, str | int | float]] = []
    failed = False
    for alternative in alternatives:
        try:
            rows.append(audit_alternative(names, points, alternative))
        except (SystemExit, ValueError) as error:
            print(f"{alternative}: {error}", file=sys.stderr)
            failed = True

    if args.sort_by is not None:
        rows.sort(
            key=lambda row: output_sort_key(row, args.sort_by),
            reverse=args.descending,
        )

    output_handle = None
    try:
        if args.output is None:
            destination = sys.stdout
        else:
            output_handle = args.output.open("w", newline="", encoding="utf-8")
            destination = output_handle
        writer = csv.DictWriter(destination, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if output_handle is not None:
            output_handle.close()

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
