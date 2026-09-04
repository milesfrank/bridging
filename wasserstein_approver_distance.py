"""Compare all voters with an alternative's approvers using Wasserstein-1.

Each row is an empirical ballot in approval space.  Transport cost is Hamming
distance, so a distance of one means changing one approval coordinate.  The
calculation is exact up to the numerical tolerance of SciPy's HiGHS linear
programming solver.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import numpy as np

from generic_dc_mpjr_min_gamma import hamming_distances, load_matrix
from proportional_audit import resolve_candidate


OUTPUT_COLUMNS = (
    "alternative",
    "approver_count",
    "population_size",
    "wasserstein_distance",
)


@dataclass(frozen=True)
class WassersteinResult:
    alternative: str
    approver_count: int
    population_size: int
    distance: float


def _optimal_transport_cost(
    source: np.ndarray,
    source_counts: np.ndarray,
    target: np.ndarray,
    target_counts: np.ndarray,
) -> float:
    """Return Wasserstein-1 between two weighted empirical distributions."""
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except ImportError as error:
        raise ValueError(
            "Wasserstein calculation requires SciPy (pip install scipy)"
        ) from error

    source_mass = source_counts / source_counts.sum()
    target_mass = target_counts / target_counts.sum()
    source_size = len(source)
    target_size = len(target)
    variable_count = source_size * target_size

    # One variable transports mass between each pair of unique ballot types.
    # Rows enforce every source supply and target demand.  The final constraint
    # is redundant mathematically, but HiGHS safely removes that redundancy.
    variable_indices = np.arange(variable_count)
    constraint_rows = np.concatenate(
        (
            np.repeat(np.arange(source_size), target_size),
            source_size + np.tile(np.arange(target_size), source_size),
        )
    )
    constraint_columns = np.concatenate(
        (variable_indices, variable_indices)
    )
    constraints = coo_matrix(
        (
            np.ones(2 * variable_count),
            (constraint_rows, constraint_columns),
        ),
        shape=(source_size + target_size, variable_count),
    ).tocsr()

    solution = linprog(
        hamming_distances(source, target).astype(float).ravel(),
        A_eq=constraints,
        b_eq=np.concatenate((source_mass, target_mass)),
        bounds=(0, None),
        method="highs",
    )
    if not solution.success:
        raise ValueError(f"optimal transport solver failed: {solution.message}")
    return max(0.0, float(solution.fun))


def wasserstein_to_approvers(
    voters: np.ndarray,
    candidate_column: int,
    alternative: str = "candidate",
) -> WassersteinResult:
    """Return W1(full electorate, electorate conditional on approval).

    Ballot rows must be binary.  Hamming distance is used as the ground cost.
    """
    voters = np.asarray(voters)
    if voters.ndim != 2 or voters.shape[0] == 0 or voters.shape[1] == 0:
        raise ValueError("voters must be a non-empty two-dimensional matrix")
    if not 0 <= candidate_column < voters.shape[1]:
        raise ValueError("candidate column is outside the voter matrix")
    if not np.all((voters == 0) | (voters == 1)):
        raise ValueError("approver Wasserstein distance requires a binary matrix")

    approving = voters[:, candidate_column] == 1
    approver_count = int(np.count_nonzero(approving))
    population_size = len(voters)
    if approver_count == 0:
        raise ValueError(f"No voters approved {alternative}")
    if approver_count == population_size:
        distance = 0.0
    else:
        approver_types, approver_counts = np.unique(
            voters[approving], axis=0, return_counts=True
        )
        nonapprover_types, nonapprover_counts = np.unique(
            voters[~approving], axis=0, return_counts=True
        )
        conditional_distance = _optimal_transport_cost(
            nonapprover_types,
            nonapprover_counts,
            approver_types,
            approver_counts,
        )

        # P_all = a P_approvers + (1-a) P_nonapprovers.  By homogeneity of
        # Wasserstein-1, W1(P_all, P_approvers) is the non-approver share times
        # W1(P_nonapprovers, P_approvers).
        distance = (
            (population_size - approver_count)
            / population_size
            * conditional_distance
        )

    return WassersteinResult(
        alternative=alternative,
        approver_count=approver_count,
        population_size=population_size,
        distance=distance,
    )


def _write_results(results: list[WassersteinResult], destination: TextIO) -> None:
    writer = csv.DictWriter(destination, fieldnames=OUTPUT_COLUMNS)
    writer.writeheader()
    for result in results:
        writer.writerow(
            {
                "alternative": result.alternative,
                "approver_count": result.approver_count,
                "population_size": result.population_size,
                "wasserstein_distance": f"{result.distance:.12g}",
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate Wasserstein-1 distance between all voter ballots and "
            "the ballots of each alternative's approvers, using Hamming cost."
        )
    )
    parser.add_argument("csv", type=Path, help="Headered binary voter-matrix CSV")
    parser.add_argument(
        "alternatives",
        nargs="*",
        help="Alternatives to calculate; defaults to every CSV column",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write results to this CSV instead of standard output",
    )
    parser.add_argument(
        "--sort",
        action="store_true",
        help="Sort from smallest to largest Wasserstein distance",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        names, voters = load_matrix(args.csv)
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1

    requested = args.alternatives or names
    results: list[WassersteinResult] = []
    failed = False
    for query in requested:
        try:
            column = resolve_candidate(names, query)
            results.append(
                wasserstein_to_approvers(voters, column, names[column])
            )
        except (SystemExit, ValueError) as error:
            print(f"{query}: {error}", file=sys.stderr)
            failed = True

    if args.sort:
        results.sort(key=lambda result: result.distance)

    output_handle = None
    try:
        if args.output is None:
            destination = sys.stdout
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            output_handle = args.output.open("w", newline="", encoding="utf-8")
            destination = output_handle
        _write_results(results, destination)
    except OSError as error:
        print(error, file=sys.stderr)
        return 1
    finally:
        if output_handle is not None:
            output_handle.close()

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
