"""Measure local representation at every level below the center count.

For each voter i and representation level l < k, this audit finds the
smallest closed Hamming ball centered at i that contains at least
ceil(l * n / k) voters. The voter passes level l when that same ball contains
at least l chosen centers. Output summarizes the per-level passing portions.
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
    "average_portion",
    "minimum_portion",
    "maximum_portion",
)


@dataclass(frozen=True)
class RepresentationLevelResult:
    representation_level: int
    target_voter_count: int
    satisfied_voter_count: int
    population_size: int

    @property
    def portion(self) -> float:
        return self.satisfied_voter_count / self.population_size


@dataclass(frozen=True)
class PortionSummary:
    average_portion: float
    minimum_portion: float
    maximum_portion: float


def representation_level_audit(
    voters: np.ndarray,
    chosen_centers: np.ndarray,
    evaluation_mask: np.ndarray | None = None,
) -> list[RepresentationLevelResult]:
    """Return the passing-voter portion for every level 1 <= l < k.

    Rows represent voters and chosen centers, while columns represent the
    common features. Centers are counted with multiplicity when rows repeat.
    ``evaluation_mask`` can restrict which voters i contribute to the passing
    portion without changing the full-electorate ball thresholds.
    """
    voters = np.asarray(voters)
    chosen_centers = np.asarray(chosen_centers)
    if voters.ndim != 2 or chosen_centers.ndim != 2:
        raise ValueError("voters and chosen_centers must be two-dimensional")
    if voters.shape[0] == 0 or chosen_centers.shape[0] == 0:
        raise ValueError("voters and chosen_centers must be non-empty")
    if voters.shape[1] != chosen_centers.shape[1]:
        raise ValueError("voters and chosen_centers must have equal width")

    n = voters.shape[0]
    k = chosen_centers.shape[0]
    if k > n:
        raise ValueError("the number of chosen centers cannot exceed |N|")
    if k < 2:
        raise ValueError("at least two chosen centers are required because l < k")

    if evaluation_mask is None:
        evaluation_mask = np.ones(n, dtype=bool)
    else:
        evaluation_mask = np.asarray(evaluation_mask, dtype=bool)
        if evaluation_mask.ndim != 1 or evaluation_mask.size != n:
            raise ValueError("evaluation_mask must contain one value per voter")
    evaluated_voter_count = int(np.count_nonzero(evaluation_mask))
    if evaluated_voter_count == 0:
        raise ValueError("at least one voter must be included in the evaluation")

    # Sorting once lets each level use order statistics directly. The target
    # voter radius is the target-th distance, and the radius needed to include
    # l centers is the l-th chosen-center distance.
    evaluated_voters = voters[evaluation_mask]
    voter_distances = np.sort(
        hamming_distances(evaluated_voters, voters), axis=1
    )
    center_distances = np.sort(
        hamming_distances(evaluated_voters, chosen_centers), axis=1
    )

    results: list[RepresentationLevelResult] = []
    for level in range(1, k):
        target_voter_count = (level * n + k - 1) // k
        ball_radii = voter_distances[:, target_voter_count - 1]
        level_center_radii = center_distances[:, level - 1]
        satisfied_count = int(np.count_nonzero(level_center_radii <= ball_radii))
        results.append(
            RepresentationLevelResult(
                representation_level=level,
                target_voter_count=target_voter_count,
                satisfied_voter_count=satisfied_count,
                population_size=evaluated_voter_count,
            )
        )
    return results


def summarize_portions(
    results: list[RepresentationLevelResult],
) -> PortionSummary:
    """Return the mean, minimum, and maximum of the level portions."""
    if not results:
        raise ValueError("at least one representation level is required")
    portions = [result.portion for result in results]
    return PortionSummary(
        average_portion=sum(portions) / len(portions),
        minimum_portion=min(portions),
        maximum_portion=max(portions),
    )


def _write_summary(
    results: list[RepresentationLevelResult], destination: TextIO
) -> None:
    summary = summarize_portions(results)
    writer = csv.DictWriter(destination, fieldnames=OUTPUT_COLUMNS)
    writer.writeheader()
    writer.writerow(
        {
            "average_portion": summary.average_portion,
            "minimum_portion": summary.minimum_portion,
            "maximum_portion": summary.maximum_portion,
        }
    )


def plot_portions(
    results: list[RepresentationLevelResult],
    output_path: Path | None = None,
    show: bool = False,
    title: str = "Passing portion by representation level",
) -> None:
    """Plot passing portion against representation level."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ValueError(
            "plotting requires matplotlib; install it or omit the plot option"
        ) from error

    levels = [result.representation_level for result in results]
    portions = [result.portion for result in results]
    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.plot(levels, portions, color="#1f4e79", linewidth=2)
    axis.set_xlabel("Representation level")
    axis.set_ylabel("Passing portion")
    axis.set_title(title)
    axis.set_ylim(0, 1)
    axis.grid(axis="y", color="0.88", linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=160)
    if show:
        plt.show()
    else:
        plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report the average, minimum, and maximum voter-representation "
            "portions across levels l < k using closed Hamming balls."
        )
    )
    parser.add_argument("voters", type=Path, help="Headered voter-matrix CSV")
    parser.add_argument(
        "centers",
        type=Path,
        nargs="?",
        help="Headered chosen-centers CSV",
    )
    parser.add_argument(
        "--candidate",
        help=(
            "Use this alternative's approvers as centers and evaluate only "
            "its non-approvers"
        ),
    )
    parser.add_argument(
        "--include-approvers",
        action="store_true",
        help="Include the candidate's approvers in the evaluated voter set",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the portion summary to this CSV instead of standard output",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        metavar="PATH",
        help="Save a graph of passing portion versus level",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display the graph interactively",
    )
    args = parser.parse_args()
    if (args.centers is None) == (args.candidate is None):
        parser.error("provide either a centers CSV or --candidate, but not both")
    if args.include_approvers and args.candidate is None:
        parser.error("--include-approvers requires --candidate")
    return args


def main() -> int:
    args = parse_args()
    try:
        voter_header, voters = load_matrix(args.voters)
        if args.candidate is not None:
            if not np.all((voters == 0) | (voters == 1)):
                raise ValueError("--candidate requires a binary voter matrix")
            candidate_column = resolve_candidate(voter_header, args.candidate)
            approving = voters[:, candidate_column] == 1
            if not np.any(approving):
                raise ValueError(
                    f"No voters approved {voter_header[candidate_column]}"
                )
            chosen_centers = voters[approving]
        else:
            center_header, chosen_centers = load_matrix(args.centers)
            if center_header != voter_header:
                raise ValueError(
                    "voter and chosen-center CSVs must have identical headers"
                )

        results = representation_level_audit(
            voters,
            chosen_centers,
            evaluation_mask=(
                ~approving
                if args.candidate is not None and not args.include_approvers
                else None
            ),
        )
        if args.plot is not None or args.show_plot:
            title = "Passing portion by representation level"
            if args.candidate is not None:
                population_label = (
                    "all voters" if args.include_approvers else "non-approvers"
                )
                title = (
                    f"{voter_header[candidate_column]} {population_label}: "
                    "passing portion by level"
                )
            plot_portions(
                results,
                output_path=args.plot,
                show=args.show_plot,
                title=title,
            )
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1

    if args.output is None:
        _write_summary(results, sys.stdout)
    else:
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            _write_summary(results, handle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
