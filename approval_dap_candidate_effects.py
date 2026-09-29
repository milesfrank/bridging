"""Approval analogues of the paper's diversity, agreement, and polarization.

For a weighted approval profile V and a ballot metric d, let kappa_k be the
minimum total distance from voters to their closest one of k representative
ballots.  Following the scalable implementation bundled with the IJCAI23
paper, this script estimates the kappa curve by greedy k-medoids among observed
ballot types.  It then computes

    agreement   = 1 - 2*kappa_1 / (n*diameter)
    diversity   = sum_k (kappa_k/k) / (n*diameter)
    polarization = 2*(kappa_1-kappa_2) / (n*diameter)

using diameter m for raw Hamming distance on m candidates and diameter 1 for
Jaccard distance.  Candidate effects are recalculated on the projected profile
after deleting that candidate and are reported as after-minus-original.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from sample_approval_rankings import ApprovalBallot, ApprovalElection, parse_preflib_cat


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASETS = (
    ROOT / "00026_frenchapproval",
    ROOT / "00073_frenchapproval",
)
METHOD = "greedy_k_medoids_among_observed_ballot_types"
NORMALIZATION_NOTE = (
    "Direct substitution into the paper's normalization. Jaccard agreement "
    "can be negative because Jaccard 1-median cost can exceed n*diameter/2."
)


@dataclass(frozen=True)
class ApprovalDapScores:
    agreement: float
    diversity: float
    polarization: float
    kappa_1: float
    kappa_2: float
    num_voters: int
    num_candidates: int
    num_unique_ballots: int
    metric: str
    diameter: float


def pool_elections(
    elections: Sequence[ApprovalElection], title: str, election_id: str
) -> ApprovalElection:
    if not elections:
        raise ValueError("at least one election is required")
    names = elections[0].candidate_names
    if any(election.candidate_names != names for election in elections[1:]):
        raise ValueError("pooled elections must use the same candidate slate")
    counts: Counter[tuple[int, ...]] = Counter()
    for election in elections:
        for ballot in election.ballots:
            counts[ballot.approved] += ballot.count
    ballots = tuple(
        ApprovalBallot(count, approved) for approved, count in sorted(counts.items())
    )
    return ApprovalElection(
        path=Path(election_id),
        title=title,
        candidate_names=names,
        ballots=ballots,
        num_voters=sum(ballot.count for ballot in ballots),
    )


def remove_candidate(election: ApprovalElection, candidate: int) -> ApprovalElection:
    """Project every approval ballot onto all candidates except ``candidate``."""
    if not 0 <= candidate < election.num_candidates:
        raise ValueError("candidate index is outside the election")
    counts: Counter[tuple[int, ...]] = Counter()
    for ballot in election.ballots:
        projected = tuple(
            value - (value > candidate)
            for value in ballot.approved
            if value != candidate
        )
        counts[projected] += ballot.count
    ballots = tuple(
        ApprovalBallot(count, approved) for approved, count in sorted(counts.items())
    )
    return ApprovalElection(
        path=election.path,
        title=election.title,
        candidate_names=election.candidate_names[:candidate]
        + election.candidate_names[candidate + 1 :],
        ballots=ballots,
        num_voters=election.num_voters,
    )


def ballot_types_and_weights(election: ApprovalElection) -> tuple[np.ndarray, np.ndarray]:
    ballot_types = np.zeros(
        (len(election.ballots), election.num_candidates), dtype=np.int16
    )
    weights = np.empty(len(election.ballots), dtype=np.float64)
    for row, ballot in enumerate(election.ballots):
        ballot_types[row, list(ballot.approved)] = 1
        weights[row] = ballot.count
    return ballot_types, weights


def approval_distance_matrix(ballot_types: np.ndarray, metric: str) -> np.ndarray:
    """Return raw Hamming counts or Jaccard dissimilarities between ballot rows."""
    ballot_types = np.asarray(ballot_types)
    if ballot_types.ndim != 2 or len(ballot_types) == 0:
        raise ValueError("ballot_types must be a non-empty matrix")
    if not np.all((ballot_types == 0) | (ballot_types == 1)):
        raise ValueError("approval ballots must be binary")
    numeric = ballot_types.astype(np.int64, copy=False)
    intersections = numeric @ numeric.T
    sizes = numeric.sum(axis=1)
    if metric == "hamming":
        return (
            sizes[:, None] + sizes[None, :] - 2 * intersections
        ).astype(np.float64)
    if metric == "jaccard":
        unions = sizes[:, None] + sizes[None, :] - intersections
        similarities = np.ones_like(unions, dtype=np.float64)
        np.divide(intersections, unions, out=similarities, where=unions != 0)
        distances = 1.0 - similarities
        np.fill_diagonal(distances, 0.0)
        return distances
    raise ValueError("metric must be 'hamming' or 'jaccard'")


def greedy_k_medoids_costs(
    distances: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return the greedy among-observed-ballots kappa curve.

    ``costs[k-1]`` is the weighted distance to the closest of the first k
    greedily selected medoids.  Once every unique type is selected the cost is
    zero, so the omitted part of the paper's sum is also zero.
    """
    distances = np.asarray(distances, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distances must be square")
    if weights.shape != (len(distances),) or np.any(weights <= 0):
        raise ValueError("weights must be positive and match distances")

    num_types = len(distances)
    nearest = np.full(num_types, np.inf)
    available = np.ones(num_types, dtype=bool)
    costs = np.empty(num_types, dtype=np.float64)
    chosen: list[int] = []
    for step in range(num_types):
        candidate_costs = (
            np.minimum(distances, nearest[None, :]) * weights[None, :]
        ).sum(axis=1)
        candidate_costs[~available] = np.inf
        medoid = int(np.argmin(candidate_costs))
        chosen.append(medoid)
        available[medoid] = False
        nearest = np.minimum(nearest, distances[medoid])
        costs[step] = float(nearest @ weights)
        if costs[step] <= 1e-12:
            costs[step:] = 0.0
            break
    return costs, tuple(chosen)


def calculate_scores(election: ApprovalElection, metric: str) -> ApprovalDapScores:
    if election.num_candidates <= 0 or election.num_voters <= 0:
        raise ValueError("election must contain candidates and voters")
    ballot_types, weights = ballot_types_and_weights(election)
    distances = approval_distance_matrix(ballot_types, metric)
    kappa, _ = greedy_k_medoids_costs(distances, weights)
    diameter = float(election.num_candidates if metric == "hamming" else 1.0)
    scale = election.num_voters * diameter
    kappa_1 = float(kappa[0])
    kappa_2 = float(kappa[1]) if len(kappa) > 1 else 0.0
    diversity = float(
        sum(cost / k for k, cost in enumerate(kappa, start=1)) / scale
    )
    return ApprovalDapScores(
        agreement=1.0 - 2.0 * kappa_1 / scale,
        diversity=diversity,
        polarization=2.0 * (kappa_1 - kappa_2) / scale,
        kappa_1=kappa_1,
        kappa_2=kappa_2,
        num_voters=election.num_voters,
        num_candidates=election.num_candidates,
        num_unique_ballots=len(election.ballots),
        metric=metric,
        diameter=diameter,
    )


def _score_rows(
    election_id: str,
    election: ApprovalElection,
    metrics: Sequence[str],
) -> Iterable[dict[str, object]]:
    for metric in metrics:
        print(
            f"{election_id} [{metric}]: baseline with "
            f"{election.num_candidates} candidates, {election.num_voters} voters, "
            f"{len(election.ballots)} ballot types"
        )
        original = calculate_scores(election, metric)
        yield _make_row(election_id, election, metric, None, original, original)
        for candidate, candidate_name in enumerate(election.candidate_names):
            projected = remove_candidate(election, candidate)
            after = calculate_scores(projected, metric)
            print(
                f"  remove {candidate_name}: "
                f"dA={after.agreement-original.agreement:+.6f}, "
                f"dD={after.diversity-original.diversity:+.6f}, "
                f"dP={after.polarization-original.polarization:+.6f}"
            )
            yield _make_row(
                election_id,
                election,
                metric,
                candidate,
                after,
                original,
            )


def _make_row(
    election_id: str,
    original_election: ApprovalElection,
    metric: str,
    removed_candidate: Optional[int],
    scores: ApprovalDapScores,
    original: ApprovalDapScores,
) -> dict[str, object]:
    baseline = removed_candidate is None
    return {
        "election_id": election_id,
        "election_title": original_election.title,
        "metric": metric,
        "method": METHOD,
        "normalization_note": NORMALIZATION_NOTE,
        "removed_candidate_id": "" if baseline else removed_candidate + 1,
        "removed_candidate": ""
        if baseline
        else original_election.candidate_names[removed_candidate],
        "is_baseline": baseline,
        "num_voters": scores.num_voters,
        "num_candidates_after": scores.num_candidates,
        "num_unique_ballots_after": scores.num_unique_ballots,
        "metric_diameter": scores.diameter,
        "agreement": scores.agreement,
        "diversity": scores.diversity,
        "polarization": scores.polarization,
        "kappa_1": scores.kappa_1,
        "kappa_2": scores.kappa_2,
        "original_agreement": original.agreement,
        "original_diversity": original.diversity,
        "original_polarization": original.polarization,
        "delta_agreement": scores.agreement - original.agreement,
        "delta_diversity": scores.diversity - original.diversity,
        "delta_polarization": scores.polarization - original.polarization,
    }


def load_analysis_elections(
    datasets: Sequence[Path], include_individual_00026: bool = False
) -> list[tuple[str, ApprovalElection]]:
    result: list[tuple[str, ApprovalElection]] = []
    for dataset in datasets:
        paths = sorted(Path(dataset).glob("*.cat"))
        if not paths:
            raise FileNotFoundError(f"no .cat files found in {dataset}")
        elections = [parse_preflib_cat(path) for path in paths]
        dataset_name = Path(dataset).name
        if dataset_name.startswith("00026"):
            pooled = pool_elections(
                elections,
                "French 2002 approval (6 districts pooled)",
                "French_00026_grouped",
            )
            result.append(("French_00026_grouped", pooled))
            if include_individual_00026:
                result.extend((election.path.stem, election) for election in elections)
        else:
            result.extend((election.path.stem, election) for election in elections)
    return result


def run_analysis(
    datasets: Sequence[Path] = DEFAULT_DATASETS,
    output_path: Path = ROOT
    / "preference_map_outputs"
    / "approval_dap_candidate_effects.csv",
    metrics: Sequence[str] = ("hamming", "jaccard"),
    include_individual_00026: bool = False,
) -> Path:
    elections = load_analysis_elections(datasets, include_individual_00026)
    rows: list[dict[str, object]] = []
    for election_id, election in elections:
        rows.extend(_score_rows(election_id, election, metrics))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output_path}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute approval diversity, agreement, and polarization with "
            "Hamming/Jaccard distances, including candidate-removal effects."
        )
    )
    parser.add_argument("--datasets", nargs="+", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "preference_map_outputs"
        / "approval_dap_candidate_effects.csv",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=("hamming", "jaccard"),
        default=("hamming", "jaccard"),
    )
    parser.add_argument(
        "--include-individual-00026",
        action="store_true",
        help="Also analyze each 00026 district separately (the pooled profile is always included)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_analysis(
        args.datasets,
        args.output,
        args.metrics,
        args.include_individual_00026,
    )


if __name__ == "__main__":
    main()
