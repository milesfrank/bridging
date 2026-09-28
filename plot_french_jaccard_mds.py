"""Plot French approval voters with MDS and Jaccard dissimilarity.

Each point represents one sampled approval ballot.  The dissimilarity between
approval sets A and B is 1 - |A intersection B| / |A union B|.  Two empty
ballots have dissimilarity zero.  Elections with different candidate slates
are embedded separately because their approval sets do not share a universe.
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from map_ordinal_preferences import classical_mds
from sample_approval_rankings import ApprovalElection, parse_preflib_cat


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASETS = (
    ROOT / "00026_frenchapproval",
    ROOT / "00073_frenchapproval",
)


@dataclass(frozen=True)
class SampledApprovalBallot:
    source_file: str
    source_title: str
    original_voters: int
    approved: tuple[int, ...]


def sample_approval_ballots(
    election: ApprovalElection, maximum: int, rng: random.Random
) -> list[SampledApprovalBallot]:
    """Sample voters without replacement, respecting ballot multiplicities."""
    if maximum <= 0:
        raise ValueError("maximum must be positive")
    expanded = [
        ballot.approved
        for ballot in election.ballots
        for _ in range(ballot.count)
    ]
    if len(expanded) > maximum:
        expanded = rng.sample(expanded, maximum)
    return [
        SampledApprovalBallot(
            election.path.name,
            election.title,
            election.num_voters,
            tuple(approved),
        )
        for approved in expanded
    ]


def jaccard_distances(ballots: np.ndarray) -> np.ndarray:
    """Return the all-pairs Jaccard dissimilarity matrix for binary rows."""
    ballots = np.asarray(ballots)
    if ballots.ndim != 2 or ballots.shape[0] == 0 or ballots.shape[1] == 0:
        raise ValueError("ballots must be a non-empty two-dimensional matrix")
    if not np.all((ballots == 0) | (ballots == 1)):
        raise ValueError("Jaccard approval distance requires binary ballots")
    numeric = ballots.astype(np.int64, copy=False)
    intersections = numeric @ numeric.T
    sizes = numeric.sum(axis=1)
    unions = sizes[:, None] + sizes[None, :] - intersections
    similarities = np.ones_like(unions, dtype=float)
    np.divide(intersections, unions, out=similarities, where=unions != 0)
    distances = 1.0 - similarities
    np.fill_diagonal(distances, 0.0)
    return distances


def normalized_stress(distances: np.ndarray, coordinates: np.ndarray) -> float:
    """Return Kruskal's normalized raw stress for the 2-D representation."""
    differences = coordinates[:, None, :] - coordinates[None, :, :]
    embedded = np.linalg.norm(differences, axis=2)
    upper = np.triu_indices(len(distances), k=1)
    denominator = float(np.square(distances[upper]).sum())
    if denominator == 0.0:
        return 0.0
    return float(
        np.sqrt(np.square(distances[upper] - embedded[upper]).sum() / denominator)
    )


def metric_mds(
    distances: np.ndarray,
    max_iterations: int = 150,
    tolerance: float = 1e-6,
) -> tuple[np.ndarray, int]:
    """Fit two-dimensional metric MDS with deterministic SMACOF updates."""
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    coordinates = classical_mds(distances)
    num_points = len(distances)
    previous_stress = np.inf
    for iteration in range(1, max_iterations + 1):
        differences = coordinates[:, None, :] - coordinates[None, :, :]
        embedded = np.linalg.norm(differences, axis=2)
        ratios = np.zeros_like(distances)
        np.divide(distances, embedded, out=ratios, where=embedded > 1e-12)
        b_matrix = -ratios
        b_matrix[np.diag_indices(num_points)] = ratios.sum(axis=1)
        updated = b_matrix @ coordinates / num_points
        updated -= updated.mean(axis=0)

        updated_differences = updated[:, None, :] - updated[None, :, :]
        updated_distances = np.linalg.norm(updated_differences, axis=2)
        stress = float(np.square(distances - updated_distances).sum() / 2.0)
        coordinates = updated
        if np.isfinite(previous_stress):
            improvement = previous_stress - stress
            if improvement >= 0.0 and improvement <= tolerance * previous_stress:
                return coordinates, iteration
        previous_stress = stress
    return coordinates, max_iterations


def _ballot_matrix(
    ballots: Sequence[SampledApprovalBallot], num_candidates: int
) -> np.ndarray:
    matrix = np.zeros((len(ballots), num_candidates), dtype=np.int8)
    for row, ballot in enumerate(ballots):
        matrix[row, list(ballot.approved)] = 1
    return matrix


def write_coordinates(
    output_path: Path,
    ballots: Sequence[SampledApprovalBallot],
    candidate_names: Sequence[str],
    coordinates: np.ndarray,
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            (
                "source_file",
                "source_title",
                "original_voters",
                "approved_count",
                "approved_candidate_ids",
                "approved_candidates",
                "x",
                "y",
            )
        )
        for ballot, (x, y) in zip(ballots, coordinates):
            writer.writerow(
                (
                    ballot.source_file,
                    ballot.source_title,
                    ballot.original_voters,
                    len(ballot.approved),
                    " | ".join(str(candidate + 1) for candidate in ballot.approved),
                    " | ".join(candidate_names[candidate] for candidate in ballot.approved),
                    x,
                    y,
                )
            )


def plot_embedding(
    output_path: Path,
    dataset_name: str,
    ballots: Sequence[SampledApprovalBallot],
    coordinates: np.ndarray,
    stress: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10.5, 7.8))
    titles = list(dict.fromkeys(ballot.source_title for ballot in ballots))
    palette = plt.get_cmap("tab10")
    for group_index, title in enumerate(titles):
        indices = np.asarray(
            [index for index, ballot in enumerate(ballots) if ballot.source_title == title]
        )
        original_voters = ballots[int(indices[0])].original_voters
        axis.scatter(
            coordinates[indices, 0],
            coordinates[indices, 1],
            s=28,
            alpha=0.58,
            color=palette(group_index % 10),
            edgecolors="none",
            label=f"{title} (sample {len(indices)} of {original_voters})",
        )
    axis.set_title(
        f"{dataset_name}: approval ballots by Jaccard-distance MDS\n"
        f"normalized stress = {stress:.3f}"
    )
    axis.set_xlabel("MDS dimension 1")
    axis.set_ylabel("MDS dimension 2")
    axis.grid(alpha=0.18)
    axis.legend(loc="best", fontsize=8, frameon=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=230, bbox_inches="tight")
    plt.close(figure)


def map_dataset(
    dataset: Path,
    output_dir: Path,
    max_voters_per_election: int = 250,
    seed: int = 23,
    max_mds_iterations: int = 150,
) -> tuple[Path, Path, float]:
    paths = sorted(Path(dataset).glob("*.cat"))
    if not paths:
        raise FileNotFoundError(f"no .cat files found in {dataset}")
    elections = [parse_preflib_cat(path) for path in paths]
    candidate_names = elections[0].candidate_names
    if any(election.candidate_names != candidate_names for election in elections[1:]):
        raise ValueError("elections in one MDS map must share a candidate slate")

    ballots: list[SampledApprovalBallot] = []
    for election_index, election in enumerate(elections):
        ballots.extend(
            sample_approval_ballots(
                election,
                max_voters_per_election,
                random.Random(seed + election_index),
            )
        )
    matrix = _ballot_matrix(ballots, len(candidate_names))
    print(
        f"{Path(dataset).name}: computing Jaccard distances for "
        f"{len(ballots)} sampled voters ..."
    )
    distances = jaccard_distances(matrix)
    coordinates, iterations = metric_mds(distances, max_mds_iterations)
    stress = normalized_stress(distances, coordinates)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(dataset).name}_jaccard_mds"
    image_path = output_dir / f"{stem}.png"
    coordinate_path = output_dir / f"{stem}.csv"
    write_coordinates(coordinate_path, ballots, candidate_names, coordinates)
    plot_embedding(image_path, Path(dataset).name, ballots, coordinates, stress)
    print(
        f"  SMACOF iterations={iterations}; stress={stress:.4f}; "
        f"wrote {image_path} and {coordinate_path}"
    )
    return image_path, coordinate_path, stress


def run_pipeline(
    datasets: Sequence[Path] = DEFAULT_DATASETS,
    output_dir: Path = ROOT / "preference_map_outputs" / "jaccard_mds",
    max_voters_per_election: int = 250,
    seed: int = 23,
    max_mds_iterations: int = 150,
) -> list[tuple[Path, Path, float]]:
    results: list[tuple[Path, Path, float]] = []
    for dataset_index, dataset in enumerate(datasets):
        results.append(
            map_dataset(
                dataset,
                output_dir,
                max_voters_per_election,
                seed + 10_000 * dataset_index,
                max_mds_iterations,
            )
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Map French approval ballots with Jaccard dissimilarity and classical MDS."
    )
    parser.add_argument("--datasets", nargs="+", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "preference_map_outputs" / "jaccard_mds",
    )
    parser.add_argument(
        "--max-voters-per-election",
        type=int,
        default=250,
        help="Maximum voters sampled without replacement from each election",
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--max-mds-iterations", type=int, default=150)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_pipeline(
        args.datasets,
        args.output_dir,
        args.max_voters_per_election,
        args.seed,
        args.max_mds_iterations,
    )


if __name__ == "__main__":
    main()
