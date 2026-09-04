"""Find the minimum gamma satisfying DC-mPJR+ under Hamming distance.

This is a domain-neutral implementation for generic centroid-clustering
instances. It reads separate matrices for the agents N, candidate centers M,
and selected centers X. Each CSV must have a header row, and all three headers
must describe the same features in the same order.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MinimumGammaResult:
    gamma: Fraction | None
    population_size: int
    candidate_center_count: int
    selected_center_count: int
    witness_center: int | None = None
    witness_radius: int | None = None
    witness_coalition_size: int | None = None
    witness_deserved_centers: int | None = None
    witness_required_distance: int | None = None

    @property
    def finite(self) -> bool:
        return self.gamma is not None


def load_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    """Load a numeric, headered CSV matrix."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError(f"{path} is empty") from error
        if not header:
            raise ValueError(f"{path} has an empty header")
        try:
            rows = [[float(value) for value in row] for row in reader]
        except ValueError as error:
            raise ValueError(f"{path} contains a nonnumeric value") from error

    if not rows:
        raise ValueError(f"{path} contains no matrix rows")
    if any(len(row) != len(header) for row in rows):
        raise ValueError(f"{path} has a row whose width differs from its header")
    return header, np.asarray(rows)


def hamming_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return all pairwise counts of unequal coordinates."""
    return np.count_nonzero(left[:, None, :] != right[None, :, :], axis=2)


def selected_indices_in_candidates(
    candidate_centers: np.ndarray, selected_centers: np.ndarray
) -> np.ndarray:
    """Map each row of X to its unique, identical row in M."""
    indices: list[int] = []
    for selected_row_number, selected_center in enumerate(selected_centers):
        matches = np.flatnonzero(
            np.all(candidate_centers == selected_center, axis=1)
        )
        if matches.size == 0:
            raise ValueError(
                f"selected-center row {selected_row_number} does not occur in M"
            )
        if matches.size > 1:
            raise ValueError(
                f"selected-center row {selected_row_number} has multiple matches "
                "in M; candidate-center rows must be unique"
            )
        indices.append(int(matches[0]))

    result = np.asarray(indices, dtype=np.int64)
    if np.unique(result).size != result.size:
        raise ValueError("X contains duplicate selected centers")
    return result


def _validate_matrices(
    agents: np.ndarray,
    candidate_centers: np.ndarray,
    selected_centers: np.ndarray,
) -> None:
    if any(matrix.ndim != 2 for matrix in (agents, candidate_centers, selected_centers)):
        raise ValueError("N, M, and X must be two-dimensional matrices")
    widths = {agents.shape[1], candidate_centers.shape[1], selected_centers.shape[1]}
    if len(widths) != 1:
        raise ValueError("N, M, and X must have the same feature count")
    n = agents.shape[0]
    m = candidate_centers.shape[0]
    k = selected_centers.shape[0]
    if n == 0 or m == 0 or k == 0:
        raise ValueError("N, M, and X must be non-empty")
    if k > n:
        raise ValueError("|X| cannot exceed |N|")
    if k > m:
        raise ValueError("|X| cannot exceed |M|")


def _validate_selected_indices(
    candidate_centers: np.ndarray,
    selected_indices: np.ndarray,
) -> None:
    if selected_indices.ndim != 1 or selected_indices.size == 0:
        raise ValueError("selected_indices must be a non-empty vector")
    if np.any(selected_indices < 0) or np.any(
        selected_indices >= candidate_centers.shape[0]
    ):
        raise ValueError("selected center index outside M")
    if np.unique(selected_indices).size != selected_indices.size:
        raise ValueError("selected center indices must be unique")


def minimum_gamma_dc_mpjr(
    agents: np.ndarray,
    candidate_centers: np.ndarray,
    selected_centers: np.ndarray,
) -> MinimumGammaResult:
    """Return the exact minimum gamma for DC-mPJR+ with Hamming distance.

    The result's witness is the default-coalition prefix imposing the largest
    lower bound on gamma. ``gamma=None`` means that no finite gamma suffices,
    which can occur when a radius-zero coalition has inadequate coverage.
    All witness row indices are zero-based.
    """
    agents = np.asarray(agents)
    candidate_centers = np.asarray(candidate_centers)
    selected_centers = np.asarray(selected_centers)
    _validate_matrices(agents, candidate_centers, selected_centers)
    selected_indices = selected_indices_in_candidates(
        candidate_centers, selected_centers
    )

    return minimum_gamma_dc_mpjr_indices(
        agents,
        candidate_centers,
        selected_indices,
    )


def minimum_gamma_dc_mpjr_indices(
    agents: np.ndarray,
    candidate_centers: np.ndarray,
    selected_indices: np.ndarray | list[int],
) -> MinimumGammaResult:
    """Return the exact minimum gamma when X is identified by rows of M.

    This indexed form supports candidate matrices containing duplicate rows,
    as can arise in the potential-approver construction. Center and witness
    indices are zero-based.
    """
    agents = np.asarray(agents)
    candidate_centers = np.asarray(candidate_centers)
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    if agents.ndim != 2 or candidate_centers.ndim != 2:
        raise ValueError("N and M must be two-dimensional matrices")
    if agents.shape[0] == 0 or candidate_centers.shape[0] == 0:
        raise ValueError("N and M must be non-empty")
    if agents.shape[1] != candidate_centers.shape[1]:
        raise ValueError("N and M must have the same feature count")
    if selected_indices.size > agents.shape[0]:
        raise ValueError("|X| cannot exceed |N|")
    _validate_selected_indices(candidate_centers, selected_indices)

    n = agents.shape[0]
    m = candidate_centers.shape[0]
    k = selected_indices.size
    selected_centers = candidate_centers[selected_indices]
    selected_mask = np.zeros(m, dtype=bool)
    selected_mask[selected_indices] = True
    agent_to_selected = hamming_distances(agents, selected_centers)

    minimum_gamma = Fraction(1, 1)
    witness: tuple[int, int, int, int, int] | None = None

    for center_index in np.flatnonzero(~selected_mask):
        distances_to_center = np.count_nonzero(
            agents != candidate_centers[center_index], axis=1
        )
        order = np.argsort(distances_to_center, kind="stable")
        ordered_distances = distances_to_center[order]
        delta = np.full(k, np.inf)

        batch_start = 0
        while batch_start < n:
            radius = int(ordered_distances[batch_start])
            batch_end = int(
                np.searchsorted(ordered_distances, radius, side="right")
            )
            new_agents = order[batch_start:batch_end]
            delta = np.minimum(
                delta,
                agent_to_selected[new_agents].min(axis=0),
            )

            coalition_size = batch_end
            deserved_centers = coalition_size * k // n
            if deserved_centers > 0:
                # Coverage reaches t exactly when gamma * radius reaches the
                # t-th-smallest delta(x). Hamming distances make this an exact
                # rational threshold with denominator equal to the radius.
                required_distance = int(
                    np.partition(delta, deserved_centers - 1)[
                        deserved_centers - 1
                    ]
                )
                current_witness = (
                    int(center_index),
                    radius,
                    coalition_size,
                    deserved_centers,
                    required_distance,
                )

                if radius == 0 and required_distance > 0:
                    return MinimumGammaResult(
                        gamma=None,
                        population_size=n,
                        candidate_center_count=m,
                        selected_center_count=k,
                        witness_center=current_witness[0],
                        witness_radius=current_witness[1],
                        witness_coalition_size=current_witness[2],
                        witness_deserved_centers=current_witness[3],
                        witness_required_distance=current_witness[4],
                    )

                if radius > 0:
                    required_gamma = Fraction(required_distance, radius)
                    if required_gamma > minimum_gamma:
                        minimum_gamma = required_gamma
                        witness = current_witness

            batch_start = batch_end

    return MinimumGammaResult(
        gamma=minimum_gamma,
        population_size=n,
        candidate_center_count=m,
        selected_center_count=k,
        witness_center=None if witness is None else witness[0],
        witness_radius=None if witness is None else witness[1],
        witness_coalition_size=None if witness is None else witness[2],
        witness_deserved_centers=None if witness is None else witness[3],
        witness_required_distance=None if witness is None else witness[4],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find the exact minimum gamma satisfying DC-mPJR+ under "
            "Hamming distance."
        )
    )
    parser.add_argument("agents", type=Path, help="Headered CSV for N")
    parser.add_argument("candidate_centers", type=Path, help="Headered CSV for M")
    parser.add_argument("selected_centers", type=Path, help="Headered CSV for X")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        agent_header, agents = load_matrix(args.agents)
        candidate_header, candidate_centers = load_matrix(args.candidate_centers)
        selected_header, selected_centers = load_matrix(args.selected_centers)
        if candidate_header != agent_header or selected_header != agent_header:
            raise ValueError("N, M, and X must have identical CSV headers")
        result = minimum_gamma_dc_mpjr(
            agents,
            candidate_centers,
            selected_centers,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    print("distance: Hamming")
    print(f"population size |N|: {result.population_size}")
    print(f"candidate centers |M|: {result.candidate_center_count}")
    print(f"selected centers |X|: {result.selected_center_count}")
    if result.finite:
        assert result.gamma is not None
        print(f"minimum gamma (exact): {result.gamma}")
        print(f"minimum gamma (decimal): {float(result.gamma):.12g}")
    else:
        print("minimum gamma: infinity")

    if result.witness_center is not None:
        print(f"binding center row in M (zero-based): {result.witness_center}")
        print(f"binding radius: {result.witness_radius}")
        print(f"binding coalition size: {result.witness_coalition_size}")
        print(f"deserved centers: {result.witness_deserved_centers}")
        print(f"required selected-center distance: {result.witness_required_distance}")


if __name__ == "__main__":
    main()
