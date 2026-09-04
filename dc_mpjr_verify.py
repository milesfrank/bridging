"""Verify gamma-DC-mPJR+ and find its exact minimum gamma.

This implements Algorithm 2 from "Check, Please: Verifiably Fair
Clustering" (arXiv:2605.12317v2). The command-line interface uses the
potential-approver construction used by the other audit scripts in this
repository: M contains one copy of each voter with the audited alternative's
coordinate forced to 1, and X contains the copies belonging to actual
approvers.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from generic_dc_mpjr_min_gamma import (
    MinimumGammaResult,
    minimum_gamma_dc_mpjr_indices,
)
from proportional_audit import load_csv, resolve_candidate


@dataclass(frozen=True)
class VerificationResult:
    satisfies: bool
    gamma: float
    population_size: int
    candidate_center_count: int
    selected_center_count: int
    witness_center: int | None = None
    witness_radius: int | None = None
    witness_coalition_size: int | None = None
    witness_deserved_centers: int | None = None
    witness_covered_centers: int | None = None


def hamming_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return all pairwise Hamming distances between two row matrices."""
    return np.count_nonzero(left[:, None, :] != right[None, :, :], axis=2)


def _validate_inputs(
    agents: np.ndarray,
    candidate_centers: np.ndarray,
    selected_indices: np.ndarray,
    gamma: float,
) -> None:
    if agents.ndim != 2 or candidate_centers.ndim != 2:
        raise ValueError("agents and candidate_centers must be two-dimensional")
    if agents.shape[0] == 0 or candidate_centers.shape[0] == 0:
        raise ValueError("agents and candidate_centers must be non-empty")
    if agents.shape[1] != candidate_centers.shape[1]:
        raise ValueError("agents and candidate_centers must have equal width")
    if selected_indices.ndim != 1 or selected_indices.size == 0:
        raise ValueError("selected_indices must be a non-empty vector")
    if selected_indices.size > agents.shape[0]:
        raise ValueError("the number of selected centers cannot exceed |N|")
    if np.any(selected_indices < 0) or np.any(
        selected_indices >= candidate_centers.shape[0]
    ):
        raise ValueError("selected center index outside M")
    if np.unique(selected_indices).size != selected_indices.size:
        raise ValueError("selected center indices must be unique")
    if not np.isfinite(gamma) or gamma < 1:
        raise ValueError("gamma must be a finite number at least 1")


def verify_gamma_dc_mpjr(
    agents: np.ndarray,
    candidate_centers: np.ndarray,
    selected_indices: np.ndarray | list[int],
    gamma: float = 1.0,
) -> VerificationResult:
    """Verify gamma-DC-mPJR+ and return the first violation, if any.

    ``agents`` represents N, rows of ``candidate_centers`` represent M, and
    ``selected_indices`` identifies the rows of M that form X. Center and
    witness indices are zero-based. Distances count unequal coordinates.
    """
    agents = np.asarray(agents)
    candidate_centers = np.asarray(candidate_centers)
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    gamma = float(gamma)
    _validate_inputs(agents, candidate_centers, selected_indices, gamma)

    n = agents.shape[0]
    m = candidate_centers.shape[0]
    k = selected_indices.size
    selected_centers = candidate_centers[selected_indices]
    selected_mask = np.zeros(m, dtype=bool)
    selected_mask[selected_indices] = True

    # This n-by-k table is independent of the unselected center c. Its row
    # minima over the current prefix P are the paper's delta(x) values.
    agent_to_selected = hamming_distances(agents, selected_centers)

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

            # Add every agent at this radius at once. This makes P the closed
            # ball B(c, radius), including all boundary ties.
            delta = np.minimum(
                delta,
                agent_to_selected[new_agents].min(axis=0),
            )
            coalition_size = batch_end
            deserved_centers = coalition_size * k // n
            covered_centers = int(np.count_nonzero(delta <= gamma * radius))

            if covered_centers < deserved_centers:
                return VerificationResult(
                    satisfies=False,
                    gamma=gamma,
                    population_size=n,
                    candidate_center_count=m,
                    selected_center_count=k,
                    witness_center=int(center_index),
                    witness_radius=radius,
                    witness_coalition_size=coalition_size,
                    witness_deserved_centers=deserved_centers,
                    witness_covered_centers=covered_centers,
                )
            batch_start = batch_end

    return VerificationResult(
        satisfies=True,
        gamma=gamma,
        population_size=n,
        candidate_center_count=m,
        selected_center_count=k,
    )


def verify_candidate(
    names: list[str],
    points: np.ndarray,
    candidate: str,
    gamma: float = 1.0,
) -> tuple[str, VerificationResult]:
    """Verify a candidate under the potential-approver construction."""
    candidate_name, potential_approvers, selected_indices = (
        _potential_approver_instance(names, points, candidate)
    )
    result = verify_gamma_dc_mpjr(
        agents=points,
        candidate_centers=potential_approvers,
        selected_indices=selected_indices,
        gamma=gamma,
    )
    return candidate_name, result


def _potential_approver_instance(
    names: list[str],
    points: np.ndarray,
    candidate: str,
) -> tuple[str, np.ndarray, np.ndarray]:
    """Construct M and the row indices of X for one candidate."""
    candidate_column = resolve_candidate(names, candidate)
    if not np.all((points == 0) | (points == 1)):
        raise ValueError("the candidate verifier requires a binary matrix")

    approving = points[:, candidate_column] == 1
    if not np.any(approving):
        raise ValueError(f"No voters approved {names[candidate_column]}")

    potential_approvers = points.copy()
    potential_approvers[:, candidate_column] = 1
    selected_indices = np.flatnonzero(approving)
    return names[candidate_column], potential_approvers, selected_indices


def minimum_gamma_candidate(
    names: list[str],
    points: np.ndarray,
    candidate: str,
) -> tuple[str, MinimumGammaResult]:
    """Find the exact minimum gamma for a potential-approver instance."""
    candidate_name, potential_approvers, selected_indices = (
        _potential_approver_instance(names, points, candidate)
    )
    result = minimum_gamma_dc_mpjr_indices(
        agents=points,
        candidate_centers=potential_approvers,
        selected_indices=selected_indices,
    )
    return candidate_name, result


def audit_candidate(
    names: list[str],
    points: np.ndarray,
    candidate: str,
    gamma: float = 1.0,
) -> tuple[str, VerificationResult, MinimumGammaResult]:
    """Verify gamma and find the exact minimum in one candidate audit."""
    candidate_name, potential_approvers, selected_indices = (
        _potential_approver_instance(names, points, candidate)
    )
    verification = verify_gamma_dc_mpjr(
        agents=points,
        candidate_centers=potential_approvers,
        selected_indices=selected_indices,
        gamma=gamma,
    )
    minimum = minimum_gamma_dc_mpjr_indices(
        agents=points,
        candidate_centers=potential_approvers,
        selected_indices=selected_indices,
    )
    return candidate_name, verification, minimum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify gamma-DC-mPJR+ and find the exact minimum gamma using "
            "Hamming distance."
        )
    )
    parser.add_argument("csv", type=Path, help="Binary point-matrix CSV")
    parser.add_argument(
        "--candidate",
        required=True,
        help="Alternative used to construct M and X",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Approximation factor (at least 1; default: 1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names, points = load_csv(args.csv)
    try:
        candidate, result, minimum = audit_candidate(
            names,
            points,
            args.candidate,
            gamma=args.gamma,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    print(f"candidate: {candidate}")
    print(f"distance: Hamming")
    print(f"gamma: {result.gamma:g}")
    print(f"population size: {result.population_size}")
    print(f"candidate centers in M: {result.candidate_center_count}")
    print(f"selected centers in X: {result.selected_center_count}")
    if minimum.finite:
        assert minimum.gamma is not None
        print(f"minimum gamma (exact): {minimum.gamma}")
        print(f"minimum gamma (decimal): {float(minimum.gamma):.12g}")
    else:
        print("minimum gamma: infinity")
    print(f"satisfies gamma-DC-mPJR+: {result.satisfies}")
    if not result.satisfies:
        print(f"witness center row (zero-based): {result.witness_center}")
        print(f"witness radius: {result.witness_radius}")
        print(f"witness coalition size: {result.witness_coalition_size}")
        print(f"deserved centers: {result.witness_deserved_centers}")
        print(f"covered selected centers: {result.witness_covered_centers}")


if __name__ == "__main__":
    main()
