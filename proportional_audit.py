"""Exact audit for proportional clustering.

This implements the rho-PF condition from Definition 3 of
"Proportionally Representative Clustering" (arXiv:2304.13917). Rows of
``points`` are individuals and columns are features. Candidate centers are
supplied as feature vectors.

For a solution X and a possible deviating center y, the relevant coalition is
formed by the voters with the largest values of D_i(X) / d(i, y), where
D_i(X) is the distance to the closest chosen center. The maximum entitlement-
sized ratio over y is the minimum proportionality factor rho.

"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AuditResult:
    rho: float
    worst_center: int
    coalition_size: int
    population_size: int
    exact: bool
    score: float


def _validate_inputs(
    points: np.ndarray, centers: np.ndarray, k: int, candidate_count: int
) -> None:
    if points.ndim != 2 or centers.ndim != 2:
        raise ValueError("points and centers must be two-dimensional arrays")
    if points.shape[1] != centers.shape[1]:
        raise ValueError("points and centers must have the same feature count")
    if points.shape[0] == 0 or centers.shape[0] == 0:
        raise ValueError("points and centers must be non-empty")
    if k < 1:
        raise ValueError("k must be positive")
    if k > centers.shape[0]:
        raise ValueError("k cannot exceed the number of chosen centers")
    if candidate_count < 1:
        raise ValueError("at least one candidate center is required")


def _audit_from_distances(
    distances_to_chosen: np.ndarray,
    distances_to_candidates: np.ndarray,
    k: int,
    candidate_offset: int = 0,
    population_size: int | None = None,
) -> AuditResult:
    n = distances_to_chosen.shape[0]
    coalition_size = int(np.ceil(n / k))
    best_rho = -np.inf
    worst_center = 0

    for candidate_index in range(distances_to_candidates.shape[1]):
        distance = distances_to_candidates[:, candidate_index]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.divide(
                distances_to_chosen,
                distance,
                out=np.zeros_like(distances_to_chosen, dtype=float),
                where=distance != 0,
            )
        ratios[(distance == 0) & (distances_to_chosen > 0)] = np.inf
        rho = float(np.partition(ratios, -coalition_size)[-coalition_size])
        if rho > best_rho:
            best_rho = rho
            worst_center = candidate_index + candidate_offset

    return AuditResult(
        rho=max(1.0, best_rho),
        worst_center=worst_center,
        coalition_size=coalition_size,
        population_size=population_size if population_size is not None else n,
        exact=True,
        score=1/max(1.0, best_rho) * k / n,
    )


def _squared_euclidean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    differences = left[:, None, :] - right[None, :, :]
    return np.einsum("ijk,ijk->ij", differences, differences)


def audit_exact(
    points: np.ndarray,
    chosen_centers: np.ndarray,
    candidate_centers: np.ndarray | None = None,
    k: int | None = None,
) -> AuditResult:
    """Return the exact minimum rho for a clustering solution.

    ``chosen_centers`` contains the feature vectors of the solution's centers.
    ``candidate_centers`` contains all centers that agents may deviate to. For
    the paper's common ``M = N`` case, pass ``points`` explicitly.
    """
    if k is None:
        k = chosen_centers.shape[0]
    if candidate_centers is None:
        candidate_centers = points
    _validate_inputs(points, chosen_centers, k, candidate_centers.shape[0])
    chosen_distances = _squared_euclidean(points, chosen_centers)
    candidate_distances = _squared_euclidean(points, candidate_centers)
    return _audit_from_distances(
        np.sqrt(chosen_distances.min(axis=1)),
        np.sqrt(candidate_distances),
        k,
        population_size=points.shape[0],
    )


def load_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        names = next(reader)
        points = np.asarray([[float(value) for value in row] for row in reader])
    return names, points


def resolve_candidate(names: list[str], query: str) -> int:
    folded = {name.casefold(): index for index, name in enumerate(names)}
    compact = {
        name.casefold().replace(" ", "").replace("-", ""): index
        for index, name in enumerate(names)
    }
    if query in names:
        return names.index(query)
    if query.casefold() in folded:
        return folded[query.casefold()]
    key = query.casefold().replace(" ", "").replace("-", "")
    if key in compact:
        return compact[key]
    listed = ", ".join(names)
    raise SystemExit(f"Unknown candidate {query!r}. Choose one of: {listed}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit proportionality of centers selected from a point matrix."
    )
    parser.add_argument("csv", type=Path)
    parser.add_argument(
        "--candidate", required=True,
        help="Candidate whose approvers become the chosen centers",
    )
    args = parser.parse_args()

    names, points = load_csv(args.csv)
    candidate_column = resolve_candidate(names, args.candidate)
    approving = points[:, candidate_column] == 1
    if not np.any(approving):
        raise SystemExit(f"No voters approved {names[candidate_column]}")

    chosen = points[approving]

    # M is the set of potential approvers: one counterfactual center per
    # voter, obtained by forcing the audited candidate's coordinate to 1.
    potential_approvers = points.copy()
    potential_approvers[:, candidate_column] = 1
    print(
        f"{names[candidate_column]} approvers as centers: "
        f"{int(approving.sum())} / {len(points)} voters"
    )
    result = audit_exact(
        points,
        chosen,
        candidate_centers=potential_approvers,
        k=len(chosen),
    )
    print(f"exact rho: {result.rho:.6g}")
    print(f"rho worst candidate row: {result.worst_center}")
    print(f"coalition entitlement: {result.coalition_size}")
    print(f"score: {result.score:.6g}")


if __name__ == "__main__":
    main()
