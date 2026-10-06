"""Cluster approval voters using candidate disagreement, then score candidates.

Requires NumPy and SciPy. Input: headered binary CSV, PrefLib .cat, or a
directory of .cat files (combined using preflib_to_matrix.combine_elections).
All rows are individual voters; multiplicities in .cat input are expanded.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform

from preflib_to_matrix import combine_elections, load_directory, parse_preflib_cat


def validate_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError("Expected at least one voter and one candidate")
    if not np.all((matrix == 0) | (matrix == 1)):
        raise ValueError("Approval entries must be 0 or 1")
    return matrix.astype(np.int64)


def candidate_distances(matrix: np.ndarray) -> np.ndarray:
    """Jaccard distances between candidate approver sets; empty/empty = 0."""
    matrix = validate_matrix(matrix)
    intersection = matrix.T @ matrix
    approvals = matrix.sum(axis=0)
    union = approvals[:, None] + approvals[None, :] - intersection
    return np.divide(union - intersection, union,
                     out=np.zeros(union.shape, dtype=float), where=union != 0)


def voter_distances(matrix: np.ndarray, distance: str = "disagreement") -> np.ndarray:
    """D(x,y) = max J(d,e) over candidates on which x,y disagree.

    distance='disagreement' uses the original D definition. 'hamming' uses
    the fraction of differing approval bits; 'jaccard' compares voter approval
    sets directly, with empty/empty = 0. All three have range [0,1].
    For disagreement distance, empty and singleton disagreement sets have distance zero. This need not
    be a metric. Duplicate ballots share computation, but remain separate
    voters in clustering, preserving voter weights and linkage tie behavior.
    The returned dense matrix requires O(n_voters**2) memory.
    """
    matrix = validate_matrix(matrix)
    if distance in ("hamming", "jaccard"):
        result = squareform(pdist(matrix.astype(bool), metric=distance))
        # Explicitly support empty/empty approval sets across SciPy versions.
        if distance == "jaccard":
            empty = ~matrix.any(axis=1)
            result[np.ix_(empty, empty)] = 0.0
        return result
    if distance != "disagreement":
        raise ValueError(f"Unknown voter distance: {distance}")
    jaccard = candidate_distances(matrix)
    ballots, inverse = np.unique(matrix, axis=0, return_inverse=True)
    distances = np.zeros((len(ballots), len(ballots)))
    cache: dict[bytes, float] = {}
    for i in range(len(ballots)):
        for j in range(i + 1, len(ballots)):
            disagreement = ballots[i] != ballots[j]
            key = np.packbits(disagreement).tobytes()
            if key not in cache:
                selected = np.flatnonzero(disagreement)
                cache[key] = (float(jaccard[np.ix_(selected, selected)].max())
                              if len(selected) > 1 else 0.0)
            distances[i, j] = distances[j, i] = cache[key]
    return distances[inverse[:, None], inverse[None, :]]


def correlation_objective(distances: np.ndarray, labels: np.ndarray) -> float:
    """Sum D for within-group pairs and 1-D for between-group pairs."""
    i, j = np.triu_indices(len(labels), 1)
    return float(np.where(labels[i] == labels[j], distances[i, j],
                          1 - distances[i, j]).sum())


def cluster_voters(distances: np.ndarray, algorithm: str = "complete",
                   cutoff: float = 0.5) -> np.ndarray:
    """Return zero-based group labels without specifying a group count.

    complete/average: merge at heights <= cutoff.
    correlation: greedy agglomeration from singleton voters, always taking
    the merge with greatest objective reduction until no improving merge
    remains. This is a heuristic, not an exact optimizer. Cutoff is unused.
    """
    distances = np.asarray(distances, dtype=float)
    if (distances.ndim != 2 or distances.shape[0] != distances.shape[1]
            or not len(distances) or not np.all(np.isfinite(distances))
            or np.any((distances < 0) | (distances > 1))
            or not np.allclose(distances, distances.T)
            or np.any(np.diag(distances) != 0)):
        raise ValueError("Distances must be symmetric, finite, in [0,1], with zero diagonal")
    if algorithm not in ("complete", "average", "correlation"):
        raise ValueError(f"Unknown clustering algorithm: {algorithm}")
    if not np.isfinite(cutoff) or not 0 <= cutoff <= 1:
        raise ValueError("Cutoff must be in [0,1]")
    if len(distances) == 1:
        return np.zeros(1, dtype=int)
    if algorithm != "correlation":
        tree = linkage(squareform(distances, checks=False), method=algorithm)
        return fcluster(tree, t=cutoff, criterion="distance") - 1

    # Merging groups changes each cross-pair cost from 1-D to D.
    # Group merge deltas add, so no repeated summation over voter pairs.
    delta = 2 * distances - 1
    np.fill_diagonal(delta, np.inf)
    groups = {i: [i] for i in range(len(distances))}
    while len(groups) > 1:
        i, j = np.unravel_index(np.argmin(delta), delta.shape)
        if delta[i, j] >= -1e-12:
            break
        others = [g for g in groups if g != i and g != j]
        merged = delta[i, others] + delta[j, others]
        delta[i, others] = delta[others, i] = merged
        delta[j, :] = delta[:, j] = np.inf
        delta[i, i] = np.inf
        groups[i].extend(groups.pop(j))
    labels = np.empty(len(distances), dtype=int)
    for label, members in enumerate(groups.values()):
        labels[members] = label
    return labels


def group_approvals(matrix: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return group sizes and group-by-candidate approval fractions."""
    matrix = validate_matrix(matrix)
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != len(matrix):
        raise ValueError("Provide exactly one group label per voter")
    _, inverse, sizes = np.unique(labels, return_inverse=True, return_counts=True)
    counts = np.zeros((len(sizes), matrix.shape[1]), dtype=np.int64)
    np.add.at(counts, inverse, matrix)
    return sizes, counts / sizes[:, None]


def score_candidates(matrix: np.ndarray, labels: np.ndarray,
                     scoring: str = "pairwise") -> np.ndarray:
    """Score with pairwise minimum, weighted Nash, or egalitarian welfare.

    Let p_ic be approval fraction in group i and w_i = |P_i|/n.
    pairwise: sum_{i<j} |P_i||P_j| min(p_ic, p_jc).
    nash: product_i p_ic**w_i (weighted geometric mean, no smoothing).
    egalitarian: min_i p_ic, the worst per-voter group approval rate.
    Egalitarian uses no extra size multiplier: fractions already normalize
    group populations. Nash is zero if any group has zero approval.
    """
    sizes, portions = group_approvals(matrix, labels)
    if scoring == "pairwise":
        scores = np.zeros(portions.shape[1])
        for i in range(len(sizes)):
            scores += (sizes[i] * sizes[i + 1:, None]
                       * np.minimum(portions[i], portions[i + 1:])).sum(axis=0)
        return scores
    if scoring == "nash":
        scores = np.zeros(portions.shape[1])
        positive = np.all(portions > 0, axis=0)
        scores[positive] = np.exp((sizes / sizes.sum()) @ np.log(portions[:, positive]))
        return scores
    if scoring == "egalitarian":
        return portions.min(axis=0)
    raise ValueError(f"Unknown scoring function: {scoring}")


def load_input(path: Path) -> tuple[list[str], np.ndarray]:
    if path.is_dir():
        election = combine_elections(load_directory(path))
    elif path.suffix.lower() == ".cat":
        election = parse_preflib_cat(path)
    else:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.reader(stream)
            names = next(reader, [])
            rows = [row for row in reader if row]
        if not names or not rows or any(len(row) != len(names) for row in rows):
            raise ValueError("Expected a headered CSV with equally sized, nonempty voter rows")
        return names, validate_matrix(np.asarray(rows, dtype=float))
    return election.candidate_names, validate_matrix(election.matrix)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSV, .cat file, or directory of .cat files")
    parser.add_argument("--clustering", choices=("complete", "average", "correlation"), default="complete")
    parser.add_argument("--distance", choices=("disagreement", "hamming", "jaccard"), default="disagreement",
                        help="Voter distance; Hamming is the fraction of differing bits (default: disagreement)")
    parser.add_argument("--cutoff", type=float, default=0.5, help="Hierarchical merge cutoff, inclusive (default: 0.5)")
    parser.add_argument("--scoring", choices=("pairwise", "nash", "egalitarian"), default="pairwise")
    parser.add_argument("--output", type=Path, help="Write candidate scores as CSV")
    parser.add_argument("--groups-output", type=Path, help="Write zero-based voter and group IDs as CSV")
    args = parser.parse_args()
    names, matrix = load_input(args.input)
    distances = voter_distances(matrix, args.distance)
    labels = cluster_voters(distances, args.clustering, args.cutoff)
    scores = score_candidates(matrix, labels, args.scoring)
    sizes, _ = group_approvals(matrix, labels)
    print(f"{len(matrix)} voters, {len(names)} candidates, {len(sizes)} groups")
    print(f"Voter distance: {args.distance}")
    print("Group sizes:", ", ".join(map(str, sizes)))
    print(f"Correlation objective: {correlation_objective(distances, labels):.10g}")
    if args.clustering == "correlation":
        print("Clustering uses greedy merges; global optimality is not guaranteed.")
    rows = [(int(c), names[c], float(scores[c])) for c in np.argsort(-scores, kind="stable")]
    for _, name, score in rows:
        print(f"{name}: {score:.10g}")
    for path, header, output_rows in (
        (args.output, ("candidate_id", "candidate", "score"), rows),
        (args.groups_output, ("voter_id", "group_id"), enumerate(labels)),
    ):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(header)
                writer.writerows(output_rows)


if __name__ == "__main__":
    main()
