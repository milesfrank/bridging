"""Construct the nested group families described by augmented Greedy Capture.

For every level k, the returned family G_k contains at most k groups, and every
group has size q_k = ceil(n / k).  Group members are zero-based row numbers in
the input CSV.  Hamming distance is the default because the matrices in this
repository are approval ballots; Euclidean distance is also supported.

When several choices are equally good, this implementation breaks ties by the
input row number (and then by the previous group's order).  This makes repeated
runs reproducible without changing any of the guarantees.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "matrices" / "frenchapproval.csv"
DEFAULT_OUTPUT = ROOT / "matrices" / "nested_greedy_capture_groups.json"


@dataclass(frozen=True)
class Capture:
    """One quota-sized group captured by a ball."""

    center: int
    radius: float
    members: tuple[int, ...]


@dataclass(frozen=True)
class Level:
    """The group family and approximation factor at one value of k."""

    k: int
    quota: int
    alpha: int
    groups: tuple[tuple[int, ...], ...]
    copied_from_next_level: bool
    captures: tuple[Capture, ...] = ()
    source_groups: tuple[int, ...] = ()


def load_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    """Load a nonempty numeric matrix with a CSV header."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError(f"{path} is empty") from error
        rows = [row for row in reader if row]

    if not header:
        raise ValueError(f"{path} has no feature columns")
    if not rows:
        raise ValueError(f"{path} has no points")
    if any(len(row) != len(header) for row in rows):
        raise ValueError("every row must have one value per header column")
    try:
        points = np.asarray(rows, dtype=float)
    except ValueError as error:
        raise ValueError(f"{path} contains a nonnumeric value") from error
    if not np.all(np.isfinite(points)):
        raise ValueError("all matrix values must be finite")
    return header, points


def pairwise_distances(points: np.ndarray, metric: str) -> np.ndarray:
    """Return the symmetric point-to-point distance matrix."""
    if points.ndim != 2 or not len(points):
        raise ValueError("points must be a nonempty two-dimensional matrix")
    if metric == "hamming":
        return np.count_nonzero(
            points[:, None, :] != points[None, :, :], axis=2
        ).astype(float)
    if metric == "euclidean":
        squared_norms = np.einsum("ij,ij->i", points, points)
        squared = squared_norms[:, None] + squared_norms[None, :] - 2 * points @ points.T
        return np.sqrt(np.maximum(squared, 0.0))
    raise ValueError(f"unknown metric: {metric}")


def augmented_greedy_capture(
    distances: np.ndarray,
    quota: int,
    *,
    _integer_distances: np.ndarray | None = None,
    _initial_shell_counts: np.ndarray | None = None,
) -> tuple[Capture, ...]:
    """Capture floor(n/quota) disjoint quota-sized groups with growing balls.

    At each iteration, choose the ball whose ``quota``-th closest uncaptured
    point has the smallest distance.  A ball may be centered at any input point,
    including one captured during an earlier iteration.  Equal distances are
    resolved by center row and member row.
    """
    distances = np.asarray(distances, dtype=float)
    n = len(distances)
    if distances.shape != (n, n):
        raise ValueError("distances must be a square matrix")
    if quota < 1 or quota > n:
        raise ValueError("quota must be between 1 and the population size")
    if np.any(distances < 0) or not np.all(np.isfinite(distances)):
        raise ValueError("distances must be finite and nonnegative")

    remaining = np.ones(n, dtype=bool)
    captures: list[Capture] = []

    # Hamming and other small integer metrics admit a much faster equivalent
    # implementation.  counts[c, r] is the number of uncaptured points in the
    # radius-r shell around c.  Removing all captured points updates all balls
    # in O(n * quota), so a complete capture run takes O(n^2) bucket updates.
    shell_counts: np.ndarray | None = None
    integer_distances: np.ndarray | None = None
    if _integer_distances is not None and _initial_shell_counts is not None:
        integer_distances = _integer_distances
        shell_counts = _initial_shell_counts.copy()
    else:
        rounded = np.rint(distances)
        integer_metric = np.allclose(distances, rounded, rtol=0.0, atol=1e-12)
        max_distance = int(rounded.max()) if integer_metric else -1
        if integer_metric and max_distance <= 10_000:
            integer_distances = rounded.astype(np.int32)
            shell_counts = np.zeros((n, max_distance + 1), dtype=np.int32)
            rows = np.repeat(np.arange(n), n)
            np.add.at(shell_counts, (rows, integer_distances.ravel()), 1)

    for _ in range(n // quota):
        available = np.flatnonzero(remaining)
        if shell_counts is not None:
            cumulative = np.cumsum(shell_counts, axis=1)
            radii = np.argmax(cumulative >= quota, axis=1)
            center = int(np.argmin(radii))
        else:
            # Generic fallback for continuous metrics.  np.partition performs
            # the expensive search in compiled code rather than a Python loop.
            eligible = np.where(remaining[None, :], distances, np.inf)
            radii = np.partition(eligible, quota - 1, axis=1)[:, quota - 1]
            center = int(np.argmin(radii))

        # The row-index key selects a canonical subset when the ball boundary
        # contains more points than are needed.
        order = np.lexsort((available, distances[center, available]))
        members = available[order[:quota]]
        captures.append(
            Capture(center, float(distances[center, members[-1]]), tuple(int(i) for i in members))
        )
        remaining[members] = False
        if shell_counts is not None and integer_distances is not None:
            row_indices = np.tile(np.arange(n), len(members))
            member_indices = np.repeat(members, n)
            shells = integer_distances[row_indices, member_indices]
            np.add.at(shell_counts, (row_indices, shells), -1)
    return tuple(captures)


def _nearest_previous_groups(
    centers: Sequence[int],
    previous_groups: Sequence[tuple[int, ...]],
    distances: np.ndarray,
) -> np.ndarray:
    """Return argmin_g max_{u in g} d(center, u) for every center."""
    center_rows = np.asarray(centers, dtype=np.int64)
    group_rows = np.asarray(previous_groups, dtype=np.int64)
    radii = distances[center_rows[:, None, None], group_rows[None, :, :]].max(axis=2)
    return np.argmin(radii, axis=1)


def construct_levels(distances: np.ndarray) -> list[Level]:
    """Construct G_n, G_{n-1}, ..., G_1 from a distance matrix."""
    distances = np.asarray(distances, dtype=float)
    n = len(distances)
    if distances.shape != (n, n) or n == 0:
        raise ValueError("distances must be a nonempty square matrix")

    groups: tuple[tuple[int, ...], ...] = tuple((i,) for i in range(n))
    levels = [Level(n, 1, 1, groups, copied_from_next_level=False)]
    previous_quota = 1
    alpha = 1

    # This histogram depends only on the metric, not k.  Reusing it avoids
    # rebuilding n^2 distance buckets at every quota change.
    rounded = np.rint(distances)
    integer_distances: np.ndarray | None = None
    initial_shell_counts: np.ndarray | None = None
    if np.allclose(distances, rounded, rtol=0.0, atol=1e-12):
        max_distance = int(rounded.max())
        if max_distance <= 10_000:
            integer_distances = rounded.astype(np.int32)
            initial_shell_counts = np.zeros((n, max_distance + 1), dtype=np.int32)
            rows = np.repeat(np.arange(n), n)
            np.add.at(initial_shell_counts, (rows, integer_distances.ravel()), 1)

    for k in range(n - 1, 0, -1):
        quota = math.ceil(n / k)
        if quota == previous_quota:
            levels.append(Level(k, quota, alpha, groups, copied_from_next_level=True))
            continue

        captures = augmented_greedy_capture(
            distances,
            quota,
            _integer_distances=integer_distances,
            _initial_shell_counts=initial_shell_counts,
        )
        new_groups: list[tuple[int, ...]] = []
        sources = _nearest_previous_groups(
            [capture.center for capture in captures], groups, distances
        )
        for capture, source_value in zip(captures, sources):
            source = int(source_value)
            old_group = groups[source]
            old_members = set(old_group)
            additions = [i for i in capture.members if i not in old_members]
            needed = quota - previous_quota
            if len(additions) < needed:
                raise RuntimeError("capture does not contain enough new points to enlarge group")
            new_group = old_group + tuple(additions[:needed])
            if len(set(new_group)) != quota:
                raise RuntimeError("constructed group does not have the required size")
            new_groups.append(new_group)

        groups = tuple(new_groups)
        alpha += 2
        levels.append(
            Level(
                k, quota, alpha, groups, copied_from_next_level=False,
                captures=captures, source_groups=tuple(int(source) for source in sources),
            )
        )
        previous_quota = quota

    return levels


def validate_levels(levels: Sequence[Level], population_size: int) -> None:
    """Check the cardinality invariants claimed by the construction."""
    if [level.k for level in levels] != list(range(population_size, 0, -1)):
        raise ValueError("levels must run from n down to 1")
    for level in levels:
        expected_quota = math.ceil(population_size / level.k)
        if level.quota != expected_quota:
            raise ValueError(f"G_{level.k} has the wrong quota")
        if len(level.groups) > level.k:
            raise ValueError(f"G_{level.k} has more than k groups")
        for group in level.groups:
            if len(group) != expected_quota or len(set(group)) != expected_quota:
                raise ValueError(f"G_{level.k} contains a group of the wrong size")
            if any(member < 0 or member >= population_size for member in group):
                raise ValueError(f"G_{level.k} contains an invalid point index")


def write_json(
    path: Path, input_path: Path, metric: str, feature_names: Sequence[str], levels: Sequence[Level]
) -> None:
    """Write every level, including construction witnesses, as JSON."""
    document = {
        "input": str(input_path),
        "metric": metric,
        "population_size": len(levels),
        "features": list(feature_names),
        "member_indexing": "zero-based CSV data-row index",
        # A copied level refers to the following k rather than repeating what
        # can be millions of identical member indices in a large hierarchy.
        "levels": [
            {
                "k": level.k,
                "quota": level.quota,
                "alpha": level.alpha,
                "copied_from_next_level": level.copied_from_next_level,
                **(
                    {"same_groups_as_k": level.k + 1}
                    if level.copied_from_next_level
                    else {"groups": [list(group) for group in level.groups]}
                ),
                **(
                    {
                        "captures": [
                            {
                                "center": capture.center,
                                "radius": capture.radius,
                                "members": list(capture.members),
                                "source_group": source,
                            }
                            for capture, source in zip(level.captures, level.source_groups)
                        ]
                    }
                    if level.captures else {}
                ),
            }
            for level in levels
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build all G_k group families using augmented Greedy Capture."
    )
    parser.add_argument("csv", type=Path, nargs="?", default=DEFAULT_CSV)
    parser.add_argument("--metric", choices=("hamming", "euclidean"), default="hamming")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_names, points = load_matrix(args.csv)
    distances = pairwise_distances(points, args.metric)
    levels = construct_levels(distances)
    validate_levels(levels, len(points))
    write_json(args.output, args.csv, args.metric, feature_names, levels)

    rebuilt = sum(not level.copied_from_next_level for level in levels)
    print(f"constructed {len(levels)} levels for {len(points)} points ({rebuilt} distinct quotas)")
    print(f"G_1: {len(levels[-1].groups)} group of size {levels[-1].quota}")
    print(f"final alpha_1: {levels[-1].alpha}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
