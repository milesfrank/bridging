"""Find every clear voter community under Hamming distance.

A proper subset S of the voters is clear when its smallest distance to a voter
outside S is strictly greater than its diameter.  Identical CSV rows still
represent distinct voters; they are collapsed only while doing the search and
are expanded back to one-based CSV data-row numbers in the output.

The output uses ``q_min=2`` and ``q_max=size`` to represent all integer quotas
for which a community qualifies.  Pass ``--expand-q`` to instead write one row
for every (q, community) pair.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from generic_dc_mpjr_min_gamma import hamming_distances, load_matrix


ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "matrices" / "frenchapproval.csv"
DEFAULT_OUTPUT = ROOT / "matrices" / "clear_communities.csv"
DEFAULT_PARTITION_OUTPUT = ROOT / "matrices" / "clear_community_partition.csv"


@dataclass(frozen=True)
class ClearCommunity:
    """One clear community and the values needed to verify it."""

    voter_rows: tuple[int, ...]
    distinct_ballot_types: int
    diameter: int
    min_external_distance: int

    @property
    def size(self) -> int:
        return len(self.voter_rows)


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.uint8)

    def find(self, item: int) -> int:
        parent = int(self.parent[item])
        while parent != item:
            grandparent = int(self.parent[parent])
            self.parent[item] = grandparent
            item, parent = parent, grandparent
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1

    def components(self) -> list[np.ndarray]:
        groups: dict[int, list[int]] = {}
        for item in range(len(self.parent)):
            groups.setdefault(self.find(item), []).append(item)
        return [np.asarray(group, dtype=np.int64) for group in groups.values()]


def find_clear_communities(voters: np.ndarray) -> list[ClearCommunity]:
    """Return all proper clear communities of at least two voters.

    The search is exhaustive: at radius r, a clear set of diameter r must be a
    connected component of the graph whose edges have distance at most r.  It
    is clear exactly when that component is also a clique (diameter at most r).
    """
    voters = np.asarray(voters)
    if voters.ndim != 2 or voters.shape[0] < 2 or voters.shape[1] == 0:
        raise ValueError("voters must have at least two rows and one column")

    ballot_types, inverse, counts = np.unique(
        voters, axis=0, return_inverse=True, return_counts=True
    )
    type_count = len(ballot_types)
    if type_count == 1:
        return []

    rows_by_type = [
        tuple(int(row) + 1 for row in np.flatnonzero(inverse == type_index))
        for type_index in range(type_count)
    ]
    distances = hamming_distances(ballot_types, ballot_types).astype(
        np.int32, copy=False
    )
    upper_left, upper_right = np.triu_indices(type_count, k=1)
    upper_distances = distances[upper_left, upper_right]
    disjoint_set = _DisjointSet(type_count)
    communities: list[ClearCommunity] = []

    # Radius zero handles communities consisting of multiple identical voters.
    # Subsequent radii add all edges at that exact integer Hamming distance.
    for radius in range(int(upper_distances.max()) + 1):
        if radius > 0:
            for edge in np.flatnonzero(upper_distances == radius):
                disjoint_set.union(
                    int(upper_left[edge]), int(upper_right[edge])
                )

        for component in disjoint_set.components():
            if len(component) == type_count:
                continue  # N has no outside voter, so the defining minimum is empty.
            population_size = int(counts[component].sum())
            if population_size < 2:
                continue

            diameter = int(distances[np.ix_(component, component)].max())
            if diameter != radius:
                continue

            outside_mask = np.ones(type_count, dtype=bool)
            outside_mask[component] = False
            min_external = int(
                distances[np.ix_(component, np.flatnonzero(outside_mask))].min()
            )
            # Components guarantee this, but retaining the explicit check makes
            # the strict inequality in the definition auditable in this code.
            if min_external <= diameter:
                continue

            voter_rows = tuple(
                sorted(row for type_index in component for row in rows_by_type[type_index])
            )
            communities.append(
                ClearCommunity(
                    voter_rows=voter_rows,
                    distinct_ballot_types=len(component),
                    diameter=diameter,
                    min_external_distance=min_external,
                )
            )

    communities.sort(key=lambda item: (item.diameter, -item.size, item.voter_rows))
    return communities


def find_partition(
    communities: list[ClearCommunity], population_size: int
) -> list[ClearCommunity] | None:
    """Return a clear-community partition of N, or ``None`` if none exists.

    Clear communities are laminar: any two that intersect are nested.  Their
    inclusion-maximal members are therefore pairwise disjoint.  A partition
    exists precisely when these maximal communities cover every voter.
    """
    if population_size < 1:
        raise ValueError("population_size must be positive")

    member_sets = [frozenset(item.voter_rows) for item in communities]
    expected_rows = frozenset(range(1, population_size + 1))
    if any(not members or not members <= expected_rows for members in member_sets):
        raise ValueError("a community contains a voter outside the population")

    maximal_indices = [
        index
        for index, members in enumerate(member_sets)
        if not any(
            members < other_members
            for other_index, other_members in enumerate(member_sets)
            if other_index != index
        )
    ]
    maximal = [communities[index] for index in maximal_indices]
    covered = frozenset(
        voter_row for community in maximal for voter_row in community.voter_rows
    )
    return maximal if covered == expected_rows else None


def write_communities(
    communities: list[ClearCommunity], output: Path, expand_q: bool = False
) -> None:
    """Write compact quota ranges, or one output row per qualifying quota."""
    output.parent.mkdir(parents=True, exist_ok=True)
    common_columns = [
        "community_id", "size", "distinct_ballot_types", "diameter_R",
        "min_external_distance", "separation_gap", "voter_rows",
    ]
    quota_columns = ["q"] if expand_q else ["q_min", "q_max"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=common_columns + quota_columns)
        writer.writeheader()
        for community_id, community in enumerate(communities, start=1):
            common = {
                "community_id": community_id,
                "size": community.size,
                "distinct_ballot_types": community.distinct_ballot_types,
                "diameter_R": community.diameter,
                "min_external_distance": community.min_external_distance,
                "separation_gap": community.min_external_distance - community.diameter,
                "voter_rows": ";".join(map(str, community.voter_rows)),
            }
            if expand_q:
                for quota in range(2, community.size + 1):
                    writer.writerow(common | {"q": quota})
            else:
                writer.writerow(common | {"q_min": 2, "q_max": community.size})


def write_partition(
    partition: list[ClearCommunity] | None, output: Path
) -> None:
    """Write one witness partition; an empty data section means none exists."""
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "partition_part", "partition_q_min", "partition_q_max", "size",
        "distinct_ballot_types", "diameter_R", "min_external_distance",
        "separation_gap", "voter_rows",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        partition_q_max = min(
            (community.size for community in partition or []), default=None
        )
        for part_number, community in enumerate(partition or [], start=1):
            writer.writerow(
                {
                    "partition_part": part_number,
                    "partition_q_min": 2,
                    "partition_q_max": partition_q_max,
                    "size": community.size,
                    "distinct_ballot_types": community.distinct_ballot_types,
                    "diameter_R": community.diameter,
                    "min_external_distance": community.min_external_distance,
                    "separation_gap": (
                        community.min_external_distance - community.diameter
                    ),
                    "voter_rows": ";".join(map(str, community.voter_rows)),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find every clear community in a voter-by-feature CSV."
    )
    parser.add_argument(
        "csv", type=Path, nargs="?", default=DEFAULT_CSV,
        help=f"headered voter matrix (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"output CSV (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--partition-output", type=Path, default=DEFAULT_PARTITION_OUTPUT,
        help=(
            "witness-partition CSV; contains only a header if no partition "
            f"exists (default: {DEFAULT_PARTITION_OUTPUT})"
        ),
    )
    parser.add_argument(
        "--expand-q", action="store_true",
        help="write one row per qualifying q instead of the compact range 2..size",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        _, voters = load_matrix(args.csv)
        communities = find_clear_communities(voters)
        partition = find_partition(communities, len(voters))
        write_communities(communities, args.output, args.expand_q)
        write_partition(partition, args.partition_output)
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1

    print(
        f"found {len(communities)} clear communities; wrote {args.output}"
    )
    if communities:
        print(
            "community sizes range from "
            f"{min(item.size for item in communities)} to "
            f"{max(item.size for item in communities)} voters"
        )
    if partition is None:
        covered_rows = {
            row for community in communities for row in community.voter_rows
        }
        print(
            "no subset of clear communities partitions the population "
            f"({len(voters) - len(covered_rows)} voters occur in no clear community); "
            f"wrote empty {args.partition_output}"
        )
    else:
        partition_q_max = min(community.size for community in partition)
        print(
            "a clear-community partition exists with "
            f"{len(partition)} parts for every integer q from 2 through "
            f"{partition_q_max}; wrote {args.partition_output}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
