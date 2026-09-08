"""Compute candidate coverage of a hierarchical clustering of voters.

Each CSV row is treated as one voter. Clustering uses complete linkage and
Hamming distance (the fraction of candidates on which two voters disagree).
Coverage is reported for the final groups and for those groups plus every
merge node above them in the tree.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage

ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "matrices" / "frenchapproval.csv"
DEFAULT_COVER_CSV = ROOT / "matrices" / "candidate_group_cover.csv"


def load_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    """Load a headered binary voter-by-candidate CSV."""
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            candidate_names = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{path} is empty") from exc

        rows = [row for row in reader if row]

    if not candidate_names:
        raise ValueError(f"{path} has no candidate columns")
    if len(rows) < 2:
        raise ValueError("At least two voters are required for clustering")
    if any(len(row) != len(candidate_names) for row in rows):
        raise ValueError("Every voter row must have one value per candidate")

    try:
        matrix = np.asarray(rows, dtype=np.uint8)
    except ValueError as exc:
        raise ValueError("The matrix must contain only numeric 0/1 values") from exc
    if not np.all((matrix == 0) | (matrix == 1)):
        raise ValueError("The matrix must contain only 0/1 values")
    return candidate_names, matrix


def compute_group_cover(
    names: list[str],
    matrix: np.ndarray,
    linkage_matrix: np.ndarray,
    n_groups: int,
) -> list[tuple[str, int, int, int, float, int, int, float]]:
    """Count covered leaf groups and all nodes in the displayed tree.

    The displayed tree has ``n_groups`` leaf groups and ``n_groups - 1``
    internal merge nodes. A node is covered when at least one voter below it
    approves the candidate.
    """
    n_voters, n_candidates = matrix.shape

    # Record candidate coverage for every original leaf and every merge node.
    node_covered = np.zeros((2 * n_voters - 1, n_candidates), dtype=bool)
    node_covered[:n_voters] = matrix.astype(bool)
    for i, merge in enumerate(linkage_matrix):
        left, right = int(merge[0]), int(merge[1])
        node_covered[n_voters + i] = node_covered[left] | node_covered[right]

    # After the first n - p merges, these active nodes are exactly the p leaf
    # groups shown by dendrogram(..., truncate_mode="lastp", p=p).
    leaf_nodes = set(range(n_voters))
    early_merges = n_voters - n_groups
    for i in range(early_merges):
        left, right = map(int, linkage_matrix[i, :2])
        leaf_nodes.remove(left)
        leaf_nodes.remove(right)
        leaf_nodes.add(n_voters + i)

    leaf_node_ids = sorted(leaf_nodes)
    higher_node_ids = list(range(n_voters + early_merges, 2 * n_voters - 1))
    displayed_node_ids = leaf_node_ids + higher_node_ids
    total_tree_nodes = 2 * n_groups - 1
    if len(displayed_node_ids) != total_tree_nodes:
        raise RuntimeError("Could not reconstruct the displayed dendrogram nodes")

    leaf_covered = node_covered[leaf_node_ids].sum(axis=0)
    tree_covered = node_covered[displayed_node_ids].sum(axis=0)

    approvers = matrix.sum(axis=0)
    return [
        (
            name,
            int(approvers[i]),
            int(leaf_covered[i]),
            n_groups,
            100.0 * float(leaf_covered[i]) / n_groups,
            int(tree_covered[i]),
            total_tree_nodes,
            100.0 * float(tree_covered[i]) / total_tree_nodes,
        )
        for i, name in enumerate(names)
    ]


def write_group_cover(
    rows: list[tuple[str, int, int, int, float, int, int, float]], out_path: Path
) -> None:
    """Write candidate group-cover counts and percentages to CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "candidate",
                "approvers",
                "leaf_groups_covered",
                "total_leaf_groups",
                "leaf_cover_percent",
                "tree_nodes_covered",
                "total_tree_nodes",
                "tree_cover_percent",
            ]
        )
        for row in rows:
            candidate, approvers, leaf, leaves, leaf_pct, tree, nodes, tree_pct = row
            writer.writerow(
                [
                    candidate,
                    approvers,
                    leaf,
                    leaves,
                    f"{leaf_pct:.2f}",
                    tree,
                    nodes,
                    f"{tree_pct:.2f}",
                ]
            )
    print(f"wrote {out_path}")


def print_group_cover(
    rows: list[tuple[str, int, int, int, float, int, int, float]],
) -> None:
    """Print a compact candidate cover table."""
    print("candidate group cover:")
    for row in rows:
        candidate, approvers, leaf, leaves, leaf_pct, tree, nodes, tree_pct = row
        print(
            f"  {candidate:15s} leaves {leaf:3d}/{leaves:<3d} ({leaf_pct:6.2f}%), "
            f"whole tree {tree:3d}/{nodes:<3d} ({tree_pct:6.2f}%), "
            f"{approvers:4d} approvers"
        )


def sort_group_cover(
    rows: list[tuple[str, int, int, int, float, int, int, float]],
    sort_by: str,
    descending: bool,
) -> list[tuple[str, int, int, int, float, int, int, float]]:
    """Return coverage rows sorted by the requested field."""
    if sort_by == "input":
        return rows
    field = {
        "candidate": 0,
        "approvers": 1,
        "leaf-cover": 4,
        "tree-cover": 7,
    }[sort_by]
    return sorted(
        rows,
        key=lambda row: row[field].casefold() if field == 0 else row[field],
        reverse=descending,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute candidate coverage of complete-linkage voter groups "
            "using Hamming distance."
        )
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--cover-csv",
        type=Path,
        default=DEFAULT_COVER_CSV,
        help="output CSV for candidate coverage of the plotted groups",
    )
    parser.add_argument(
        "--groups",
        "--truncate",
        dest="groups",
        type=int,
        default=80,
        metavar="GROUPS",
        help="number of terminal groups; use 0 for individual voters (default: 80)",
    )
    parser.add_argument(
        "--sort",
        choices=("input", "candidate", "approvers", "leaf-cover", "tree-cover"),
        default="input",
        help="field used to sort terminal and CSV output (default: input order)",
    )
    parser.add_argument(
        "--descending",
        action="store_true",
        help="sort the selected field from greatest to least",
    )
    args = parser.parse_args()

    if args.groups < 0:
        parser.error("--groups must be zero or positive")

    names, matrix = load_matrix(args.csv)
    if args.groups > matrix.shape[0]:
        parser.error("--groups cannot exceed the number of voters")
    print(f"loaded {matrix.shape[0]} voters x {len(names)} candidates")
    print("computing complete-linkage clustering with Hamming distance ...")
    linkage_matrix = linkage(matrix, method="complete", metric="hamming")

    n_groups = args.groups or matrix.shape[0]
    cover_rows = compute_group_cover(names, matrix, linkage_matrix, n_groups)
    cover_rows = sort_group_cover(cover_rows, args.sort, args.descending)
    print_group_cover(cover_rows)
    write_group_cover(cover_rows, args.cover_csv)


if __name__ == "__main__":
    main()
