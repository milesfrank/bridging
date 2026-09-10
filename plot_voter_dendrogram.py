"""Plot voter clustering and report group-size metrics by cut height.

Each CSV row is one voter. Voters are clustered with complete linkage using
integer Hamming distance: the number of candidate approvals on which they
disagree. Joins through ``--max-height`` are collapsed into plotted groups.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage

ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "matrices" / "frenchapproval.csv"
DEFAULT_PLOT = ROOT / "matrices" / "voter_dendrogram.png"
DEFAULT_METRICS_CSV = ROOT / "matrices" / "voter_group_sizes_by_height.csv"

METRIC_HEADER = [
    "height", "groups", "smallest_group", "q1_group_size",
    "median_group_size", "mean_group_size", "q3_group_size",
    "largest_group", "std_group_size", "coefficient_of_variation",
    "largest_to_smallest_ratio", "largest_group_percent", "gini_coefficient",
]


def load_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    """Load and validate a headered binary voter-by-candidate CSV."""
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


def groups_at_height(
    linkage_matrix: np.ndarray, n_voters: int, height: int
) -> tuple[list[int], np.ndarray]:
    """Return active node IDs and descending group sizes after a height cut."""
    active = set(range(n_voters))
    for i, merge in enumerate(linkage_matrix):
        if float(merge[2]) > height:
            break
        left, right = int(merge[0]), int(merge[1])
        active.remove(left)
        active.remove(right)
        active.add(n_voters + i)

    node_ids = sorted(active)
    sizes = np.asarray(
        [
            1 if node < n_voters else int(linkage_matrix[node - n_voters, 3])
            for node in node_ids
        ],
        dtype=int,
    )
    return node_ids, np.sort(sizes)[::-1]


def gini_coefficient(values: np.ndarray) -> float:
    """Return 0 for equal-sized groups and values approaching 1 for imbalance."""
    ordered = np.sort(values.astype(float))
    n = len(ordered)
    weighted_sum = np.sum(np.arange(1, n + 1) * ordered)
    return float(2 * weighted_sum / (n * ordered.sum()) - (n + 1) / n)


def group_size_metrics(height: int, sizes: np.ndarray) -> tuple[float, ...]:
    """Summarize the relative sizes of all groups at one cut height."""
    q1, median, q3 = np.percentile(sizes, [25, 50, 75])
    mean = float(np.mean(sizes))
    std = float(np.std(sizes))
    smallest = int(np.min(sizes))
    largest = int(np.max(sizes))
    return (
        height, len(sizes), smallest, float(q1), float(median), mean,
        float(q3), largest, std, std / mean if mean else 0.0,
        largest / smallest, 100.0 * largest / int(np.sum(sizes)),
        gini_coefficient(sizes),
    )


def metrics_by_height(
    linkage_matrix: np.ndarray, n_voters: int, through_height: int = 0
) -> list[tuple[float, ...]]:
    """Calculate metrics at every attainable integer distance level."""
    maximum = max(through_height, int(math.ceil(float(linkage_matrix[-1, 2]))))
    return [
        group_size_metrics(h, groups_at_height(linkage_matrix, n_voters, h)[1])
        for h in range(maximum + 1)
    ]


def print_metrics(rows: list[tuple[float, ...]], selected_height: int) -> None:
    """Print a compact comparison of group sizes across heights."""
    print("\ngroup sizes by cut height:")
    print("height  groups     min      q1  median    mean      q3     max  max/min  largest%    CV   Gini")
    for row in rows:
        height, groups, minimum, q1, median, mean, q3, maximum, _, cv, ratio, share, gini = row
        marker = "*" if height == selected_height else " "
        print(
            f"{marker}{int(height):5d} {int(groups):7d} {int(minimum):7d} "
            f"{q1:7.1f} {median:7.1f} {mean:7.1f} {q3:7.1f} "
            f"{int(maximum):7d} {ratio:8.1f} {share:8.2f}% {cv:5.2f} {gini:6.3f}"
        )
    print("* selected plotting height; CV and Gini are 0 when groups are equal-sized")


def write_metrics(rows: list[tuple[float, ...]], out_path: Path) -> None:
    """Write the height-level group-size comparison to CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(METRIC_HEADER)
        for row in rows:
            writer.writerow(
                [int(row[0]), int(row[1]), int(row[2])]
                + [f"{value:.4f}" for value in row[3:7]]
                + [int(row[7])]
                + [f"{value:.4f}" for value in row[8:]]
            )
    print(f"wrote {out_path}")


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    n_voters: int,
    max_height: int,
    out_path: Path,
    show: bool,
) -> None:
    """Plot the hierarchy with groups through max_height collapsed to leaves."""
    _, sizes = groups_at_height(linkage_matrix, n_voters, max_height)
    n_groups = len(sizes)
    width = min(24.0, max(12.0, 0.12 * n_groups))
    fig, ax = plt.subplots(figsize=(width, 7.5))

    def group_label(node_id: int) -> str:
        size = (
            1
            if node_id < n_voters
            else int(linkage_matrix[node_id - n_voters, 3])
        )
        return f"n={size}"

    dendrogram(
        linkage_matrix,
        truncate_mode="lastp",
        p=n_groups,
        show_leaf_counts=True,
        show_contracted=True,
        leaf_label_func=group_label,
        leaf_rotation=90,
        leaf_font_size=max(4, min(8, 700 / n_groups)),
        color_threshold=max_height,
        above_threshold_color="#4c4c4c",
        ax=ax,
    )
    ax.axhline(
        max_height, color="#b22222", linewidth=1.2, linestyle="--",
        label=f"cut height = {max_height}",
    )
    ax.set_title(
        f"Complete-linkage voter dendrogram: {n_groups} groups at height {max_height}"
    )
    ax.set_xlabel("Terminal group size (number of voters)")
    ax.set_ylabel("Number of candidate approvals that differ")
    ax.legend(loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    print(f"wrote {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot complete-linkage voter groups and report their sizes."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_PLOT)
    parser.add_argument(
        "--metrics-csv", type=Path, default=DEFAULT_METRICS_CSV,
        help="output CSV containing group-size metrics at every height",
    )
    parser.add_argument(
        "--max-height", type=int, default=4, metavar="HEIGHT",
        help="collapse joins at or below this integer Hamming distance (default: 4)",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if args.max_height < 0:
        parser.error("--max-height must be a nonnegative integer")
    names, matrix = load_matrix(args.csv)
    if args.max_height > len(names):
        parser.error("--max-height cannot exceed the number of candidates")
    print(f"loaded {matrix.shape[0]} voters x {len(names)} candidates")
    print("computing complete-linkage clustering with integer Hamming distance ...")
    linkage_matrix = linkage(matrix, method="complete", metric="cityblock")

    rows = metrics_by_height(linkage_matrix, matrix.shape[0], args.max_height)
    print_metrics(rows, args.max_height)
    _, selected_sizes = groups_at_height(linkage_matrix, matrix.shape[0], args.max_height)
    print(f"\nselected group sizes, largest to smallest:\n  {selected_sizes.tolist()}")
    write_metrics(rows, args.metrics_csv)
    plot_dendrogram(linkage_matrix, matrix.shape[0], args.max_height, args.out, args.show)


if __name__ == "__main__":
    main()
