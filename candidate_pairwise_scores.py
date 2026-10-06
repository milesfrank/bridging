"""Score candidates from opposing voter sets induced by distinct approval columns.

For every unordered pair of distinct candidate approval sets S,T, let
U = S - T and V = T - S. Candidate c receives
min(|V| * |A(c) intersect U|, |U| * |A(c) intersect V|) / n**2.
Sum over all such pairs, with no normalization by the number of pairs.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from cluster_candidate_scores import load_input, validate_matrix


def candidate_pairwise_scores(matrix: np.ndarray) -> np.ndarray:
    """Return B(c,E) in original candidate-column order.

    Each row represents one voter. Duplicate candidate approval sets occur
    only once among comparison sets, but every original candidate gets a
    score. Nested sets contribute zero because U or V is empty. A single
    distinct approval set therefore gives all-zero scores.

    For q distinct columns, time is O(q**2 * n * m); no n-by-n voter distance
    matrix, clustering, or cutoff is needed.
    """
    matrix = validate_matrix(matrix)
    approval_sets = np.unique(matrix.T, axis=0).astype(bool)
    numerators = np.zeros(matrix.shape[1], dtype=float)
    for i, s in enumerate(approval_sets):
        for t in approval_sets[i + 1:]:
            u, v = s & ~t, t & ~s
            u_size, v_size = int(u.sum()), int(v.sum())
            if not u_size or not v_size:
                continue
            u_approvals = matrix[u].sum(axis=0)
            v_approvals = matrix[v].sum(axis=0)
            numerators += np.minimum(v_size * u_approvals, u_size * v_approvals)
    return numerators / len(matrix) ** 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path,
                        help="Headered binary CSV, PrefLib .cat, or directory of .cat files")
    parser.add_argument("--output", type=Path, help="Write descending candidate scores to CSV")
    args = parser.parse_args()
    names, matrix = load_input(args.input)
    scores = candidate_pairwise_scores(matrix)
    q = len(np.unique(matrix.T, axis=0))
    print(f"{len(matrix)} voters, {len(names)} candidates, {q} distinct approval sets")
    print(f"{q * (q - 1) // 2} unordered comparisons; normalization n^2 = {len(matrix) ** 2}")
    rows = [(int(c), names[c], float(scores[c])) for c in np.argsort(-scores, kind="stable")]
    for _, name, score in rows:
        print(f"{name}: {score:.12g}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("candidate_id", "candidate", "score"))
            writer.writerows(rows)


if __name__ == "__main__":
    main()
