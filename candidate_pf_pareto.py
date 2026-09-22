"""Audit candidate fairness across independent entitlement parameters k.

The quota ceil(n/k) is constant on each reported inclusive k interval.
Dominance always compares one fixed challenger with one fixed target over
the entire curve, and excludes self comparisons.

This uses rho-PF from proportional_audit.py, not the passing fractions in
representation_level_audit.py. For candidate c, X is its approvers and M is
the electorate with coordinate c forced to 1. At quota q = ceil(n/k), rho
is max(1, max_y q-th-largest_i D(i,X)/d(i,y)). All voters participate.
The default Euclidean distance matches proportional_audit.py on binary data;
--metric hamming uses the metric from representation_level_audit.py instead.
Unlike the existing proportional audit, k is independent of the size of X.
An empty approver set is assigned infinity at every quota.

Examples:
    python candidate_pf_pareto.py
    python candidate_pf_pareto.py --winner "Method A=Chirac" --winner "Method B=Jospin"

With no --winner, report comparisons for every candidate. Smaller rho is
better. The dominance CSV includes all challenger/target pairs, including
failed dominance checks; weak dominance allows ties at every quota.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from generic_dc_mpjr_min_gamma import load_matrix
from proportional_audit import resolve_candidate

ROOT = Path(__file__).resolve().parent


def quota_intervals(n: int) -> list[tuple[int, int, int]]:
    """Return (first k, last k, ceil(n/k)), covering all integers 1..n."""
    if n < 1:
        raise ValueError("population size must be positive")
    intervals = []
    first = 1
    while first <= n:
        quota = (n + first - 1) // first
        last = n if quota == 1 else (n - 1) // (quota - 1)
        intervals.append((first, last, quota))
        first = last + 1
    return intervals


def dominance(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return weak and strict-improvement dominance matrices (lower is better).

    Entry [a, b] means a dominates b. Weak dominance includes equal curves;
    strict-improvement dominance additionally requires improvement somewhere.
    Positive infinity is supported, with infinity equal to itself.
    """
    curves = np.asarray(curves, dtype=float)
    if curves.ndim != 2 or curves.shape[1] == 0 or np.any(np.isnan(curves)):
        raise ValueError("curves must be a nonempty-width matrix without NaNs")
    weak = np.all(curves[:, None, :] <= curves[None, :, :], axis=2)
    strict = weak & np.any(curves[:, None, :] < curves[None, :, :], axis=2)
    np.fill_diagonal(weak, False)
    return weak, strict


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def candidate_curves(voters: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Compute candidate-by-quota rho values using weighted unique ballots.

    Each unique ballot retains its voter multiplicity in coalition order
    statistics. Deviating centers are processed individually to avoid an
    electorate-squared distance matrix. Sorting each ratio vector once
    supplies every quota, including k values larger than the approver count.
    """
    voters = np.asarray(voters)
    if voters.ndim != 2 or min(voters.shape) == 0:
        raise ValueError("voters must be a nonempty two-dimensional matrix")
    if not np.all((voters == 0) | (voters == 1)):
        raise ValueError("voter matrix must be binary")
    if metric not in ("euclidean", "hamming"):
        raise ValueError("metric must be euclidean or hamming")
    locations, counts = np.unique(voters, axis=0, return_counts=True)
    quotas = np.array([q for _, _, q in quota_intervals(len(voters))])
    curves = np.ones((voters.shape[1], len(quotas)))
    for column in range(voters.shape[1]):
        approved = locations[:, column] == 1
        if not np.any(approved):
            curves[column] = np.inf
            continue
        nearest = np.full(len(locations), np.inf)
        for center in locations[approved]:
            nearest = np.minimum(nearest, np.count_nonzero(locations != center, axis=1))
        if metric == "euclidean":
            nearest = np.sqrt(nearest)
        potential = locations.copy()
        potential[:, column] = 1
        for center in np.unique(potential, axis=0):
            distance = np.count_nonzero(locations != center, axis=1).astype(float)
            if metric == "euclidean":
                distance = np.sqrt(distance)
            ratios = np.divide(nearest, distance, out=np.zeros_like(nearest),
                               where=distance != 0)
            ratios[(distance == 0) & (nearest > 0)] = np.inf
            order = np.argsort(-ratios)
            ranks = np.searchsorted(np.cumsum(counts[order]), quotas, side="left")
            curves[column] = np.maximum(curves[column], ratios[order[ranks]])
    return curves


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path, nargs="?",
                        default=ROOT / "matrices" / "frenchapproval.csv")
    parser.add_argument("--metric", choices=("euclidean", "hamming"), default="euclidean")
    parser.add_argument("--winner", action="append", default=[], metavar="METHOD=CANDIDATE",
                        help="Repeat for each method's winner (or each tied winner)")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "matrices" / "candidate_pf_by_k.csv")
    parser.add_argument("--dominance-output", type=Path,
                        default=ROOT / "matrices" / "candidate_pf_dominance.csv")
    args = parser.parse_args()
    try:
        names, voters = load_matrix(args.csv)
        if len(set(names)) != len(names):
            raise ValueError("candidate names must be unique")
        targets = []
        for item in args.winner:
            method, separator, candidate = item.partition("=")
            if not separator or not method.strip() or not candidate.strip():
                raise ValueError("--winner must have the form METHOD=CANDIDATE")
            targets.append((method.strip(), resolve_candidate(names, candidate.strip())))
        if not targets:
            targets = [("", i) for i in range(len(names))]
        if args.output.resolve() == args.dominance_output.resolve():
            raise ValueError("output files must have different paths")
        if args.csv.resolve() in (args.output.resolve(), args.dominance_output.resolve()):
            raise ValueError("output files must not overwrite the input matrix")
        curves = candidate_curves(voters, args.metric)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    intervals = quota_intervals(len(voters))
    rows = []
    for column, name in enumerate(names):
        for index, (first, last, quota) in enumerate(intervals):
            rows.append(dict(candidate=name, approvers=int(voters[:, column].sum()),
                             population_size=len(voters), metric=args.metric,
                             k_min=first, k_max=last, quota=quota,
                             rho=curves[column, index]))
    write_csv(args.output, ["candidate", "approvers", "population_size", "metric",
                            "k_min", "k_max", "quota", "rho"], rows)
    weak, strict = dominance(curves)
    comparisons = []
    for method, target in targets:
        challengers = np.flatnonzero(weak[:, target])
        for challenger in range(len(names)):
            if challenger == target:
                continue
            comparisons.append(dict(method=method, winner=names[target],
                                    challenger=names[challenger],
                                    weakly_dominates=bool(weak[challenger, target]),
                                    strictly_better_somewhere=bool(strict[challenger, target]),
                                    equal_curve=bool(np.array_equal(curves[challenger], curves[target]))))
        label = f"{method}: " if method else ""
        print(f"{label}{names[target]}: weak dominators: "
              + (", ".join(names[i] for i in challengers) or "none"))
    write_csv(args.dominance_output,
              ["method", "winner", "challenger", "weakly_dominates",
               "strictly_better_somewhere", "equal_curve"], comparisons)
    print(f"Audited {len(names)} candidates across {len(intervals)} quotas "
          f"covering k=1..{len(voters)} ({args.metric}).")
    print(f"Wrote {args.output} and {args.dominance_output}")


if __name__ == "__main__":
    main()
