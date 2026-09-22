"""Monte Carlo PF audit for the nested Greedy Capture group families.

At each distinct group family, draw one voter uniformly and independently from
every group and compute the exact PF approximation factor of the resulting set
of voter locations.  Copied levels share both groups and quota, so their trial
results are reused.  Candidate deviation centers are all voter locations;
duplicate locations are collapsed because they give identical PF ratios.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from nested_greedy_capture import DEFAULT_CSV, DEFAULT_OUTPUT, load_matrix


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / "matrices" / "pf_by_greedy_capture_level.csv"
OUTPUT_COLUMNS = (
    "k",
    "quota",
    "groups",
    "theoretical_alpha",
    "trials",
    "rho_min",
    "rho_p05",
    "rho_median",
    "rho_mean",
    "rho_p95",
    "rho_max",
    "fraction_1_pf",
    "infinite_trials",
    "mean_distinct_representatives",
)


@dataclass(frozen=True)
class TrialSummary:
    rhos: np.ndarray
    distinct_representatives: np.ndarray


def load_group_levels(path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Load JSON and materialize references used by copied levels."""
    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    raw_levels = document.get("levels")
    if not isinstance(raw_levels, list) or not raw_levels:
        raise ValueError(f"{path} does not contain a nonempty levels list")

    levels: list[dict[str, object]] = []
    current_groups: tuple[tuple[int, ...], ...] | None = None
    previous_k: int | None = None
    for raw in raw_levels:
        if not isinstance(raw, dict):
            raise ValueError("every level must be a JSON object")
        k = int(raw["k"])
        if previous_k is not None and k != previous_k - 1:
            raise ValueError("JSON levels must be ordered from n down to 1")
        if "groups" in raw:
            current_groups = tuple(tuple(int(i) for i in group) for group in raw["groups"])
        elif int(raw.get("same_groups_as_k", -1)) != k + 1 or current_groups is None:
            raise ValueError(f"G_{k} has an invalid copied-group reference")
        levels.append({**raw, "materialized_groups": current_groups})
        previous_k = k
    return document, levels


def distances_to_unique_locations(
    voters: np.ndarray, unique_locations: np.ndarray, metric: str
) -> np.ndarray:
    """Return voter-to-candidate distances, preserving compact integer Hamming data."""
    if metric == "hamming":
        return np.count_nonzero(
            voters[:, None, :] != unique_locations[None, :, :], axis=2
        ).astype(np.uint16)
    if metric == "euclidean":
        left_norms = np.einsum("ij,ij->i", voters, voters)
        right_norms = np.einsum("ij,ij->i", unique_locations, unique_locations)
        squared = left_norms[:, None] + right_norms[None, :] - 2 * voters @ unique_locations.T
        return np.sqrt(np.maximum(squared, 0.0))
    raise ValueError(f"unsupported metric in group JSON: {metric!r}")


def exact_pf_rho(
    distance_to_selected: np.ndarray,
    distance_to_candidates: np.ndarray,
    quota: int,
) -> float:
    """Return exact rho-PF using an explicit blocking-coalition quota."""
    n = len(distance_to_selected)
    if distance_to_candidates.shape[0] != n:
        raise ValueError("candidate distances have the wrong population size")
    if not 1 <= quota <= n:
        raise ValueError("quota must be between 1 and the population size")

    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.divide(
            distance_to_selected[:, None],
            distance_to_candidates,
            out=np.zeros(distance_to_candidates.shape, dtype=float),
            where=distance_to_candidates != 0,
        )
    ratios[
        (distance_to_candidates == 0) & (distance_to_selected[:, None] > 0)
    ] = np.inf
    quota_ratios = np.partition(ratios, n - quota, axis=0)[n - quota]
    return max(1.0, float(np.max(quota_ratios)))


def run_trials(
    groups: Sequence[Sequence[int]],
    quota: int,
    voter_type: np.ndarray,
    distance_to_candidates: np.ndarray,
    trials: int,
    rng: np.random.Generator,
) -> TrialSummary:
    """Sample representatives and exactly audit each resulting location set."""
    rhos = np.empty(trials, dtype=float)
    distinct = np.empty(trials, dtype=int)
    for trial in range(trials):
        representatives = np.asarray(
            [group[int(rng.integers(len(group)))] for group in groups],
            dtype=np.int64,
        )
        selected_types = np.unique(voter_type[representatives])
        distinct[trial] = len(selected_types)
        distance_to_selected = distance_to_candidates[:, selected_types].min(axis=1)
        rhos[trial] = exact_pf_rho(
            distance_to_selected, distance_to_candidates, quota
        )
    return TrialSummary(rhos, distinct)


def _number(value: float) -> str:
    return "infinity" if np.isinf(value) else f"{value:.12g}"


def summary_row(
    level: dict[str, object], summary: TrialSummary, trials: int
) -> dict[str, str | int]:
    """Convert raw trial results to one CSV row."""
    rhos = summary.rhos
    minimum = float(np.min(rhos))
    p05, median, p95 = (
        float(x) for x in np.percentile(rhos, [5, 50, 95], method="higher")
    )
    mean = float(np.mean(rhos))
    maximum = float(np.max(rhos))
    groups = level["materialized_groups"]
    return {
        "k": int(level["k"]),
        "quota": int(level["quota"]),
        "groups": len(groups),
        "theoretical_alpha": int(level["alpha"]),
        "trials": trials,
        "rho_min": _number(minimum),
        "rho_p05": _number(p05),
        "rho_median": _number(median),
        "rho_mean": _number(mean),
        "rho_p95": _number(p95),
        "rho_max": _number(maximum),
        "fraction_1_pf": f"{np.mean(rhos <= 1.0):.12g}",
        "infinite_trials": int(np.count_nonzero(np.isinf(rhos))),
        "mean_distinct_representatives": f"{np.mean(summary.distinct_representatives):.12g}",
    }


def experiment(
    group_json: Path,
    voter_csv: Path,
    output: Path,
    trials: int,
    seed: int,
) -> list[dict[str, str | int]]:
    """Run the experiment and write one summary row for every k."""
    document, levels = load_group_levels(group_json)
    _, voters = load_matrix(voter_csv)
    if int(document.get("population_size", -1)) != len(voters):
        raise ValueError("group JSON and voter CSV have different population sizes")
    metric = str(document.get("metric", "hamming"))

    unique_locations, voter_type = np.unique(voters, axis=0, return_inverse=True)
    candidate_distances = distances_to_unique_locations(voters, unique_locations, metric)
    rng = np.random.default_rng(seed)

    rows: list[dict[str, str | int]] = []
    cached_summary: TrialSummary | None = None
    for level in levels:
        if not bool(level.get("copied_from_next_level", False)):
            cached_summary = run_trials(
                level["materialized_groups"],
                int(level["quota"]),
                voter_type,
                candidate_distances,
                trials,
                rng,
            )
        if cached_summary is None:
            raise ValueError("first level cannot copy another level")
        rows.append(summary_row(level, cached_summary, trials))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly select one voter per nested group and audit exact PF."
    )
    parser.add_argument("--groups", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--voters", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    return args


def main() -> None:
    args = parse_args()
    rows = experiment(args.groups, args.voters, args.output, args.trials, args.seed)
    finite_maxima = [
        float(row["rho_max"])
        for row in rows
        if row["rho_max"] != "infinity"
    ]
    infinite_levels = sum(row["rho_max"] == "infinity" for row in rows)
    print(f"audited {len(rows)} levels with {args.trials} trials per distinct group family")
    print(f"largest finite observed rho: {max(finite_maxima):.6g}")
    print(f"levels with an infinite trial: {infinite_levels}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
